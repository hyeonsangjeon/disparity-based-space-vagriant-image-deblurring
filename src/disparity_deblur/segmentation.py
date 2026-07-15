import cv2
import maxflow
import numpy as np

from .models import SegmentationResult


def disparity_segmentation(
    blurred: np.ndarray,
    feature_points: np.ndarray,
    disparities: np.ndarray,
    *,
    color_labels: int = 12,
    graph_cut_size: int = 384,
    graph_cut_smoothness: float = 0.35,
    disparity_threshold: float = 1.5,
    minimum_disparity_features: int = 1,
    max_regions: int = 6,
    minimum_region_fraction: float = 0.01,
) -> SegmentationResult:
    """Create color regions and merge adjacent regions with similar disparities."""

    if blurred.ndim != 3 or blurred.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {blurred.shape}")
    if (
        feature_points.ndim != 2
        or feature_points.shape[1:] != (2,)
        or disparities.shape != feature_points.shape
    ):
        raise ValueError("feature points and disparities must both have shape Nx2")
    initial = graph_cut_oversegmentation(
        blurred,
        color_labels=color_labels,
        max_size=graph_cut_size,
        smoothness=graph_cut_smoothness,
    )
    vectors, counts = _representative_disparities(
        initial,
        feature_points,
        disparities,
        minimum_features=minimum_disparity_features,
    )
    merged = _merge_by_disparity(
        initial,
        blurred,
        vectors,
        disparity_threshold=disparity_threshold,
        max_regions=max_regions,
        minimum_region_fraction=minimum_region_fraction,
    )
    merged_vectors, merged_counts = _representative_disparities(
        merged,
        feature_points,
        disparities,
        minimum_features=minimum_disparity_features,
    )
    return SegmentationResult(
        initial_labels=initial,
        merged_labels=merged,
        region_disparities=merged_vectors,
        region_feature_counts=merged_counts,
    )


def graph_cut_oversegmentation(
    image: np.ndarray,
    *,
    color_labels: int,
    max_size: int,
    smoothness: float,
) -> np.ndarray:
    """Oversegment an RGB image while preserving its original aspect ratio."""

    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    small_width = max(16, int(round(width * scale)))
    small_height = max(16, int(round(height * scale)))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    rgb_u8 = np.rint(np.clip(small, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    yy, xx = np.mgrid[:small_height, :small_width]
    spatial = np.stack(
        (xx / max(small_width - 1, 1), yy / max(small_height - 1, 1)),
        axis=-1,
    ).astype(np.float32)
    features = np.concatenate((lab, spatial * 0.08), axis=-1)
    samples = features.reshape(-1, features.shape[-1])

    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    _, seeds, centers = cv2.kmeans(
        samples,
        color_labels,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    distances = np.sum(
        (features[..., None, :] - centers[None, None, ...]) ** 2,
        axis=-1,
        dtype=np.float64,
    )
    normalizer = float(np.median(np.min(distances, axis=-1)))
    distances /= max(normalizer, 1e-6)
    pairwise = np.full((color_labels, color_labels), smoothness, dtype=np.float64)
    np.fill_diagonal(pairwise, 0.0)
    labels = maxflow.fastmin.aexpansion_grid(
        np.ascontiguousarray(distances, dtype=np.float64),
        pairwise,
        labels=seeds.reshape(small_height, small_width).astype(np.int32),
        max_cycles=4,
    )
    components = _split_connected_components(labels.astype(np.int32))
    full = cv2.resize(
        components.astype(np.int32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return _relabel(full)


def _representative_disparities(
    labels: np.ndarray,
    feature_points: np.ndarray,
    disparities: np.ndarray,
    *,
    minimum_features: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign robust median feature displacement to each content region."""

    region_count = int(labels.max()) + 1
    vectors = np.full((region_count, 2), np.nan, dtype=np.float64)
    counts = np.zeros(region_count, dtype=np.int32)
    if len(feature_points) == 0:
        vectors[:] = 0.0
        return vectors, counts

    boundary = np.zeros(labels.shape, dtype=np.uint8)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary = cv2.dilate(boundary, np.ones((3, 3), dtype=np.uint8))

    height, width = labels.shape
    assignments: list[list[np.ndarray]] = [[] for _ in range(region_count)]
    for point, disparity in zip(feature_points, disparities, strict=True):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height and boundary[y, x] == 0:
            assignments[int(labels[y, x])].append(disparity)

    for region, values in enumerate(assignments):
        counts[region] = len(values)
        if len(values) >= minimum_features:
            array = np.asarray(values, dtype=np.float64)
            vectors[region] = np.median(array, axis=0)

    known = np.flatnonzero(np.isfinite(vectors[:, 0]))
    if len(known) == 0:
        vectors[:] = np.median(disparities, axis=0)
        return vectors, counts

    centroids = _region_centroids(labels, region_count)
    for region in np.flatnonzero(~np.isfinite(vectors[:, 0])):
        nearest = known[
            np.argmin(np.linalg.norm(centroids[known] - centroids[region], axis=1))
        ]
        vectors[region] = vectors[nearest]
    return vectors, counts


def _merge_by_disparity(
    labels: np.ndarray,
    image: np.ndarray,
    disparities: np.ndarray,
    *,
    disparity_threshold: float,
    max_regions: int,
    minimum_region_fraction: float,
) -> np.ndarray:
    """Merge adjacent regions by disparity, color, area, and region-count limits."""

    region_count = int(labels.max()) + 1
    disparity_map = disparities[labels]
    parent = np.arange(region_count)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in _adjacent_pairs(labels):
        if np.linalg.norm(disparities[left] - disparities[right]) <= disparity_threshold:
            union(left, right)

    merged = np.vectorize(find, otypes=[np.int32])(labels)
    merged = _relabel(merged)
    minimum_area = max(1, int(round(labels.size * minimum_region_fraction)))

    while True:
        region_count = int(merged.max()) + 1
        areas = np.bincount(merged.ravel(), minlength=region_count)
        if region_count <= max_regions and areas.min() >= minimum_area:
            break
        candidates = np.flatnonzero(areas < minimum_area)
        source = int(candidates[np.argmin(areas[candidates])]) if len(candidates) else int(
            np.argmin(areas)
        )
        neighbors = _neighbors_of(merged, source)
        if not neighbors:
            break
        vectors = _region_mean_vectors(merged, disparity_map)
        colors = _region_mean_colors(merged, image)
        color_scale = max(
            np.median(
                [
                    np.linalg.norm(colors[a] - colors[b])
                    for a, b in _adjacent_pairs(merged)
                ]
            ),
            1e-6,
        )
        disparity_scale = max(disparity_threshold, 1e-6)
        target = min(
            neighbors,
            key=lambda neighbor: (
                np.linalg.norm(vectors[source] - vectors[neighbor])
                / disparity_scale
                + np.linalg.norm(colors[source] - colors[neighbor]) / color_scale
            ),
        )
        merged[merged == source] = target
        merged = _relabel(merged)
        if int(merged.max()) + 1 == 1:
            break
    return merged


def _split_connected_components(labels: np.ndarray) -> np.ndarray:
    """Give disconnected components independent contiguous labels."""

    output = np.full(labels.shape, -1, dtype=np.int32)
    next_label = 0
    for value in np.unique(labels):
        count, components = cv2.connectedComponents(
            (labels == value).astype(np.uint8), connectivity=8
        )
        for component in range(1, count):
            output[components == component] = next_label
            next_label += 1
    return output


def _adjacent_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Return unordered region pairs that share a horizontal or vertical edge."""

    pairs: set[tuple[int, int]] = set()
    for left, right in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        changed = left != right
        for a, b in zip(left[changed], right[changed], strict=True):
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    return pairs


def _neighbors_of(labels: np.ndarray, region: int) -> set[int]:
    """Return all regions directly adjacent to one region."""

    neighbors: set[int] = set()
    for left, right in _adjacent_pairs(labels):
        if left == region:
            neighbors.add(right)
        elif right == region:
            neighbors.add(left)
    return neighbors


def _region_centroids(labels: np.ndarray, count: int) -> np.ndarray:
    """Compute region centroids in x, y coordinate order."""

    yy, xx = np.indices(labels.shape)
    areas = np.bincount(labels.ravel(), minlength=count).clip(min=1)
    x = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=count) / areas
    y = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=count) / areas
    return np.stack((x, y), axis=1)


def _region_mean_colors(labels: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Compute one RGB mean for each region."""

    count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=count).clip(min=1)
    channels = [
        np.bincount(
            labels.ravel(), weights=image[..., channel].ravel(), minlength=count
        )
        / areas
        for channel in range(3)
    ]
    return np.stack(channels, axis=1)


def _region_mean_vectors(labels: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute one mean vector for each region."""

    count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=count).clip(min=1)
    components = [
        np.bincount(
            labels.ravel(), weights=vectors[..., component].ravel(), minlength=count
        )
        / areas
        for component in range(vectors.shape[2])
    ]
    return np.stack(components, axis=1)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to contiguous zero-based integers."""

    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)
