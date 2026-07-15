"""Manifest-driven, deterministic benchmark support for the restoration pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .config import PipelineConfig
from .image_io import read_input_image, write_png
from .pipeline import DisparityDeblurPipeline

Processor = Callable[[np.ndarray, np.ndarray, PipelineConfig], np.ndarray]


@dataclass(frozen=True)
class InputFile:
    """Manifest-relative input path and its expected SHA-256 digest."""

    path: str
    sha256: str


@dataclass(frozen=True)
class Dataset:
    """One reproducible benchmark pair and its publication metadata."""

    identifier: str
    description: str
    visibility: str
    rights: str | None
    blurred: InputFile
    noisy: InputFile
    reference: InputFile | None = None
    source_hashes: Mapping[str, str] | None = None
    noisy_label: str | None = None
    input_processing: Mapping[str, object] | None = None
    config_overrides: Mapping[str, object] | None = None


@dataclass(frozen=True)
class BenchmarkManifest:
    """Validated collection of benchmark datasets."""

    schema_version: int
    datasets: tuple[Dataset, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    """One HPO configuration with measured quality and runtime."""

    config: PipelineConfig
    metrics: Mapping[str, float | str | bool]
    runtime_seconds: float


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load a benchmark manifest without resolving any local data paths."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("benchmark manifest schema_version must be 1")
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("benchmark manifest must contain a non-empty datasets array")

    datasets: list[Dataset] = []
    identifiers: set[str] = set()
    for raw in raw_datasets:
        if not isinstance(raw, dict):
            raise ValueError("each benchmark dataset must be an object")
        identifier = _required_string(raw, "id")
        if identifier in identifiers:
            raise ValueError(f"duplicate benchmark dataset id: {identifier}")
        identifiers.add(identifier)
        visibility = _required_string(raw, "visibility")
        if visibility not in {"public", "private"}:
            raise ValueError(f"{identifier}: visibility must be public or private")
        source_hashes = raw.get("source_hashes")
        if source_hashes is not None and (
            not isinstance(source_hashes, dict)
            or not all(
                isinstance(key, str) and _is_sha256(value)
                for key, value in source_hashes.items()
            )
        ):
            raise ValueError(f"{identifier}: source_hashes must contain SHA-256 values")
        datasets.append(
            Dataset(
                identifier=identifier,
                description=_required_string(raw, "description"),
                visibility=visibility,
                rights=_optional_string(raw, "rights"),
                blurred=_parse_input(raw, identifier, "blurred"),
                noisy=_parse_input(raw, identifier, "noisy"),
                reference=(
                    _parse_input(raw, identifier, "reference")
                    if "reference" in raw
                    else None
                ),
                source_hashes=source_hashes,
                noisy_label=_optional_string(raw, "noisy_label"),
                input_processing=_optional_mapping(raw, identifier, "input_processing"),
                config_overrides=_parse_config_overrides(raw, identifier),
            )
        )
    return BenchmarkManifest(schema_version=1, datasets=tuple(datasets))


def deterministic_noise(
    image: np.ndarray,
    *,
    seed: int,
    sigma: float,
) -> np.ndarray:
    """Apply reproducible zero-mean RGB Gaussian noise to a normalized image."""
    if sigma < 0:
        raise ValueError("sigma cannot be negative")
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0.0, 1.0)


def image_metrics(result: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return PSNR, SSIM, edge fidelity, and a ringing/saturation artifact penalty."""
    _validate_pair(result, reference)
    result64 = result.astype(np.float64)
    reference64 = reference.astype(np.float64)
    mse = float(np.mean((result64 - reference64) ** 2))
    psnr = 99.0 if mse <= 1e-14 else float(10.0 * np.log10(1.0 / mse))

    result_luma = _luminance(result64)
    reference_luma = _luminance(reference64)
    blur_result = cv2.GaussianBlur(result_luma, (0, 0), 1.5)
    blur_reference = cv2.GaussianBlur(reference_luma, (0, 0), 1.5)
    c1, c2 = 0.01**2, 0.03**2
    mu_product = blur_result * blur_reference
    covariance = cv2.GaussianBlur(result_luma * reference_luma, (0, 0), 1.5)
    covariance -= mu_product
    variance_result = cv2.GaussianBlur(result_luma**2, (0, 0), 1.5) - blur_result**2
    variance_reference = (
        cv2.GaussianBlur(reference_luma**2, (0, 0), 1.5) - blur_reference**2
    )
    ssim_map = ((2.0 * mu_product + c1) * (2.0 * covariance + c2)) / (
        (blur_result**2 + blur_reference**2 + c1)
        * (variance_result + variance_reference + c2)
    )
    ssim = float(np.clip(np.mean(ssim_map), -1.0, 1.0))

    result_edges = _gradient_magnitude(result_luma)
    reference_edges = _gradient_magnitude(reference_luma)
    edge_error = np.mean(np.abs(result_edges - reference_edges))
    edge_scale = np.mean(reference_edges) + 1e-6
    edge_fidelity = float(np.clip(1.0 - edge_error / (2.0 * edge_scale), 0.0, 1.0))
    result_high_frequency = float(np.var(result_luma - blur_result))
    reference_high_frequency = float(np.var(reference_luma - blur_reference))
    excess_detail = max(
        0.0, result_high_frequency / max(reference_high_frequency, 1e-8) - 2.0
    )
    saturation = float(np.mean((result64 < 0.002) | (result64 > 0.998)))
    artifact_penalty = min(1.0, 0.08 * excess_detail + 0.2 * saturation)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "edge_fidelity": edge_fidelity,
        "artifact_penalty": artifact_penalty,
    }


def objective_for_reference(metrics: Mapping[str, float]) -> float:
    """Favor reference agreement and detail fidelity while penalizing artifacts."""
    return float(
        0.48 * min(metrics["psnr"] / 50.0, 1.0)
        + 0.35 * metrics["ssim"]
        + 0.17 * metrics["edge_fidelity"]
        - 0.20 * metrics["artifact_penalty"]
    )


def proxy_metrics(result: np.ndarray, blurred: np.ndarray) -> dict[str, float]:
    """Conservative no-reference score; it intentionally never compares to noisy input."""
    _validate_pair(result, blurred)
    result_luma = _luminance(result.astype(np.float64))
    blurred_luma = _luminance(blurred.astype(np.float64))
    detail_gain = max(
        0.0,
        float(np.var(_gradient_magnitude(result_luma)) - np.var(_gradient_magnitude(blurred_luma))),
    )
    deviation = float(np.mean(np.abs(result_luma - blurred_luma)))
    saturation = float(np.mean((result < 0.002) | (result > 0.998)))
    artifact_penalty = min(1.0, max(0.0, deviation - 0.14) * 4.0 + saturation * 0.2)
    objective = float(
        0.55 * min(detail_gain / 0.015, 1.0)
        + 0.25 * min(deviation / 0.06, 1.0)
        - 0.35 * artifact_penalty
    )
    return {
        "objective": objective,
        "detail_gain": detail_gain,
        "deviation_from_blurred": deviation,
        "artifact_penalty": artifact_penalty,
    }


def select_hpo_candidate(
    candidates: Sequence[CandidateEvaluation],
) -> CandidateEvaluation:
    """Select the highest objective with deterministic configuration tie-breaking."""

    if not candidates:
        raise ValueError("HPO requires at least one candidate")
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate.metrics["objective"]),
            _config_key(candidate.config),
        ),
    )


class BenchmarkRunner:
    """Runs coarse-to-fine HPO and writes only source-independent benchmark output."""

    def __init__(
        self,
        manifest: BenchmarkManifest,
        *,
        dataset_root: str | Path,
        output_dir: str | Path,
        hpo_max_dimension: int = 192,
        output_max_dimension: int = 512,
        processor: Processor | None = None,
    ) -> None:
        if hpo_max_dimension < 64 or output_max_dimension < hpo_max_dimension:
            raise ValueError("output_max_dimension must be at least hpo_max_dimension >= 64")
        self.manifest = manifest
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.hpo_max_dimension = hpo_max_dimension
        self.output_max_dimension = output_max_dimension
        self.processor = processor or _run_pipeline

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        entries = [self._run_dataset(dataset) for dataset in self.manifest.datasets]
        index = {
            "schema_version": 1,
            "datasets": entries,
            "privacy": {
                "input_paths_recorded": False,
                "private_outputs_must_remain_unpublished": True,
            },
        }
        index_path = self.output_dir / "benchmark.json"
        _write_json(index_path, index)
        self._write_summary(entries)
        return index_path

    def _run_dataset(self, dataset: Dataset) -> dict[str, object]:
        blurred = self._load(dataset.blurred)
        noisy = self._load(dataset.noisy)
        reference = self._load(dataset.reference) if dataset.reference else None
        _validate_pair(blurred, noisy)
        if reference is not None:
            _validate_pair(blurred, reference)

        hpo_blurred, hpo_noisy, hpo_reference = _resize_triplet(
            blurred, noisy, reference, self.hpo_max_dimension
        )
        base_config = _base_config(hpo_blurred.shape)
        coarse = self._evaluate_candidates(
            dataset,
            hpo_blurred,
            hpo_noisy,
            hpo_reference,
            [replace(base_config, **overrides) for overrides in _coarse_overrides()],
            "coarse",
        )
        best_coarse = select_hpo_candidate(coarse)
        fine = self._evaluate_candidates(
            dataset,
            hpo_blurred,
            hpo_noisy,
            hpo_reference,
            _fine_configs(best_coarse.config),
            "fine",
        )
        best = select_hpo_candidate((*coarse, *fine))

        final_blurred, final_noisy, final_reference = _resize_triplet(
            blurred, noisy, reference, self.output_max_dimension
        )
        final_config = replace(
            _fit_config_to_image(best.config, final_blurred.shape),
            **dict(dataset.config_overrides or {}),
        )
        _validate_public_reference_config(dataset, final_config)
        started = perf_counter()
        result = self.processor(final_blurred, final_noisy, final_config)
        final_runtime = perf_counter() - started
        metrics = _evaluate(result, final_blurred, final_reference)
        baseline_metrics = (
            image_metrics(final_blurred, final_reference)
            if final_reference is not None
            else None
        )
        dataset_dir = self.output_dir / dataset.identifier
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_png(dataset_dir / "blurred.png", final_blurred)
        write_png(dataset_dir / "noisy.png", final_noisy)
        write_png(dataset_dir / "result.png", result)
        if final_reference is not None:
            write_png(dataset_dir / "reference.png", final_reference)
        comparison = _comparison(
            final_blurred,
            final_noisy,
            result,
            final_reference,
            noisy_label=dataset.noisy_label or "Noisy",
        )
        _write_webp(dataset_dir / "comparison.webp", comparison)
        _write_webp(dataset_dir / "thumbnail.webp", _thumbnail(comparison))
        assets: dict[str, str | None] = {
            "blurred": f"{dataset.identifier}/blurred.png",
            "noisy": f"{dataset.identifier}/noisy.png",
            "result": f"{dataset.identifier}/result.png",
            "reference": (
                f"{dataset.identifier}/reference.png"
                if final_reference is not None
                else None
            ),
            "comparison": f"{dataset.identifier}/comparison.webp",
            "thumbnail": f"{dataset.identifier}/thumbnail.webp",
        }
        asset_checksums = {
            role: _sha256(self.output_dir / path)
            for role, path in assets.items()
            if path is not None
        }

        record: dict[str, object] = {
            "id": dataset.identifier,
            "description": dataset.description,
            "visibility": dataset.visibility,
            "rights": dataset.rights,
            "objective_type": "reference" if reference is not None else "no-reference-proxy",
            "source_hashes": dict(dataset.source_hashes or {}),
            "input_checksums": {
                "blurred": dataset.blurred.sha256,
                "noisy": dataset.noisy.sha256,
                **({"reference": dataset.reference.sha256} if dataset.reference else {}),
            },
            "original_size": _size_record(blurred),
            "output_size": _size_record(final_blurred),
            "selected_config": asdict(final_config),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "runtime_seconds": final_runtime,
            "hpo": {
                "max_dimension": self.hpo_max_dimension,
                "coarse": [_candidate_record(candidate) for candidate in coarse],
                "fine": [_candidate_record(candidate) for candidate in fine],
                "selected_objective": float(best.metrics["objective"]),
            },
            "assets": assets,
            "asset_checksums": asset_checksums,
        }
        if dataset.noisy_label is not None:
            record["noisy_label"] = dataset.noisy_label
        if dataset.input_processing is not None:
            record["input_processing"] = dict(dataset.input_processing)
        if dataset.config_overrides is not None:
            record["config_overrides"] = dict(dataset.config_overrides)
        _write_json(dataset_dir / "run.json", record)
        return record

    def _evaluate_candidates(
        self,
        dataset: Dataset,
        blurred: np.ndarray,
        noisy: np.ndarray,
        reference: np.ndarray | None,
        configs: Sequence[PipelineConfig],
        stage: str,
    ) -> list[CandidateEvaluation]:
        return [
            self._evaluate_candidate(
                dataset, blurred, noisy, reference, config, f"{stage}-{index}"
            )
            for index, config in enumerate(configs)
        ]

    def _evaluate_candidate(
        self,
        dataset: Dataset,
        blurred: np.ndarray,
        noisy: np.ndarray,
        reference: np.ndarray | None,
        config: PipelineConfig,
        cache_name: str,
    ) -> CandidateEvaluation:
        cache_dir = self.output_dir / ".cache" / dataset.identifier
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_name}-{_config_digest(config)}.npz"
        if cache_path.is_file():
            cached = np.load(cache_path)
            result = cached["result"]
            runtime_seconds = float(cached["runtime_seconds"])
        else:
            started = perf_counter()
            result = self.processor(blurred, noisy, config)
            runtime_seconds = perf_counter() - started
            np.savez_compressed(cache_path, result=result, runtime_seconds=runtime_seconds)
        metrics = _evaluate(result, blurred, reference)
        return CandidateEvaluation(config, metrics, runtime_seconds)

    def _load(self, item: InputFile | None) -> np.ndarray:
        if item is None:
            raise ValueError("required input is missing")
        path = self.dataset_root / item.path
        if _sha256(path) != item.sha256:
            raise ValueError(f"checksum mismatch for benchmark input {item.path}")
        return read_input_image(path)

    def _write_summary(self, entries: Sequence[Mapping[str, object]]) -> None:
        lines = [
            "# Benchmark summary",
            "",
            "Comparisons are ordered **blurred + noisy -> result**, followed by a reference when available.",
            "",
            "| Dataset | Objective type | Full-resolution objective | Blurred PSNR | Result PSNR | Result SSIM | Runtime (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for entry in entries:
            metrics = entry["metrics"]
            assert isinstance(metrics, Mapping)
            lines.append(
                "| {id} | {objective_type} | {objective:.6f} | {baseline_psnr} | {psnr} | {ssim} | {runtime:.3f} |".format(
                    id=entry["id"],
                    objective_type=entry["objective_type"],
                    objective=float(metrics["objective"]),
                    baseline_psnr=(
                        f"{baseline['psnr']:.3f}"
                        if isinstance((baseline := entry.get("baseline_metrics")), Mapping)
                        else "N/A"
                    ),
                    psnr=f"{metrics['psnr']:.3f}" if "psnr" in metrics else "N/A",
                    ssim=f"{metrics['ssim']:.4f}" if "ssim" in metrics else "N/A",
                    runtime=float(entry["runtime_seconds"]),
                )
            )
        lines.extend(
            [
                "",
                "Reference-backed entries report PSNR and SSIM against their published reference.",
                "No-reference-proxy entries use a conservative full-resolution proxy objective, not PSNR, SSIM, or noisy-input similarity.",
                "",
            ]
        )
        (self.output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_input(raw: Mapping[str, object], identifier: str, role: str) -> InputFile:
    value = raw.get(role)
    if not isinstance(value, dict):
        raise ValueError(f"{identifier}: {role} must be an object")
    path = _required_string(value, "path")
    path_object = Path(path)
    if path_object.is_absolute() or ".." in path_object.parts:
        raise ValueError(f"{identifier}: {role}.path must be a safe relative path")
    digest = _required_string(value, "sha256")
    if not _is_sha256(digest):
        raise ValueError(f"{identifier}: {role}.sha256 must be a SHA-256 hex digest")
    return InputFile(path=path, sha256=digest)


def _required_string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest value {key!r} must be a non-empty string")
    return value


def _optional_string(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest value {key!r} must be a non-empty string when present")
    return value


def _optional_mapping(
    raw: Mapping[str, object],
    identifier: str,
    key: str,
) -> Mapping[str, object] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{identifier}: {key} must be an object")
    return dict(value)


def _parse_config_overrides(
    raw: Mapping[str, object],
    identifier: str,
) -> Mapping[str, object] | None:
    value = _optional_mapping(raw, identifier, "config_overrides")
    if value is None:
        return None
    valid_fields = {field.name for field in fields(PipelineConfig)}
    unknown = sorted(set(value) - valid_fields)
    if unknown:
        raise ValueError(
            f"{identifier}: unknown PipelineConfig override(s): {', '.join(unknown)}"
        )
    try:
        replace(PipelineConfig(), **dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{identifier}: invalid config_overrides: {error}") from error
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_config(shape: tuple[int, ...]) -> PipelineConfig:
    minimum = min(shape[:2])
    kernel_size = min(15, max(7, (minimum // 8) | 1))
    patch_size = max(64, min(128, minimum))
    return PipelineConfig(
        random_seed=20260715,
        max_corners=1200,
        kernel_size=kernel_size,
        color_labels=8,
        graph_cut_size=max(64, min(160, minimum)),
        max_regions=4,
        patch_size=max(patch_size, kernel_size),
        tikhonov=0.002,
        data_weight=220.0,
        beta_max=24.0,
        unsharp_amount=0.1,
        detail_fusion_amount=0.05,
    )


def _coarse_overrides() -> tuple[dict[str, float], ...]:
    return (
        {"tikhonov": 0.001, "data_weight": 160.0, "unsharp_amount": 0.0},
        {"tikhonov": 0.002, "data_weight": 220.0, "unsharp_amount": 0.1},
        {"tikhonov": 0.004, "data_weight": 300.0, "unsharp_amount": 0.18},
    )


def _validate_public_reference_config(dataset: Dataset, config: PipelineConfig) -> None:
    if dataset.visibility != "public" or dataset.reference is None:
        return
    if config.input_detail_blend > 0.25:
        raise ValueError(
            "public reference-backed benchmarks cannot replace regional deconvolution "
            "with an input-detail blend above 0.25"
        )
    if config.noisy_structure_blend > 0.25:
        raise ValueError(
            "public reference-backed benchmarks cannot replace regional deconvolution "
            "with a noisy-structure blend above 0.25"
        )


def _fine_configs(best: PipelineConfig) -> tuple[PipelineConfig, ...]:
    return (
        replace(
            best,
            tikhonov=max(best.tikhonov / 1.5, 1e-5),
            detail_fusion_amount=max(0.0, best.detail_fusion_amount - 0.03),
        ),
        replace(
            best,
            tikhonov=best.tikhonov * 1.5,
            data_weight=best.data_weight * 1.15,
        ),
    )


def _fit_config_to_image(config: PipelineConfig, shape: tuple[int, ...]) -> PipelineConfig:
    minimum = min(shape[:2])
    return replace(
        config,
        graph_cut_size=max(64, min(config.graph_cut_size, minimum)),
        patch_size=max(config.kernel_size, min(config.patch_size, minimum)),
    )


def _evaluate(
    result: np.ndarray, blurred: np.ndarray, reference: np.ndarray | None
) -> dict[str, float | str | bool]:
    if reference is None:
        metrics: dict[str, float | str | bool] = proxy_metrics(result, blurred)
        metrics["reference_available"] = False
        return metrics
    metrics = image_metrics(result, reference)
    metrics["objective"] = objective_for_reference(metrics)
    metrics["reference_available"] = True
    return metrics


def _run_pipeline(
    blurred: np.ndarray, noisy: np.ndarray, config: PipelineConfig
) -> np.ndarray:
    return DisparityDeblurPipeline(config).process(blurred, noisy).restored


def _resize_triplet(
    blurred: np.ndarray,
    noisy: np.ndarray,
    reference: np.ndarray | None,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resize a paired dataset by its longest axis while preserving aspect ratio."""

    dimensions = blurred.shape[:2]
    scale = min(1.0, maximum / max(dimensions))
    if scale == 1.0:
        return blurred, noisy, reference
    width = max(1, round(blurred.shape[1] * scale))
    height = max(1, round(blurred.shape[0] * scale))
    resize = lambda image: cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resize(blurred), resize(noisy), resize(reference) if reference is not None else None


def _comparison(
    blurred: np.ndarray,
    noisy: np.ndarray,
    result: np.ndarray,
    reference: np.ndarray | None,
    *,
    noisy_label: str = "Noisy",
) -> np.ndarray:
    """Build a labeled horizontal comparison from equal-sized images."""

    images = [blurred, noisy, result]
    if reference is not None:
        images.append(reference)
    labels = ["Blurred", noisy_label, "Result"] + (
        ["Reference"] if reference is not None else []
    )
    labeled = []
    for image, label in zip(images, labels, strict=True):
        preview = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(
            preview,
            label,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled.append(preview.astype(np.float32) / 255.0)
    return np.concatenate(labeled, axis=1)


def _thumbnail(image: np.ndarray) -> np.ndarray:
    """Bound the longest thumbnail axis without changing aspect ratio."""

    maximum = 640
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return image
    output_width = max(1, round(width * scale))
    output_height = max(1, round(height * scale))
    return cv2.resize(
        image,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _write_webp(path: Path, image: np.ndarray) -> None:
    rgb = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    success = cv2.imwrite(
        str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 88]
    )
    if not success:
        raise OSError(f"failed to write {path}")


def _candidate_record(candidate: CandidateEvaluation) -> dict[str, object]:
    return {
        "config": asdict(candidate.config),
        "metrics": dict(candidate.metrics),
        "runtime_seconds": candidate.runtime_seconds,
    }


def _config_key(config: PipelineConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _config_digest(config: PipelineConfig) -> str:
    return sha256(_config_key(config).encode("utf-8")).hexdigest()[:16]


def _size_record(image: np.ndarray) -> dict[str, int]:
    return {"width": int(image.shape[1]), "height": int(image.shape[0])}


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_pair(left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape:
        raise ValueError(f"benchmark images differ in shape: {left.shape} vs {right.shape}")
    if left.ndim != 3 or left.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got {left.shape}")


def _luminance(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    horizontal = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    vertical = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(horizontal, vertical)


def main() -> None:
    """Run a manifest from the ``disparity-deblur-benchmark`` console command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run deterministic coarse-to-fine HPO and write publishable comparisons."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root containing manifest-relative image files; never recorded in output.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hpo-max-dimension", type=int, default=192)
    parser.add_argument("--output-max-dimension", type=int, default=512)
    args = parser.parse_args()
    output = BenchmarkRunner(
        load_manifest(args.manifest),
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        hpo_max_dimension=args.hpo_max_dimension,
        output_max_dimension=args.output_max_dimension,
    ).run()
    print(output)
