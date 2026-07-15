# Third-party licenses and attribution

## Public showcase data

The `benchmarks/public-assets/gopro-flower/` images are derived from the
GOPRO_Large Flower example by Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee
(SNU CVLab). They are licensed under [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/).

* Official dataset page: <https://seungjunnah.github.io/Datasets/gopro>
* Pinned blurred source: <https://raw.githubusercontent.com/SeungjunNah/DeepDeblur_release/2d5a698560e658718f0520e48dfb15bd52c80118/images/Flower_blur1.png>
* Pinned sharp source: <https://raw.githubusercontent.com/SeungjunNah/DeepDeblur_release/2d5a698560e658718f0520e48dfb15bd52c80118/images/Flower_sharp1.png>
* Citation: Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee, “Deep Multi-Scale
  Convolutional Neural Network for Dynamic Scene Deblurring,” CVPR 2017.

The exact source checksums, derived checksums, downscaling method, and
deterministic noise transformation are in `benchmarks/ATTRIBUTION.json`.

## Python dependencies

The reference implementation declares `numpy`, `opencv-python-headless`,
`pymaxflow`, and `scipy` in `pyproject.toml`. They are separate works under
their respective licenses. `uv.lock` records the resolved package metadata and
license information where supplied by the upstream distributions.
