# COCO milestone validation

The full-data gate compares standard Python WBF with scalar C++ and both CUDA kernels on the entire COCO `val2017` annotation set. It checks output counts and ordered category IDs before COCOeval applies its detection limits.

## Full-data results

The November 18, 2026 run passed on all **5,000 images**. Scalar C++, `cuda_groups`, and `cuda_coop` matched the Python oracle's output counts and ordered labels exactly. The maximum absolute coordinate error, maximum absolute score error, and absolute COCO AP difference were all **0** for each backend.

| Method | AP | AP50 | AP small | AP medium | AP large | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python WBF | 0.558221 | 0.741723 | 0.401088 | 0.603007 | 0.695210 | 0.741066 |
| C++ WBF | 0.558221 | 0.741723 | 0.401088 | 0.603007 | 0.695210 | 0.741066 |
| CUDA groups WBF | 0.558221 | 0.741723 | 0.401088 | 0.603007 | 0.695210 | 0.741066 |
| CUDA cooperative WBF | 0.558221 | 0.741723 | 0.401088 | 0.603007 | 0.695210 | 0.741066 |
| CPU hard NMS | 0.524948 | 0.722043 | 0.359010 | 0.571840 | 0.671194 | 0.698850 |
| CPU Gaussian Soft-NMS | 0.521155 | 0.713373 | 0.354541 | 0.567487 | 0.668655 | 0.684990 |
| TorchVision CUDA hard NMS | 0.524948 | 0.722043 | 0.359010 | 0.571840 | 0.671194 | 0.698850 |

The loader kept **5,749,126** input detections, discarded **508,895** reversed or zero-area boxes, and discarded **503** nonpositive-score rows. The full-precision metrics and run provenance are in `runs/m3/quality.json` and `runs/m3/metadata.json`.

## Reproduction

Use the Python 3.11 environment, optional evaluation dependencies, and CUDA package described in the [README](../README.md#milestone-3-real-data-verification-and-comparisons). In the validation workspace, the two assets were staged under `build/m3-data`.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/build/m3-cuda/stage"
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m cuda_wbf.bench verify \
  --predictions build/m3-data/benchmark.zip \
  --annotations build/m3-data/instances_val2017.json \
  --baselines --output runs/m3
```

| Asset | SHA-256 |
| --- | --- |
| `benchmark.zip` | `3a8c2e7cfb985a8b2062702e2e8ebbc0447979fb7abcd827795238a99412880a` |
| `instances_val2017.json` | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |

The prediction archive contains 503 EffNetB7 placeholder rows with category `0` and a negative score. The loader counts these under the nonpositive-score rule and validates COCO category IDs on the remaining rows. It also preserves category IDs written as integer-valued decimals, such as `1.0` in a DetRS source.

## Environment

- One NVIDIA A100-SXM4-40GB (compute capability 8.0) on the development host, driver 580.159.03 and CUDA Toolkit 12.8.93.
- Python 3.11.18, GCC 13.3, CMake 3.28.3, pycocotools 2.0.11, torch 2.8.0+cu128, and torchvision 0.23.0+cu128.
- The WBF oracle used ensemble-boxes 1.0.9, NumPy 2.2.6, and Numba 0.61.2.

## Test coverage

- The installed CPU wheel passed 222 tests from outside the checkout, with 296 GPU tests skipped.
- The staged CUDA build passed 516 tests, including the TorchVision CUDA hard NMS fixture. Two missing-backend checks were skipped because both CUDA backends were present.
