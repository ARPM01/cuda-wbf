# cuda-wbf

A research project to implement Weighted Boxes Fusion (WBF) for object detection in CUDA and measure when it reduces postprocessing latency versus the Python reference.

Start with the [implementation plan](IMPLEMENTATION_PLAN.md). It defines the reference behavior, the two CUDA kernels, the SHARC bring-up sequence, and the COCO benchmark protocol. The plan includes a CPU-only build for development on machines without CUDA.

The repository currently contains the plan. The implementation and benchmark runner are pending.
