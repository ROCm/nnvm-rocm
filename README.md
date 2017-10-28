# nnvm-rocm
Test different front-ends for nnvm to run on AMD GPUs using ROCm

* getting_started.py: A simple usage of ROCm backend with TVM
* mxnet_imagenet_inference.py: Runs Gluon Resnet 50 model on a cat image
* simple_superres_onnx.py: Loads and runs a simple 4 layer image super-resolution network in ONNX format
* advanced_superres_onnx.py: Loads and runs a state of the art image super-resolution network in ONNX format
* export_superres.py: An example of defining a network in PyTorch and exporting it in ONNX format

In order to use ROCm backend, you need to build TVM with LLVM 5.0 or higher. You also need to have LLD linker installed and ld.lld command should be on your PATH. If your ld.lld is installed as ld.lld-5.0, you need to sym link it to ld.lld.
Finally, you should install ROCm following the instruction [here](https://rocm.github.io/install.html).

You need to have the latest NNVM and TVM installed to run the examples.

To run ONNX examples, you need to install onnx library. Install it with
```
conda install -c ezyang onnx
```

To run MXNet-Gluon examples, you need to install MXNet package. Install it with
```
pip install mxnet --user

```

