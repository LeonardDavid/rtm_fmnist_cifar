# BNN-QNN-ErrorEvaluation
A framework for error tolerance training and evaluation of Binarized and Quantized Neural Networks

## CUDA-based Binarization, Quantization, and Error Injection

First, install PyTorch. For fast binarization/quantization and error injection during training, CUDA support is needed. To enable it, install pybind11 and CUDA toolkit.

Then, to install the CUDA-kernels, go to the folder ```code/cuda/``` and run

```./install_kernels.sh```
