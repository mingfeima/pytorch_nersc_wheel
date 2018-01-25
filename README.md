# pytorch_nersc_wheel
This repo contains PyTorch wheel installation file for NERSC Cori super computer, source code from [Intel-Pytorch](https://github.com/intel/pytorch)

## Performance has been optimized for the following features:
* Conv2d optimization with MKL-DNN
* Conv3d optimization with vol2col and col2vol parallelization
* LSTM optimization with sigmoid and tanh parallelization
* AVX512 support for Intel Xeon Skylake CPU and Xeon Phi (KNL, KNM)
* Provide icc support to accommodate NERSC compilation environment

## Installation
* `pytorch_original_gcc` - compiled with origin PyTorch master with gcc
* `pytorch_intel_icc` - compiled with Intel-PyTorch `icc` branch with icc, specialized for Haswell
* `pytorch_intel_icc512` - compiled with Intel-PyTorch `icc` branch with icc and AVX512 support, specialized for KNL

## Benchmark
* [Conv3d](https://github.com/MlWoo/PyTorch-benchmark)
* [Alexnet](https://github.com/mingfeima/convnet-benchmarks)

## Batch script for Cori
On Haswell compute nodes
```
#!/bin/bash

#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -q regular
#SBATCH -L SCRATCH
#SBATCH -C haswell

export OMP_NUM_THREADS=32
python benchmark.py
```
On KNL compute nodes
```
#!/bin/bash -l

#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -q regular
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache

export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=68
python benchmark.py
```
