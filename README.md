# pytorch_nersc_wheel
This repo contains PyTorch wheel installation file for NERSC Cori super computer, source code from [Intel-Pytorch](https://github.com/intel/pytorch), [icc](https://github.com/intel/pytorch/tree/icc) branch.

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
* [LSTM](https://github.com/xhzhao/pytorch-rnn-benchmark)

## Batch script for Cori
On Haswell compute nodes
```bash
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
```bash
#!/bin/bash

#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -q regular
#SBATCH -L SCRATCH
#SBATCH -C knl,quad,cache

export KMP_AFFINITY="granularity=fine,compact,1,0"
export OMP_NUM_THREADS=68
python benchmark.py
```

## Distributed Training
This section provides basic guidelines of implmenting Synchronous SGD with pytorch distributed package. 
[pytorch.distributed](http://pytorch.org/docs/master/distributed.html) provides an MPI-like interface for exchanging tensor data across multi-node networks. Note that a friendly wrapper of distrubited for Synchronous SGD is provides by [torch.nn.parallel.DistributedDataParallel](http://pytorch.org/docs/master/nn.html#torch.nn.parallel.DistributedDataParallel), but it supports only `nccl` and `gloo` backend. 
In this section, we will go through how to implement Synchronous SGD with `mpi` backend with an example of multi node version of [OpenNMT-py](https://github.com/mingfeima/OpenNMT-py/tree/dist). Also please be aware, PyTorch provides all fundamental communication primitives, so you also build Asynchronous SGD or any other fancy communication patterns such as [ring-allreduce](http://pytorch.org/tutorials/intermediate/dist_tuto.html#our-own-ring-allreduce) as well.
A more general tutorial for distributed training on PyTorch can be found by [Writing Distributed Applications with PyTorch](http://pytorch.org/tutorials/intermediate/dist_tuto.html).

I wrote a helper class [Dist](https://github.com/mingfeima/OpenNMT-py/blob/dist/onmt/Dist.py) which can be ported easily.
```python
# init distributed with mpi backend (train.py#L192)
dist = onmt.Dist(opt.dist, opt.dist_backend, opt.debug_print)

# prepare dataset partitions
# you need to determine the which batch to be trained on a rank N
# if you intend to perform shuffer before selecting the batches, make sure the shuffled batch order is unique across all ranks
# (train.py#L256)
dist.broadcast(batchOrder)

for epoch in range(num_epochs):
  # synchronous parameters across all ranks, perform broadcast (train.py#L259)
  dist.syncParams(model.parameters())
  
  for i in range(num_batches / mpi_world_size):
    # get training batch from dataset according to rank()
    # train your gradients as normal
    torch.autograd.backward(inputs, grads)
     
    # accumulate gradients across all ranks, perform allreduce ()
    dist.accGradParams(model.parameters())
    
    # Update the parameters as normal
    # note that we only sync parameters at the begining of each epoch
    # the updated parameter is supposed to be identical across ranks since the accumulated gradient is identical
    optim.step()
  
```
At last, the loss caculation should be modified as some factors (e.g. total number of words, total number of images) also need to be accumulated (Loss.py#L75)
