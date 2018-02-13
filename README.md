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
An example of distributed training MNIST with synchronous SGD is provided `mnist_dist.py`.
Generally, it takes only 4 steps to apply distributed training.
Please notice that this example serves as an fundamental prototype of distributed learning, `torch.distributed` provides all kinds of communication primitives by which you can implement any kind of synchronization algorithm such as DeepSpeech's [ring-allreduce](http://pytorch.org/tutorials/intermediate/dist_tuto.html#our-own-ring-allreduce).

1. Init with `mpi` backend
```python
import torch
import torch.distributed as dist
dist.init_process_group(backend='mpi')
```
2. Partition you local dataset using `dist.get_rank()` and `mpi.get_world_size()`.
The dataset partition pattern is user defined, you may randomly select batch for each rank or use a uniform shuffled index as shown in the example.
3. After your network is initialized, synchronize weights across all ranks.
Theoratically you only need to synchronize weights once at the beginning of training.
Pratically I tend to synchronize weights at the beginning of every epoch to kill accumulated numerical error.
```python
def sync_params(model):
    """ broadcast rank 0 parameter to all ranks """
    for param in model.parameters():
        dist.broadcast(param.data, 0)
```
4. After each backward step, synchronize gradients
```python
def sync_grads(model):
    """ all_reduce grads from all ranks """
    for param in model.parameters():
        dist.all_reduce(param.grad.data)
```
I also did a little trick rewritten `print()`, with `debug_print=True`, print message will be addad a prefix of [rank/world_size], with `debug_print=False`, only message from master rank will be printed while rest of ranks will be muted.

To lauch this example on Cori:
 ```bash
cd examples/
salloc --qos=interactive -N 2 -t 00:30:00 -C haswell
# to kill MPI implementation doesn't support multithreading WARNING:
export MPICH_MAX_THREAD_SAFETY=multiple
srun -N 2 python -u mnist_dist.py
```
