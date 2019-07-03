### Groth16 prover
This directory contains a reference CPU implementation of  the
Groth16 prover
using [libsnark](README-libsnark.md).

There are two provers implemented in this repository. The first is `libsnark/main.cpp`, which has _the_ CPU libsnark reference prover that we compare benchmarks against. You can modify this with debug info if you need, but you shouldn't try and add CUDA calls in it.

The other prover is in `cuda_prover_piecewise.cu`. It also calls the CPU-based libsnark functions, although through a wrapper library to avoid having to include the libsnark headers‡. Once you have GPU implementations of the expensive algorithms (FFT, multiexp), replace the calls to methods in `B` with calls to your GPU functions. You will probably need to add functions to the wrapper classes (implemented in `libsnark/prover_reference_functions.cpp` and `libsnark/prover_reference_include/prover_reference_functions.hpp`) to copy libsnark data into the format you need for CUDA. If you need help dealing with the wrapper functions, ask around in the `#snark-challenge` discord channel.

#### Dependencies

The code should compile and run on Ubuntu 18.04 with the following dependencies installed:

``` bash
sudo apt-get install -y build-essential \
    cmake \
    git \
    libomp-dev \
    libgmp3-dev \
    libprocps-dev \
    python-markdown \
    libboost-all-dev \
    libssl-dev \
    pkg-config \
    nvidia-cuda-toolkit
```


Building on MacOS is not recommended as CUDA support is harder to use. (Apple mostly ships with AMD.)


#### Build
``` bash
./build.sh
```

#### Generate parameters and inputs
``` bash
./generate_parameters
```

When iterating on your implementation for correctness (and not performance)
smaller constraint systems are fine. In that case, `./generate_parameters fast`
will give you smaller parameters that don't take as long to generate or
prove with.

### Run
``` bash
./main MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output
./main MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output
./cuda_prover_piecewise MNT4753 compute MNT4753-parameters MNT4753-input MNT4753-output_cuda
./cuda_prover_piecewise MNT6753 compute MNT6753-parameters MNT6753-input MNT6753-output_cuda
```

### Check results
``` bash
sha256sum MNT4753-output MNT6753-output MNT4753-output_cuda MNT6753-output_cuda
```

‡ Due to some bug with `nvcc` in version `10.1`, including any of the `mnt4`/`mnt6` `libff` headers (a library libsnark uses for finite field arithmetic) leads to a compilation failure. The wrapper library uses something like the [pImpl idiom](https://en.cppreference.com/w/cpp/language/pimpl) to hide the need for those headers from the caller.
