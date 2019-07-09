### GPU Groth16 prover (3x faster than CPU)

This is a GPU Groth16 prover that won the `2x` speedup prize.
It follows the template [from the reference](https://github.com/CodaProtocol/snark-challenge-prover-reference).

This prover requires a substantial amount of RAM to use. The reference machine has 32GB.

This prover has a slow preprocessing step! Note the changed instructions below.

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
./main MNT4753 preprocess MNT4753-parameters # IMPORTANT PREPROCESSING STEPS
./main MNT6753 preprocess MNT6753-parameters
```

The preprocessed filenames are currently hardcoded to `MNT4753_preprocessed` and `MNT6753_preprocessed`.

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

### How to make this faster

There are 3 components to this SNARK prover which can be optimized.

The multiexponentiations are still the bottleneck and optimization efforts should probably be focused there right now.

#### G1 multiexponentiation

- The [current best implementation](https://github.com/CodaProtocol/gpu-groth16-prover-3x/blob/master/multiexp/reduce.cu#L49) performs a "map-reduce" to implement the multiexponentiation with a batched double-and-add being the base multiexponentiation "map" function. The Pippenger algorithm ([described here](https://pdfs.semanticscholar.org/486e/573e23ad21623d6f4f7ff035b77e1db7b835.pdf) and implemented for another curve [here](https://github.com/matter-labs/belle_cuda/blob/master/sources/multiexp.cu) is likely to be significantly faster.

#### G2 multiexponentiation

- The above remarks about the Pippenger algorithm also apply here. To repeat, the [current best implementation](https://github.com/CodaProtocol/gpu-groth16-prover-3x/blob/master/multiexp/reduce.cu#L49) performs a "map-reduce" to implement the multiexponentiation with a batched double-and-add being the base multiexponentiation "map" function. The Pippenger algorithm ([described here](https://pdfs.semanticscholar.org/486e/573e23ad21623d6f4f7ff035b77e1db7b835.pdf) and implemented for another curve [here](https://github.com/matter-labs/belle_cuda/blob/master/sources/multiexp.cu) is likely to be significantly faster.

- The technique in [this paper](https://eprint.iacr.org/2008/117.pdf) can be used to speed up the G2 multi-exponentiation by about 2x.

#### FFT

This implementation did not implement the fast-fourier-transform for the GPU and used the existing C++ CPU implementation. It may be possible to get additional speedups by implementing the FFT on the GPU, although the current-best implementation performs the FFT on the CPU while the GPU is busy working, so it may not be blocking anything at the moment.
