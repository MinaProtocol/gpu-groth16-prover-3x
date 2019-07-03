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
