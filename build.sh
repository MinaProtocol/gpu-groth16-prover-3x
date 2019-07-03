#!/bin/bash
mkdir build
pushd build
  cmake -DMULTICORE=ON -DUSE_PT_COMPRESSION=OFF $EXTRA_CMAKE_ARGS_FOR_CI ..
  make -j12 main generate_parameters cuda_prover_piecewise
popd
mv build/libsnark/main .
mv build/libsnark/generate_parameters .
mv build/cuda_prover_piecewise .
