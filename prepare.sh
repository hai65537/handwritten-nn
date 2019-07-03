#!/bin/bash

mkdir -p data/mnist
pushd data/mnist

for f in {train,t10k}-{images-idx3,labels-idx1}-ubyte; do
    curl http://yann.lecun.com/exdb/mnist/$f.gz | gzip -d - > "$f"
done

popd
