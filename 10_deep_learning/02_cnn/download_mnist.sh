#!/bin/sh

url_base=http://yann.lecun.com/exdb/mnist

mkdir -p dataset
cd dataset

curl -O ${url_base}/train-images-idx3-ubyte.gz
curl -O ${url_base}/train-labels-idx1-ubyte.gz
curl -O ${url_base}/t10k-images-idx3-ubyte.gz
curl -O ${url_base}/t10k-labels-idx1-ubyte.gz

gunzip *.gz