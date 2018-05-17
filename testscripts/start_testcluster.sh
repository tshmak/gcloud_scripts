#!/bin/sh

cluster start testcluster \
  --master-machine-type n1-highmem-8 \
  --worker-machine-type n1-standard-8 \
  --num-workers 2 \
  --num-preemptible-workers 2 \
  --version devel \
  --spark 2.2.0 \
  --zone asia-east1-a

