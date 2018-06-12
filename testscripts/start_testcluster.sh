#!/bin/sh

cluster start testcluster \
  --master-machine-type n1-highmem-8 \
  --worker-machine-type n1-standard-8 \
  --num-workers 4 \
  --num-preemptible-workers 4 \
  --version devel \
  --spark 2.2.0 \
  --zone asia-east1-a \
  --pkgs scikit-learn \
  --init gs://ukb_testdata/additional_init.sh
