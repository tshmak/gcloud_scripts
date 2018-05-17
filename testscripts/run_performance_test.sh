#!/bin/sh
name='performancetest1'
submitscript='GWAS_test_tim.py'

cluster start $name \
  --master-machine-type n1-standard-8 \
  --worker-machine-type n1-standard-8 \
  --num-workers 8 \
  --version devel \
  --spark 2.2.0 \
  --zone asia-east1-a

cluster submit $name $submitscript

cluster stop $name
