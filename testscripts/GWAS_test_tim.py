#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hail as hl
import hail.expr.aggregators as agg
import numpy as np
import matplotlib.pyplot as plt
from math import log, isnan
from pprint import pprint
hl.init()
vds = hl.import_plink('gs://ukb_testdata/maf_0.01_10.bed',
                      'gs://ukb_testdata/maf_0.01_10.bim',
                      'gs://ukb_testdata/maf_0.01_10.fam' )
# import phenotype
table = (hl.import_table(
    'gs://ukb_testdata/sleep_duration.tsv',
    delimiter='\t', no_header=True, missing='NA', impute=True, types={'f0': hl.tstr}).key_by('f0'))
vds = vds.annotate_cols(**table[vds.s])
# import covar
dic = {}
for i in np.arange(1, 41):
    dic['pc' + str(i)] = hl.tfloat
pcas = hl.import_table('gs://ukb_testdata/covar.txt', delimiter=' ', types=dic).key_by('FID')
pcas = pcas.drop('IID')
vds = vds.annotate_cols(**pcas[vds.s])
vds.select_cols('f1')

gwas = hl.linear_regression(y=vds.f1, x=vds.GT.n_alt_alleles(),
    covariates=[vds['pc'+str(i)] for i in range(1, 12)])
results = gwas.rows()
results.write('gs://ukb_testdata/GWAS_test_tim/GWAS_test_tim', overwrite=True)
