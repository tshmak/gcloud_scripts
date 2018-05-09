"""Conversion Script."""

import hail as hl

path = 'gs://ukb_testdata/maf_0.01_10'

hc = hl.HailContext()

(hc.import_plink(
    bed=path+'.bed',
    bim=path+'.bim',
    fam=path+'.fam')).write('gs://ukb_testdata/maf_0.01_10.vds')
