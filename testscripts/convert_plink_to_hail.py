"""Conversion Script."""

import hail as hl

print("This only is adviced for Hail 0.1")

path = 'gs://ukb_testdata/maf_0.01_10'

hc = hl.HailContext()

(hc.import_plink(
    bed=path+'.bed',
    bim=path+'.bim',
    fam=path+'.fam')).write('gs://ukb_testdata/maf_0.01_10.vds')
