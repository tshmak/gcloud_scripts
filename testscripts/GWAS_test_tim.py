# coding: utf-8
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
vds.rows().select('locus', 'alleles').show(5)
print('Show samples')
vds.s.show(5)
vds.entry.take(5)
# import phenotype
table = (hl.import_table(
    'gs://ukb_testdata/sleep_duration.tsv',
    delimiter='\t', no_header=True, missing='NA', impute=True, types={'f0': hl.tstr}).key_by('f0'))
table.show()
vds = vds.annotate_cols(**table[vds.s])
print(vds.col.dtype.pretty())
dic = {}
for i in np.arange(1, 41):
    dic['pc' + str(i)] = hl.tfloat
pcas = hl.import_table('gs://ukb_testdata/covar.txt', delimiter=' ', types=dic).key_by('FID')
pcas = pcas.drop('IID')
vds = vds.annotate_cols(**pcas[vds.s])
vds.describe()
pprint(vds.aggregate_cols(agg.stats(vds.f1)))
vds = hl.sample_qc(vds)
vds.select_cols('f1')
gwas = hl.linear_regression(y=vds.f1, x=vds.GT.n_alt_alleles(),
    covariates=[vds['pc'+str(i)] for i in range(1, 12)])
results = gwas.rows()
results.write('gs://ukb_testdata/GWAS_test_tim/GWAS_test_tim', overwrite=True)
# gres = hl.read_table('gs://ukb_testdata/gwasresults')
# gres.linreg.show(4)
# def qqplot(pvals, xMax, yMax):
#     spvals = sorted(filter(lambda x: x and not(isnan(x)), pvals))
#     exp = [-log(float(i) / len(spvals), 10) for i in np.arange(1, len(spvals) + 1, 1)]
#     obs = [-log(p, 10) for p in spvals]
#     plt.clf()
#     plt.scatter(exp, obs)
#     plt.plot(np.arange(0, max(xMax, yMax)), c="red")
#     plt.xlabel("Expected p-value (-log10 scale)")
#     plt.ylabel("Observed p-value (-log10 scale)")
#     plt.xlim(0, xMax)
#     plt.ylim(0, yMax)
#     plt.show()
# get_ipython().run_line_magic('matplotlib', 'inline')
# pvalues = gres.linreg.p_value.collect()
# qqplot(pvalues, 5, 6)
