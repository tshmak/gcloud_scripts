import hail as hl
import hail.expr.aggregators as agg
import numpy as np
import matplotlib.pyplot as plt
from math import log, isnan
from pprint import pprint
import time
hl.init() # Initialize Hail and Spark.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from pyspark.sql.functions import udf

# Import a PLINK dataset (BED, BIM, FAM) as a MatrixTable
vds = hl.import_plink('gs://ukb_testdata/data/maf_0.01_10.bed',
                      'gs://ukb_testdata/data/maf_0.01_10.bim',
                      'gs://ukb_testdata/data/maf_0.01_10.fam' )

# import real phenotype
table = (hl.import_table(
    'gs://ukb_testdata/data/sleep_duration.tsv',
    delimiter='\t', no_header=True, missing='NA', impute=True, types={'f0': hl.tstr}).key_by('f0'))
vds = vds.annotate_cols(**table[vds.s])
# import covar
# dic = {}
# for i in np.arange(1, 41):
#     dic['pc' + str(i)] = hl.tfloat
# pcas = hl.import_table('gs://ukb_testdata/data/covar.txt', delimiter=' ', types=dic).key_by('FID')
# pcas = pcas.drop('IID')
# vds = vds.annotate_cols(**pcas[vds.s])
# vds.select_cols('f1')

print("current time is: ", time.asctime(time.localtime(time.time())))
bed = hl.import_bed('gs://ukb_testdata/data/Berisa.EUR.hg19_modif.bed')
print("current time is: ", time.asctime(time.localtime(time.time())))

vds = vds.annotate_rows(LD_block = bed[vds.locus].target)

gts_as_rows = vds.annotate_rows(
    mean = hl.agg.mean(hl.float(vds.GT.n_alt_alleles())),
    genotypes = hl.agg.collect(hl.float(vds.GT.n_alt_alleles())),
    phenotypes = hl.agg.collect(hl.float(vds.f1))
).rows()

groups = gts_as_rows.group_by(
  ld_block = gts_as_rows.LD_block
).aggregate(
  genotypes = hl.agg.collect(gts_as_rows.genotypes)
, ys = hl.agg.collect(gts_as_rows.phenotypes)
)

df = groups.to_spark()

def get_mean_square_error_LASSO(X, y):
    X = numpy.array(X)
    y = numpy.array(y[0])
    X = X.transpose()
    y = y.transpose()
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X, y)
    y_predict = clf.predict(X)
    #return float(y_predict)
    return float(mean_squared_error(y, y_predict))

get_mean_square_error_LASSO_udf = udf(get_mean_square_error_LASSO)

# try different parameters, get loss value for each LD block, sum them
print("current time is: ", time.asctime(time.localtime(time.time())))
df.select(get_mean_square_error_LASSO_udf("genotypes", "ys")).show()
print("current time is: ", time.asctime(time.localtime(time.time())))

def get_mean_square_error_ElasticNet(X, y):
    X = numpy.array(X)
    y = numpy.array(y[0])
    X = X.transpose()
    y = y.transpose()
    clf = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5)
    clf.fit(X, y)
    y_predict = clf.predict(X)
    return float(mean_squared_error(y, y_predict))

get_mean_square_error_ElasticNet_udf = udf(get_mean_square_error_ElasticNet)    

print("current time is: ", time.asctime(time.localtime(time.time())))
df.select(get_mean_square_error_ElasticNet_udf("genotypes", "ys").alias("mean_square_error")).show()
print("current time is: ", time.asctime(time.localtime(time.time())))






