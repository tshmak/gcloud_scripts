# coding: utf-8
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

# ## key step
# ### 1. extract pca info, transform it to dataframe
# ### 2. build linear regression model, predict y and get y residuals
# ### 3. store y residuals in hail MatrixTable
# ### 4. run gwas and compare time

# Import a PLINK dataset (BED, BIM, FAM) as a MatrixTable
vds = hl.import_plink('gs://ukb_testdata/maf_0.01_10.bed',
                      'gs://ukb_testdata/maf_0.01_10.bim',
                      'gs://ukb_testdata/maf_0.01_10.fam' )

# Import delimited text file (text table) as Table
# import phenotype
table = (hl.import_table(
    'gs://ukb_testdata/sleep_duration.tsv',
    delimiter='\t', no_header=True, missing='NA', impute=True, types={'f0': hl.tstr}).key_by('f0'))
# table.show()
vds = vds.annotate_cols(**table[vds.s])
# print(vds.col.dtype.pretty())

dic = {}
for i in np.arange(1, 41):
    dic['pc' + str(i)] = hl.tfloat
pcas = hl.import_table('gs://ukb_testdata/covar.txt', delimiter=' ', types=dic).key_by('FID')
pcas = pcas.drop('IID')
vds = vds.annotate_cols(**pcas[vds.s])
vds.describe()
# pprint(vds.aggregate_cols(agg.stats(vds.f1)))

# transform hail matrixTable to pandas dataframe
temp = vds.cols().to_pandas()
covariant = temp[['pc'+str(i) for i in range(1, 12)]]
f1_y = temp["f1"]

#f1_y.values
covariant = covariant.fillna(0)
f1_y = f1_y.fillna(0)

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(covariant.values, f1_y.values)

# predict and get y residuals
f1_y_predict = regr.predict(covariant.values)
y_residual2 = list(f1_y - f1_y_predict)

# save datafram to bucket
# np.savetxt('y_residual.txt', y_residual2)
dat = pd.DataFrame(data=y_residual2, index=temp["s"], columns=["y_residual"])
# dat = pd.DataFrame([0, 0, 0, 0])
file_path = 'gs://ukb_testdata/zxli_test/testfile.csv'
with hl.hadoop_open(file_path,'w') as f:
    dat.to_csv(f)

# create new table from dataframe
t1 = hl.Table.parallelize([{"s":i[0],"y_residual":i[1]} for i in zip(temp["s"], y_residual2)], 
                          hl.tstruct(s=hl.tstr ,y_residual=hl.tfloat64), key='s')

# add y residual to hail matrixTable
vds = vds.annotate_cols(**t1[vds.s])

gwas = hl.linear_regression(y=vds.f1, x=vds.GT.n_alt_alleles(),
    covariates=[vds['pc'+str(i)] for i in range(1, 12)])

print("current time is: ", time.asctime(time.localtime(time.time())))
results = gwas.rows()
results.write('gs://ukb_testdata/zxli_test/GWAS_test', overwrite=True)
print("current time is: ", time.asctime(time.localtime(time.time())))

gwas2 = hl.linear_regression(y=vds.y_residual, x=vds.GT.n_alt_alleles())

print("current time is: ", time.asctime(time.localtime(time.time())))
results2 = gwas2.rows()
results2.write('gs://ukb_testdata/zxli_test/GWAS_test2', overwrite=True)
print("current time is: ", time.asctime(time.localtime(time.time())))

