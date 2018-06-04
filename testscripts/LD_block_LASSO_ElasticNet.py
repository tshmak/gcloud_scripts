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

mt = hl.balding_nichols_model(3, 100, 100)
gts_as_rows = mt.annotate_rows(
  mean = hl.agg.mean(hl.float(mt.GT.n_alt_alleles()))
, genotypes = hl.agg.collect(hl.float(mt.GT.n_alt_alleles()))
).rows()
groups = gts_as_rows.group_by(
  ld_block = gts_as_rows.locus.position // 10
).aggregate(
  genotypes = hl.agg.collect(gts_as_rows.genotypes)
, ys = hl.agg.collect(gts_as_rows.mean)
)

df = groups.to_spark()

import findspark
findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import udf

def get_intercept(X, y):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X, y)
    return float(clf.intercept_)

get_intercept_udf = udf(get_intercept)
df.select(get_intercept_udf("genotypes", "ys").alias("intercept")).show()

