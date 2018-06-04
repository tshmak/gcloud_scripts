import hail as hl
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

from pyspark.sql.functions import udf

def get_intercept(X, y):
    from sklearn import linear_model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X, y)
    return float(clf.intercept_)

get_intercept_udf = udf(get_intercept)
df.select(get_intercept_udf("genotypes", "ys").alias("intercept")).show()
