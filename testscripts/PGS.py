# one statment
mt.annotate_cols(prs = hl.agg.sum(mt.GT.n_alt_alleles() * mt.linreg.beta))
