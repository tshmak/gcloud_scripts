nsample_select = 10000
nsnps_select = 100
selectrow = [False] * 369578
selectcol = [False] * 300163
selectrow[0:(nsnps_select - 1)] = [True] * nsnps_select
selectcol[0:(nsample_select - 1)] = [True] * nsample_select
selrow = hl.literal(selectrow)
selcol = hl.literal(selectcol)

ss = vds.select_cols(selcol)