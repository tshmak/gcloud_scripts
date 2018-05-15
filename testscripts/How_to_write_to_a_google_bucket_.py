import hail as hl
import pandas as pd

dat = pd.DataFrame([0, 0, 0, 0])
file_path = 'gs://ukb_testdata/testfile.csv'

with hl.hadoop_open(file_path,'w') as f:
    dat.to_csv(f)