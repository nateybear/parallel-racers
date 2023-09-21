import numpy as np
import pandas as pd

## pull in results from single and multithread and merge
multi = pd.read_csv(r'interim/results_multi.csv')
single = pd.read_csv(r'interim/results_single.csv')

final = multi.merge(single,how='inner',on='test')
final.to_csv(r'thread_test_results.csv',index=False)