import numpy as np
import pandas as pd

results_1 = pd.read_csv(r'outputs/results1.csv')
results_2 = pd.read_csv(r'outputs/results2.csv')
results_3 = pd.read_csv(r'outputs/results3.csv')

results = pd.concat([results_1,results_2,results_3],axis=0).reset_index()

results.to_csv(r'outputs/results_py.csv',index=False)

print(results)

