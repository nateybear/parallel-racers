import numpy as np
import pandas as pd
import subprocess

# Run the various scripts (outside of program)
progs = ["multithread_test_py.py","singlethread_test_py.py","jit_multithread_test_py.py","explicit_multithread_test_py.py"]

t=0
for p in progs:
    t+=1
    subprocess.run(["python", p])
    print(t)
    

## pull in results from single and multithread (default threads) and merge
multi = pd.read_csv(r'interim/results_multi.csv')
single = pd.read_csv(r'interim/results_single.csv')

## pull in results from jit single and multithread (default threads)
multi_jit = pd.read_csv(r'interim/results_multi_jit.csv')
# single = pd.read_csv(r'interim/results_single.csv')               ##### have not created/run this yet

## pull in results from explicit 16 threads
multi_16 = pd.read_csv(r'interim/results_multi_16.csv')


final1 = multi.merge(single,how='inner',on='test')
final2 = final1.merge(multi_jit,how='inner',on='test')
final3 = final2.merge(multi_16,how='inner',on='test')

print(final3)

final3.to_csv(r'thread_test_results.csv',index=False)