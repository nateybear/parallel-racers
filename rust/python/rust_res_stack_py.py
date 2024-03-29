import numpy as np
import pandas as pd
import subprocess


## run component programs independently
# Run the VFI script
subprocess.run(["python", "rust_VFI_py.py"])

# Run the implicit parallelization and serial script
subprocess.run(["python", "rust_serial_py.py"])

# Run the multiprocess
subprocess.run(["python", "rust_mp_py.py"])


## import results of vectorized, serial and multiprocess tests
def concat_res(n_processes):
    results_1 = pd.read_csv(r'outputs/results1.csv')
    results_2 = pd.read_csv(r'outputs/results2.csv')
    results_3 = pd.read_csv(f'outputs\\results3_p{n_processes}.csv')

    results = pd.concat([results_1,results_2,results_3],axis=0).reset_index().drop(columns=['index'])

    results.to_csv(f'outputs\\results_py_p{n_processes}.csv',index=False)

    print(results)

concat_res(n_processes=2)
concat_res(n_processes=4)
concat_res(n_processes=8)
concat_res(n_processes=16)
