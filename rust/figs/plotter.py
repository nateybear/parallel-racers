import pandas as pd
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
import seaborn as sns 


P = [2, 4, 8, 16] 

dfls = len(P)*[0]
for i in range(len(P)):
    # pull the julia file 
    dfJ = pd.read_csv("../out/julia_P{}.csv".format(P[i]))
    dfJ['P'] = P[i]
    # pull the matlab file 
    dfM = pd.read_csv("../out/matlab_P{}_N10.csv".format(P[i]))
    dfM['P'] = P[i]
    # pull the python file 
    dfP = pd.read_csv("../out/results_py_p{}.csv".format(P[i]))
    dfP['P'] = P[i]
    # store in a tuple 
    dfls[i] = (dfJ, dfM, dfP)

# pull in full tbls 
dfJ = pd.concat([tup[0] for tup in dfls]).reset_index(drop = True)
dfM = pd.concat([tup[1] for tup in dfls]).reset_index(drop = True)
dfP = pd.concat([tup[2] for tup in dfls]).reset_index(drop = True)
# adjust col names for python  
dfP.columns = dfM.columns 


# {1:"Vectorized", 2:"Serial", 3:"ParFor"}
dfJV =  dfJ[dfJ.method == 1].groupby('size')['elapsed'].mean().reset_index()
dfMV =  dfM[dfM.Method == 1].groupby('Size')['Elapsed'].mean().reset_index()
dfPV =  dfP[dfP.Method == 1].groupby('Size')['Elapsed'].mean().reset_index()


# PLOT STUFF
csfont          = {'fontname':"Liberation Serif", 'fontsize':16}
palette         = ["#FF6700", "#FCB07E", "#6B717E", "#3A6EA5", "#004E98", "#070707"]
linetypes       = [ "--", "-", ":", "-."]

##################################
### Vectorized Implementation  ### 
##################################

# --- Levels --- #  
sns.set(style="white",color_codes=False)
fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
plt.plot(np.log10(dfJV['size']), dfJV.elapsed, label = "Julia",  ls = linetypes[0], color = palette[0], lw=3)
plt.plot(np.log10(dfMV.Size), dfMV.Elapsed, label = "Matlab", ls = linetypes[1], color = palette[2], lw=3)
plt.plot(np.log10(dfPV.Size), dfPV.Elapsed, label = "Python", ls = linetypes[2], color = palette[3], lw=3)
L  = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 12) 
plt.xlabel(r'$\log(N)$', **csfont)
plt.ylabel("Runtime (Sec)", **csfont) 
# plt.xlim([0.85,1.8])
# plt.ylim([0.85,5.0])
plt.yticks(fontname = "Liberation Serif", fontsize = 14)
plt.xticks([3,4,5,6,7],fontname = "Liberation Serif", fontsize = 14)
# plt.show()
plt.savefig("./vectorizedRT.pdf",bbox_inches='tight',format= "pdf",dpi=600)

# --- Normalized by best parallel --- #
fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
plt.plot(np.log10(dfJV['size']), dfJV.elapsed/dfJ[dfJ.method == 3].groupby('size')['elapsed'].min().values, label = "Julia",  ls = linetypes[0], color = palette[0], lw=3)
plt.plot(np.log10(dfMV.Size), dfMV.Elapsed/dfM[dfM.Method == 3].groupby('Size')['Elapsed'].min().values, label = "Matlab", ls = linetypes[1], color = palette[2], lw=3)
plt.plot(np.log10(dfPV.Size), dfPV.Elapsed/dfP[dfP.Method == 3].groupby('Size')['Elapsed'].min().values, label = "Python", ls = linetypes[2], color = palette[3], lw=3)
L  = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 12) 
plt.xlabel(r'$\log(N)$', **csfont)
plt.ylabel("Runtime Vectorized/Parallel (Sec)", **csfont) 
# plt.xlim([0.85,1.8])
# plt.ylim([0.85,5.0])
plt.yticks(fontname = "Liberation Serif", fontsize = 14)
plt.xticks([3,4,5,6,7],fontname = "Liberation Serif", fontsize = 14)
# plt.show()
plt.savefig("./vectorizedNormRT.pdf",bbox_inches='tight',format= "pdf",dpi=600)

##########################################
### Parallel vs Serial Implementation  ### 
##########################################

# --- Julia --- # 
fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
for i in range(len(P)):
    yvals = dfJ.loc[(dfJ.method == 3) & (dfJ.P == P[i]), 'elapsed'].values/dfJ.loc[(dfJ.method == 2) & (dfJ.P == P[i]), 'elapsed'].values
    xvals = np.log10(dfJV['size'])
    plt.plot(xvals, yvals, label = "{} Processors".format(P[i]),  ls = linetypes[i], color = palette[i], lw=3)
L  = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 12) 
plt.xlabel(r'$\log(N)$', **csfont)
plt.ylabel("Runtime Parallel/Serial (Seconds)", **csfont) 
plt.axhline(y=1, color = palette[5], linestyle=':', lw = 2)
# plt.xlim([0.85,1.8])
plt.ylim([0.0,10.0])
plt.yticks(fontname = "Liberation Serif", fontsize = 14)
plt.xticks([3,4,5,6,7],fontname = "Liberation Serif", fontsize = 14)
# plt.show()
plt.savefig("./juliaNormParFor.pdf",bbox_inches='tight',format= "pdf",dpi=600)


# --- Matlab --- # 
fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
for i in range(len(P)):
    yvals = dfM.loc[(dfM.Method == 3) & (dfM.P == P[i]), 'Elapsed'].values/dfM.loc[(dfM.Method == 2) & (dfM.P == P[i]), 'Elapsed'].values
    xvals = np.log10(dfMV['Size'])
    plt.plot(xvals, yvals, label = "{} Processors".format(P[i]),  ls = linetypes[i], color = palette[i], lw=3)
L  = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 12) 
plt.xlabel(r'$\log(N)$', **csfont)
plt.ylabel("Runtime Parallel/Serial (Seconds)", **csfont) 
plt.axhline(y=1, color = palette[5], linestyle=':', lw = 2)
# plt.xlim([0.85,1.8])
plt.ylim([0.0,1000.0])
plt.yticks(fontname = "Liberation Serif", fontsize = 14)
plt.xticks([3,4,5,6,7],fontname = "Liberation Serif", fontsize = 14)
# plt.show()
plt.savefig("./matlabNormParFor.pdf",bbox_inches='tight',format= "pdf",dpi=600)


# --- Matlab --- # 
fig, ax = plt.subplots(figsize=(6.4, 4.8*(1.6/2)))
for i in range(len(P)):
    yvals = dfP.loc[(dfP.Method == 3) & (dfP.P == P[i]), 'Elapsed'].values/dfP.loc[(dfP.Method == 2) & (dfP.P == P[i]), 'Elapsed'].values
    xvals = np.log10(dfPV['Size'])
    plt.plot(xvals, yvals, label = "{} Processors".format(P[i]),  ls = linetypes[i], color = palette[i], lw=3)
L  = plt.legend()
plt.setp(L.texts, family='Liberation Serif', fontsize = 12) 
plt.xlabel(r'$\log(N)$', **csfont)
plt.ylabel("Runtime Parallel/Serial (Seconds)", **csfont) 
plt.axhline(y=1, color = palette[5], linestyle=':', lw = 2)
# plt.xlim([0.85,1.8])
# plt.ylim([0.0,10.0])
plt.yticks(fontname = "Liberation Serif", fontsize = 14)
plt.xticks([3,4,5,6,7],fontname = "Liberation Serif", fontsize = 14)
# plt.show()
plt.savefig("./pythonNormParFor.pdf",bbox_inches='tight',format= "pdf",dpi=600)
