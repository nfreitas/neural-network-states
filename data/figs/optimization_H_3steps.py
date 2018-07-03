import numpy as np
from matplotlib.pylab import plt

fig, ax = plt.subplots()

dt1 = np.linspace(0,500*.005,500)
mE1 = np.load('../mE_H_40_.5_.005_500.pkl')
sE1 = np.load('../sE_H_40_.5_.005_500.pkl')
ax.errorbar(dt1,mE1/40,sE1/(40*np.sqrt(1000)), label='$\\tau=0.005 \: J^{-1}$')

dt2 = np.linspace(0,1000*.0025,1000)
mE2 = np.load('../mE_H_40_.5_.0025_1000.pkl')
sE2 = np.load('../sE_H_40_.5_.0025_1000.pkl')
ax.errorbar(dt2,mE2/40,sE2/(40*np.sqrt(1000)), label='$\\tau=0.0025 \: J^{-1}$')

dt3 = np.linspace(0,2000*.00125,2000)
mE3 = np.load('../mE_H_40_.5_.00125_2000.pkl')
sE3 = np.load('../sE_H_40_.5_.00125_2000.pkl')
ax.errorbar(dt3,mE3/40,sE3/(40*np.sqrt(1000)), label='$\\tau=0.00125 \: J^{-1}$')

ax.plot(dt2,[-1.06348]*len(dt2),'-.',color='black', label='VMC')

ax.legend()

ax.set_xlabel('Imaginary time ( $1/J$ )')
ax.set_ylabel('Energy per spin $\epsilon$ ( $J$ )')

plt.show()
