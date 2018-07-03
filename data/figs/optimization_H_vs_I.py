import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

dt1 = np.linspace(0,1000*.005,200)
mE1 = np.load('../mE_H_40_.5_.005_1000.pkl')
sE1 = np.load('../sE_H_40_.5_.005_1000.pkl')

ax.errorbar(dt1,mE1/40,sE1/(40*np.sqrt(1000)))

dt2 = np.linspace(0,1000*.005,200)
mE2 = np.load('../mE_I_40_.5_.005_1000.pkl')
sE2 = np.load('../sE_I_40_.5_.005_1000.pkl')

ax.errorbar(dt2,mE2/40,sE2/(40*np.sqrt(1000)))

ax.plot(dt2,[-1.06348]*len(dt2),'-.',color='black')

ax.legend(['Method II, $\\tau = 0.005 \: J^{-1}$', 'Method I, $\\tau=0.005 \: J^{-1}$', 'VMC'][::-1])

ax.set_xlabel('Imaginary time ( $1/J$ )')
ax.set_ylabel('Energy per spin $\epsilon$ ( $J$ )')

plt.show()
