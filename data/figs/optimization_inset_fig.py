import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()

dt1 = .005*np.arange(500)
mE1 = np.load('../mE_H_40_.5_.005_500.pkl')
sE1 = np.load('../sE_H_40_.5_.005_500.pkl')

ax.errorbar(dt1,mE1/40,sE1/(40*np.sqrt(1000)))
dt2 = .02*np.arange(125)
mE2 = np.load('../mE_H_40_.5_.02_125.pkl')
sE2 = np.load('../sE_H_40_.5_.02_125.pkl')

ax.errorbar(dt2,mE2/40,sE2/(40*np.sqrt(1000)))

ax.plot(dt2,[-1.06348]*len(dt2),'-.',color='black')

ax.legend(['$\\tau=.02$', '$\\tau=.005$', 'VMC'][::-1])

ax.set_xlabel('Imaginary time ( $1/J$ )')
ax.set_ylabel('Energy per spin $\epsilon$ ( $J$ )')

ax_ins = plt.axes((.53,.33,.35,.35))
ax_ins.errorbar(dt1[200:],mE1[200:]/40,sE1[200:]/(40*np.sqrt(1000)))
ax_ins.errorbar(dt2[50:],mE2[50:]/40,sE2[50:]/(40*np.sqrt(1000)))
ax_ins.plot(dt2[50:],[-1.06348]*len(dt2[50:]),'-.',color='black')


plt.show()
