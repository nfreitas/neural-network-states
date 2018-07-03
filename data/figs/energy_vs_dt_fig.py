import matplotlib.pyplot as plt
import numpy as np

fig, (ax0, ax1) = plt.subplots(nrows=2)

dt = np.load('../dt_H_40_.5_vs_dt.pkl')[1:]
mE = np.load('../mE_H_40_.5_vs_dt.pkl')[1:]
sE = np.load('../sE_H_40_.5_vs_dt.pkl')[1:]

ax0.errorbar(dt,mE/40,sE/(40*np.sqrt(1000)),fmt='-x')
ax0.plot(dt,[-1.06348]*len(dt),'-.',color='black')
#ax0.set_xlabel('Time step')
ax0.set_ylabel('Energy per spin $\epsilon$ ( $J$ )')

ax0.legend(['Method I', 'VMC'][::-1])
#ax0.set_title('(a) $N=40$, $h=0.5$', loc='left')
ax0.text(0.003,-1.045,'(a) $N=40$, $h=0.5$',fontsize=12)

dt = np.load('../dt_H_40_1_vs_dt.pkl')[0:]
mE = np.load('../mE_H_40_1_vs_dt.pkl')[0:]
sE = np.load('../sE_H_40_1_vs_dt.pkl')[0:]

ax1.errorbar(dt,mE/40,sE/(40*np.sqrt(1000)),fmt='x-')
ax1.plot(dt,[-1.27243]*len(dt),'-.',color='black')
ax1.set_xlabel('Time step ( 1/J )')
ax1.set_ylabel('Energy per spin $\epsilon$ ( $J$ )')

ax1.legend(['Method I', 'VMC'][::-1])
#ax1.set_title('(b) $N=40$, $h=1$', loc='left')
ax1.text(0.003,-1.10,'(b) $N=40$, $h=1$',fontsize=12)


plt.show()
