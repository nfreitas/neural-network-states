import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()

A = np.loadtxt('../sx_vs_t_exact_24_2_200_.005.csv')
t2 = A[:,0]
sx2 = A[:,1]

A = np.loadtxt('../sx_vs_t_exact_20_0.2_200_.005.csv')
tp2 = A[:,0]
sxp2 = A[:,1]

ax.plot(tp2,np.real(sxp2), 'g--', label='Exact rep. $h=1/5$')
ax.plot(t2,np.real(sx2), 'k--', label='Exact rep. $h=2$')

#sxp2 = np.load('../sx_I_14_.2_.002_500.pkl')
#ssxp2 = np.load('../ssx_I_14_.2_.002_500.pkl')
#ax.errorbar(dt1,sxp2,ssxp2/(np.sqrt(10000)), marker='.', ls='none')
#

dt1 = np.linspace(0,200*.005,200)[::3]
sxp2 = np.load('../sx_I_24_0.2_.005_200.pkl')[::3]
ssxp2 = np.load('../ssx_I_24_0.2_.005_200.pkl')[::3]
ax.errorbar(dt1,sxp2,ssxp2/(np.sqrt(5000)), marker='.', ls='none',color='blue', label='RBM-NNS. $h=1/5$')

sx2 = np.load('../sx_I_24_2_.005_200.pkl')[::3]
ssx2 = np.load('../ssx_I_24_2_.005_200.pkl')[::3]
ax.errorbar(dt1,sx2,ssx2/(np.sqrt(5000)), marker='.', ls='none',color='orange', label='RBM-NNS. $h=2$')

ax.legend()

ax.set_xlabel('t ( $1/J$ )')
ax.set_ylabel('$\\langle \\sigma^x \\rangle (t)$')

plt.show()
