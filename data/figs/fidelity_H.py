import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()

A = np.load('../fid_A_12_200_w.2.pkl')
M = np.load('../fid_M_12_200_w.2.pkl')
S = np.load('../fid_S_12_200_w.2.pkl')

ax.errorbar(A,1-M[:,0],S[:,0]/np.sqrt(200))
ax.errorbar(A,1-M[:,1],S[:,1]/np.sqrt(200))
ax.errorbar(A,1-M[:,2],S[:,2]/np.sqrt(200))


ax.legend(['Numerical method', 'Weak X', 'Strong X'])

ax.set_xlabel('$x$')
ax.set_ylabel('$\mathcal{I}$')

plt.show()
