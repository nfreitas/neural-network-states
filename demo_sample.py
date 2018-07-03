import numpy as np

import nqs
import models
import sampler

E = []

def measure(M,TI,S):

    e = TI.localEnergy(S.wf,S.st)
#    v = S.wf.varGradient(S.st)

    M['E'] = M['E'] + (e-M['E'])/S.sweeps

    E.append(e[0][0])
#    ga = v['a'] * e
#    gb = v['b'] * e
#    gW = v['W'] * e
#
#    M['ga'] = M['ga'] + (ga-M['ga'])/S.sweeps
#    M['gb'] = M['gb'] + (gb-M['gb'])/S.sweeps
#    M['gW'] = M['gW'] + (gW-M['gW'])/S.sweeps

wf = nqs.brbm_nqs.load_file('./carleo_code/Nqs/Ground/Ising1d_40_1_2.wf')

TI = models.TransIsing1D(nspins=40,hfield=1,pbc=False)

M = {'E':0,'ga':0,'gb':0,'gW':0}
S = sampler.metropolis(wf=wf)
S.afterSweep = lambda S: measure(M,TI,S)
S.run(nsweeps=3000,thermfactor=.1,output=True)
