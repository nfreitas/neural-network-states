import numpy as np
import sampler
import models
import nqs
import utils

E = []

def measure_energy(M,TI,S):

    e = TI.localEnergy(S.wf,S.st)
    E.append(e[0])
    M['E'] = M['E'] + (e-M['E'])/S.sweeps
    M['E2'] = M['E2'] + (e**2-M['E2'])/S.sweeps

    M['C'] = M['C'] + (S.st[0]*S.st - M['C'])/S.sweeps

def measure_sigmax(M,TI,S):

    e = np.real(np.exp(S.wf.logPoP(S.st,[0])))
    E.append(e)
    M['E'] = M['E'] + (e-M['E'])/S.sweeps
    M['E2'] = M['E2'] + (e**2-M['E2'])/S.sweeps

def run_trotter(N,h,st,dt,init_wf=None):

    trotter_st = st
    trotter_dt = dt
    decay_rate = 1

    mod = models.TransIsing1D(N,h,pbc=True)

    if init_wf == None:
        wf = nqs.bbm_nqs(N,0)
    else:
        wf = init_wf

    meanE = []
    stdE = []

    print("Applying Trotter Evolution...")
    for k in range(trotter_st):

        wf.rmHiddenNodes(keep=1*N)

        print(k)
        if True:#k%5==0:#k==trotter_st-1:
            print("Measuring Energy...")
            M = {'E':0, 'E2':0, 'C':0}
            S = sampler.metropolis(wf=wf.asRBM())
            S.afterSweep = lambda S: measure_sigmax(M,mod,S)
            S.run(nsweeps=10000, output=False)
            print(M['E'], np.sqrt(M['E2']-M['E']**2))
            meanE.append(M['E'][0,0])
            stdE.append(np.sqrt(M['E2']-M['E']**2)[0,0])

        tao = trotter_dt*(decay_rate**(k/trotter_st))
        mod.applyTrotterStep(wf,tao,renorm=True)

    return mod, wf, meanE, stdE

def run_inf_trotter(N,h,st,dt,init_wf=None):

    trotter_st = st
    trotter_dt = dt
    decay_rate = 1

    mod = models.TransIsing1D(N,h,pbc=True)

    if init_wf == None:
        wf = nqs.brbm_nqs(N,N)
        wf.W[:N,:] = .1*np.eye(N,dtype=complex)
    else:
        wf = init_wf

    meanS = []
    stdS = []

    print("Applying Trotter Evolution...")
    for k in range(trotter_st):

        print(k)
        if True:#k%5==0:#True:#k==trotter_st-1:
            print("Measuring SigmaX...")
            M = {'E':0, 'E2':0}
            S = sampler.metropolis(wf=wf)
            S.afterSweep = lambda S: measure_sigmax(M,mod,S)
            S.run(nsweeps=5000, output=False)
            print(M['E'], np.sqrt(M['E2']-M['E']**2))
            meanS.append(M['E'][0,0])
            stdS.append(np.sqrt(M['E2']-M['E']**2)[0,0])

        tao = trotter_dt*(decay_rate**(k/trotter_st))
        mod.applyTrotterStep(wf,tao,inf=True)

    return mod, wf, meanS, stdS

def run_inf_trotter2(N,h,st,dt,init_wf=None):

    trotter_st = st
    trotter_dt = dt
    decay_rate = 1

    mod = models.TransIsing1D(N,h,pbc=True)

    if init_wf == None:
        wf = nqs.brbm_nqs(N,0)
    else:
        wf = init_wf

    meanE = []
    stdE = []

    print("Applying Trotter Evolution...")
    for k in range(trotter_st):

#        wf.rmHiddenNodes(keep=4*N)

        print(k)
        if True:#k==trotter_st-1:
            print("Measuring SigmaX...")
            M = {'E':0, 'E2':0, 'C':0}
            S = sampler.metropolis(wf=wf)
            S.afterSweep = lambda S: measure_sigmax(M,mod,S)
            S.run(nsweeps=1000, output=False)
            print(M['E'], np.sqrt(M['E2']-M['E']**2))
            meanE.append(M['E'][0,0])
            stdE.append(np.sqrt(M['E2']-M['E']**2)[0,0])

        tao = trotter_dt*(decay_rate**(k/trotter_st))
        mod.applyTrotterStep(wf,tao,inf=True,hidden_crz=True)

    return mod, wf, meanE, stdE

def run_real(N,h,st,dt):

    mod = models.TransIsing1D(N,h,pbc=True)
    H = mod.buildFullH()

    A = np.eye(2**N,dtype=complex) - 1j*dt*H #- (dt**2)*H.dot(H)/2

    psi = np.ones((2**N,1),dtype=complex)/(2**(N/2))

    SX = []

    for k in range(st):

        print(k)
        if k%5==0:
            print('measuring sx')
            SX.append(sigmax(psi))

        psi = A.dot(psi)

    return SX

def sigmax(psi):

    S = 0
    N = 0
    for k in range(len(psi)):
        if k%2 == 0:
            kp = k+1
        else:
            kp = k-1
        S += psi[k]*np.conj(psi[kp])
        N += psi[k]*np.conj(psi[k])

    return S/N
