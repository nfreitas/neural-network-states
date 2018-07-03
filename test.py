import numpy as np
import sampler
import models
import nqs
import utils

E = []

def measure(M,TI,S):

    e = TI.localEnergy(S.wf,S.st)
    E.append(e[0])
    M['E'] = M['E'] + (e-M['E'])/S.sweeps
    M['E2'] = M['E2'] + (e**2-M['E2'])/S.sweeps

    M['C'] = M['C'] + (S.st[0]*S.st - M['C'])/S.sweeps

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

        tao = trotter_dt*(decay_rate**(k/trotter_st))
        mod.applyTrotterStep(wf,-1j*tao,renorm=True)

        wf.rmHiddenNodes(keep=2*N)
#        wf.W = wf.W/np.max(wf.W)

        print(k)
        if True:#k%50==0:#k==trotter_st-1:
            print("Measuring Energy...")
            M = {'E':0, 'E2':0, 'C':0}
            S = sampler.metropolis(wf=wf.asRBM())
            S.afterSweep = lambda S: measure(M,mod,S)
            S.run(nsweeps=1000, output=False)
            print(M['E'], np.sqrt(M['E2']-M['E']**2))
            meanE.append(M['E'][0,0])
            stdE.append(np.sqrt(M['E2']-M['E']**2)[0,0])

    return mod, wf, meanE, stdE

def run_inf_trotter(N,h,st,dt,init_wf=None):

    trotter_st = st
    trotter_dt = dt
    decay_rate = 1

    mod = models.TransIsing1D(N,h,pbc=True)

    if init_wf == None:
        wf = nqs.brbm_nqs(N,N)
        #wf.W = .1*np.random.rand(N,N).astype(complex)
        wf.W[:N,:] = .1*np.eye(N,dtype=complex)
        #        for k in range(N):
        #            wf.W[N+k,k] = .2
        #        for k in range(N-1):
        #            wf.W[N+k,k+1] = .2
    else:
        wf = init_wf

    meanE = []
    stdE = []

    print("Applying Trotter Evolution...")
    for k in range(trotter_st):

#        print('a', wf.nh)
#        wf.rmVisibleInt()
#        print('b', wf.nh)
#        wf.rmHiddenNodes(keep=4*N)

        print(k)
        if k%5==0:#True:#k==trotter_st-1:
            print("Measuring Energy...")
            M = {'E':0, 'E2':0, 'C':0}
            S = sampler.metropolis(wf=wf)
            S.afterSweep = lambda S: measure(M,mod,S)
            S.run(nsweeps=1000, output=False)
            print(M['E'], np.sqrt(M['E2']-M['E']**2))
            meanE.append(M['E'][0,0])
            stdE.append(np.sqrt(M['E2']-M['E']**2)[0,0])

        tao = trotter_dt*(decay_rate**(k/trotter_st))
        mod.applyTrotterStep(wf,-1j*tao,inf=True)

    return mod, wf, meanE, stdE

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
            print("Measuring Energy...")
            M = {'E':0, 'E2':0, 'C':0}
            S = sampler.metropolis(wf=wf)
            S.afterSweep = lambda S: measure(M,mod,S)
            S.run(nsweeps=1000, output=False)
            print(M['E'], np.sqrt(M['E2']-M['E']**2))
            meanE.append(M['E'][0,0])
            stdE.append(np.sqrt(M['E2']-M['E']**2)[0,0])

        tao = trotter_dt*(decay_rate**(k/trotter_st))
        mod.applyTrotterStep(wf,-1j*tao,inf=True,hidden_crz=True)

    return mod, wf, meanE, stdE

def run_dt(N,h,DT,total_t):

    meanE = []
    stdE = []

    for dt in DT:

        mod, wf, mE, sE = run_trotter(N,h,int(total_t/dt),dt)

        meanE.append(mE[0])
        stdE.append(sE[0])


    return meanE, stdE



def run_h(N):

    H = np.linspace(.1,1.9,30)
    E = []
    TE = []
    S = []
    F = []

    for h in H:

        mod, wf, e, s, c = run_trotter(N,h)

        E.append(e)
        S.append(s)

        ha = mod.buildFullH()
        en, ev = np.linalg.eig(ha)

        TE.append(np.min(en))
        v = ev[:,np.argmin(en)]

        rbm = wf.asRBM()
        logPsi = [rbm.logPsi(utils.intToReg(k,rbm.nv))[0,0] for k in range(2**rbm.nv)]
        Psi = np.exp(np.array(logPsi)-rbm.log_norm()[0,0])
        fid = np.abs(np.sum(Psi*v.flatten()))

        print('fidelity', fid)
        F.append(fid)

    return H, E, S, TE, F
