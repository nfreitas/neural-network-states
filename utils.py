import numpy as np
import nqs
import sampler
import optimize
#import matplotlib.pyplot as plt


def max_abs(C, k=None):

    if k == None:

        M = []

        for j in range(len(C)):
            M.append(max_abs(C,j))

        m = np.max(M)

    else:

        m = C[k]

        for j in range(len(C)):
            if j == k:
                continue

            s = np.sign(np.real(m*np.conj(C[j])))
            m += s*C[j]


    return m


def solve_iter(f, x0, err=1e-10, max_steps=100):

    x = x0
    st = 0
    while st < max_steps:
        nx = f(x)

        d = np.sum(np.abs(nx-x)**2)

        st += 1
        x = nx
#        print(x,d)
        if d<err:
            break
#        print(x,d)

    return x

def approx1(W,b,x):

    w = np.sum(np.abs(W))
#    w = np.abs(max_abs(W))

    f = lambda s: np.cosh(s+b+x)*np.cosh(s+b-x)
    c0 = f(0)
    cp = f(w)
    cm = f(-w)

#    print(w, b, x)
#    print(c0, cp, cm)

    bp0 = 0
    def recur(bp):
        nbp = np.arccosh(((cp/c0)**(1/2))*np.cosh(bp))/2
        nbp -= np.arccosh(((cm/c0)**(1/2))*np.cosh(bp))/2
        return nbp
    bp = solve_iter(recur, bp0)

    beta = np.arccosh(((cp/c0)**(1/2))*np.cosh(bp))/(2*w)
    beta += np.arccosh(((cm/c0)**(1/2))*np.cosh(bp))/(2*w)

    alpha2 = c0/(np.cosh(bp)**2)

    return alpha2, beta, bp

def approx2(W, b, x):

    w = np.sum(np.abs(W))
#    print("aaaaaaaaaaaaaa", w)
#    w = np.abs(max_abs(W))

    fp = np.log(np.cosh(w+b+x)) - np.log(np.cosh(w+b-x))
    fm = np.log(np.cosh(w-b-x)) - np.log(np.cosh(w-b+x))

    beta = (fp+fm)/2
    alpha = (fp-fm)/(2*w)

    return alpha, beta

def approx3(W,b):

    w = np.sum(np.abs(W))
#    w = np.abs(max_abs(W))

    f = lambda s: np.cosh(s+b)
    c0 = f(0)
    cp = f(w)
    cm = f(-w)

#    print(w, b, x)
#    print(c0, cp, cm)

    bp0 = 0
    def recur(bp):
        nbp = np.arccosh(((cp/c0)**(1/2))*np.cosh(bp))/2
        nbp -= np.arccosh(((cm/c0)**(1/2))*np.cosh(bp))/2
        return nbp
    bp = solve_iter(recur, bp0)

    beta = np.arccosh(((cp/c0)**(1/2))*np.cosh(bp))/(2*w)
    beta += np.arccosh(((cm/c0)**(1/2))*np.cosh(bp))/(2*w)

    alpha2 = c0/(np.cosh(bp)**2)

    return alpha2, beta, bp

def compare_wfs(s1, s2):

    N = s1.nv

    p1 = np.array([np.exp(s1.logPsi_dict[k][0][0]) for k in range(2**N)])
    p1 = p1/np.exp(s1.log_norm())
    p2 = np.array([np.exp(s2.logPsi_dict[k][0][0]) for k in range(2**N)])
    p2 = p2/np.exp(s2.log_norm())
    ph1 = np.array([np.imag(s1.logPsi_dict[k][0][0]) for k in range(2**N)])
    ph2 = np.array([np.imag(s2.logPsi_dict[k][0][0]) for k in range(2**N)])

    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.abs(p1.flatten())**2)
    plt.plot(np.abs(p2.flatten())**2)

    plt.subplot(212)
    plt.plot(ph1)
    plt.plot(ph2)
    plt.show()

def fidelity(nqs1, nqs2, noPhase=False):

    nv = nqs1.nv

    f = 0
    n1 = 0
    n2 = 0

    for n in range(2**nv):
        v = intToReg(n,nv)

        l1 = nqs1.logPsi(v)
        l2 = nqs2.logPsi(v)

        if noPhase:
            l1 = np.real(l1)
            l2 = np.real(l2)

        f += np.exp(l1.conj()+l2)
        n1 += np.exp(l1.conj()+l1)
        n2 += np.exp(l2.conj()+l2)

#    print(n1,n2)
    return np.abs(f)**2/(n1*n2)

def regToInt(reg):

    s = 0
    for k in range(len(reg)):
        s += (2**k)*(1+reg[k])/2

    return int(s)

def intToReg(k,N):

    s = bin(k)[2:].zfill(N)
    r = np.array([int(b) for b in s]).reshape(N,1)
    r = -1 + 2*r

    return r[::-1]

def det_reduceUBM(ubm, nsteps=60, learning_rate=.1):

    # create and initialize rbm wave function
    rbm = nqs.brbm_nqs(ubm.nv, ubm.nh)
    rbm.setRandomParams()

    rbm.a = ubm.a
    rbm.b = ubm.b
    rbm.W = ubm.W
    rbm.Y = ubm.Y

#    rbm.a = np.real(ubm.a) + 0j
#    rbm.b[:ubm.nh] = np.real(ubm.b) + 0j
#    rbm.W[:ubm.nh,:ubm.nv] = np.real(ubm.W) + 0j
#    rbm.Y[:ubm.nv,:ubm.nv] = np.real(ubm.Y) + 0j

    # optimization
    O = optimize.det_fidelity_GA(rbm, ubm)
    #O = optimize.det_rel_entropy_GD(rbm, ubm)
    O.opt(nsteps,learning_rate)

    new = rbm.asUBM()

    # now I have to take care of the phases of the coefficients
#    a, Y = optimize.opt_phases(ubm.logPsi_dict, ubm.nv, 5000, 1e-2,1)
#
#    new.Y += 1j*Y
#    new.a += 1j*a

#    return new, O.rel_entropy[-1], a, Y, ubm
    return new, O.fidelity[-1]
#    return new, O.rel_entropy[-1]

def reduceUBM(ubm, nh_factor=1, nsteps=300, nsweeps=200, learning_rate=1e-2):

    # create and initialize rbm wave function
    rbm = nqs.brbm_nqs(ubm.nv, int(ubm.nh*nh_factor))
    rbm.setRandomParams()

    rbm.a = np.real(ubm.a) + 0j
    rbm.b[:ubm.nh] = np.real(ubm.b) + 0j
    rbm.W[:ubm.nh,:ubm.nv] = np.real(ubm.W) + 0j
    rbm.Y[:ubm.nv,:ubm.nv] = np.real(ubm.Y) + 0j

    print(np.real(ubm.Y))

    # optimize with real parameters to adjust the absolute value
    # of each coefficient in the ubm
    S = sampler.metropolis(wf=rbm)
    O = optimize.fidelity_GA(S, ubm)
    O.opt(nsteps,nsweeps,learning_rate)

    # now I have to take care of the phases of the coefficients
#    a, Y = optimize.opt_phases(ubm.logPsi_dict, ubm.nv, 1000, 1e-2)

    # create a new ubm
    new = rbm.asUBM()
#    new.Y = 1j*Y
#    new.a += 1j*a

    return new, O.fidelity[-1]


#def g1(ind,b,W,v):
#
#    a = b + W.dot(v)
#
#    g = np.exp(a[ind])/(2*np.cosh(a[ind]))
#
#    return g
#
#def g1_exp(ind, b, W):
#
#    w = W[ind,:].reshape((len(W[ind,:]),1))
#
#    g = np.exp(b[ind])/(2*np.cosh(b[ind]))
#
#    d = np.exp(b[ind])/(2*np.cosh(b[ind])) - np.exp(b[ind])*np.tanh(b[ind])/(2*np.cosh(b[ind]))
#    d = d*w
#
#    H = np.exp(b[ind])/(2*np.cosh(b[ind])) - np.exp(b[ind])*np.tanh(b[ind])/np.cosh(b[ind])
#    H += -1*np.exp(b[ind])/(2*(np.cosh(b[ind])**3)) + np.exp(b[ind])*(np.tanh(b[ind])**2)/2
#    H = H*(d.dot(d.T))
#
#    return g, d, H
#
#
#def g2(ind,b,x,W,v):
#
#    a = b + W.dot(v)
#    F = np.cosh(a+x)/np.cosh(a)
#
#    return np.prod(np.delete(F,ind))
#
#def g2_exp(ind,b,x,W):
#
#    M, N = W.shape
#
#    cbx = np.cosh(b+x)
#    cb = np.cosh(b)
#    sbx = np.sinh(b+x)
#    sb = np.sinh(b)
#
#    tb  = sb/cb
#
#    fcc = cbx/cb
#
#    g = np.prod(np.delete(fcc,ind))
#
#    d = 0
#    H = 0
#    for l in range(M):
#        if l == ind:
#            continue
#        wl = W[l,:].reshape((N,1))
#        f = np.prod(np.delete(fcc,[ind,l]))
#        d += f*(sbx[l]/cb[l]-cbx[l]*tb[l]/cb[l])*wl
#
#        f *= cbx[l]/cb[l] - 2*sbx[l]*sb[l]/(cb[l]**3) + cbx[l]*sb[l]/(cb[l]**3) - cbx[l]/(cb[l]**3)
#        H += f*(wl.dot(wl.T))
#
#        for m in range(M):
#            if m==ind or m==l:
#                continue
#
#            wm = W[m,:].reshape((N,1))
#
#            f = np.prod(np.delete(fcc,[ind,l,m]))
#
#            f *= sbx[l]/cb[l]-cbx[l]*tb[l]/cb[l]
#            f *= sbx[m]/cb[m]-cbx[m]*tb[m]/cb[m]
#            H += f*(wl.dot(wm.T))
#
#    return g, d, H
#
#def test_exp(ind):
#
#    N = 5
#
#    b = -1+2*np.random.rand(N).reshape((N,1))
#    x = -1+2*np.random.rand(N).reshape((N,1))
#    W = -1+2*np.random.rand(N,N)
#    W = W
#
#    g, d, H = func_exp(ind,b,x,W)
#
#    for n in range(2**N):
##        print(n,2**nv)
#        s = bin(n)[2:].zfill(N)
#        Y = np.array([int(b) for b in s]).reshape(N,1)
#        Y = -1 + 2*Y
#
#        Y = Y
#        print(func(ind,b,x,W,Y), g+d.T.dot(Y)+Y.T.dot(H.dot(Y))/2)
#
#def func(ind,b,x,W,v):
#
#    f = g1(ind,b,W,v)*g2(ind,b,x,W,v) + g1(ind,-1*b,-1*W,v)*g2(ind,b,-1*x,W,v)
#
#    return f
#
#def func_exp(ind,b,x,W):
#
#    g1p, g1p_d, g1p_H = g1_exp(ind,b,W)
#    g1m, g1m_d, g1m_H = g1_exp(ind,-b,-W)
#
#    g2p, g2p_d, g2p_H = g2_exp(ind,b,x,W)
#    g2m, g2m_d, g2m_H = g2_exp(ind,b,-1*x,W)
#
#    f = g1p*g2p + g1m*g2m
#
#    d = g1p_d*g2p + g1p*g2p_d - g1m_d*g2m + g1m*g2m_d
#
#    H = g1p_H*g2p + g1p_d.dot(g2p_d.T)
#    H += g2p_d.dot(g1p_d.T) + g1p*g2p_H
#    H += g1m_H*g2m - g1m_d.dot(g2m_d.T)
#    H += -g2m_d.dot(g1m_d.T) + g1m*g2m_H
#
#    return f, d, H
#
#def test_red():
#
#    N = 5
#
#    S1 = nqs.brbm_nqs(N,N)
#    S1.setRandomParams()
##    S1.b = np.zeros_like(S1.b)
#    S1.W = S1.W
#
#    S2 = S1.asUBM()
#    S2.X[int(N/2),:] = np.random.rand(N)
#    S2.X[:,int(N/2)] = S2.X[int(N/2),:]
#    S2.X[int(N/2),int(N/2)] = 0
#
##    print(fidelity(S1,S2.asRBM()))
#
#    S2.delHiddenInt(int(N/2))
##    S2.visibleIntToHiddenNodes()
#
#    f = fidelity(S1,S2.asRBM())
#
#    return f[0,0]
