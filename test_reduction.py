import numpy as np
import nqs
import utils


def randomTest(N,M,a):

    S = nqs.wrbbm_nqs(N,M)
    T = nqs.bbm_nqs(N,M)
    U = nqs.bbm_nqs(N,M)
    V = nqs.bbm_nqs(N,M)

    S.setRandomParams()
    S.b = np.zeros_like(S.b)
    S.a = np.zeros_like(S.a)
    S.Y = np.zeros_like(S.Y)
    S.X[:-1,:-1] = 0
    S.X = a*S.X
    S.W = .2*S.W

    T.setParams(S.params())
    T.renormalize1()

    U.setParams(S.params())
#    U.X = np.zeros_like(U.X)
    U.renormalize2()

    V.setParams(S.params())
    V.renormalize3()

    f1 = utils.fidelity(S,T.asRBM())
    f2 = utils.fidelity(S,U.asRBM())
    f3 = utils.fidelity(S,V.asRBM())

#    print(S.W[-1,:])
#    print(T.W[-1,:])
    return f1, f2, f3


def samples(N,K,a):

    F1 = []
    F2 = []
    F3 = []

    for k in range(K):

        f1, f2, f3 = randomTest(N,N,a)

        F1.append(f1[0,0])
        F2.append(f2[0,0])
        F3.append(f3[0,0])

        print(k, F1[-1],F2[-1],F3[-1])

    return F1, F2, F3

def fid_vs_a(N,K,A):

    M = []
    S = []

    for a in A:

        F1, F2, F3 = samples(N,K,a)

        M.append((np.mean(F1), np.mean(F2), np.mean(F3)))
        S.append((np.std(F1), np.std(F2), np.std(F3)))

    return M, S
