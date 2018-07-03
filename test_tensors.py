import utils
import numpy as np
import scipy.optimize as opt


def kron2(a,b):

    if a==b:
        return 1
    else:
        return 0

def kron4(a,b,c,d):

    if a==b==c==d:
        return 1
    else:
        return 0

def tensor_T(C2,A):

    N,M  = A.shape

    T = np.zeros((N,N,N,N))

    AC = A.dot(C2)

    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):

                    for j in range(M):
                        T[a,b,c,d] += A[a,j]*A[b,j]*AC[c,j]*AC[d,j]

    return T


def tensor_R(C2,A):

    N,M  = A.shape

    R = np.zeros((N,N,N,N))

    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):

                    for j in range(M):
                        for l in range(M):
                            R[a,b,c,d] += (C2[j,l]**2)*A[a,j]*A[b,j]*A[c,l]*A[d,l]

    return R

def corr_h(X):

    N = 0
    C2 = np.zeros_like(X)
    C4 = np.zeros((4,4,4,4))

    for k in range(2**4):

        S = utils.intToReg(k,4)

        P = np.exp(S.T.dot(X.dot(S))/2)
        N += P

        for n in range(4):
            for m in range(4):

                C2[n,m] += S[n]*S[m]*P

                for r in range(4):
                    for s in range(4):

                        C4[n,m,r,s] += P*S[n]*S[m]*S[r]*S[s]

    return C2/N, C4/N


def corr_q(C4, A):

    N,M  = A.shape

    CQ4 = np.zeros((N,N,N,N))

    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):

                    for j in range(M):
                        for k in range(M):
                            for l in range(M):
                                for m in range(M):
                                    CQ4[a,b,c,d] += A[a,j]*A[b,k]*A[c,l]*A[d,m]*C4[j,k,l,m]

    return CQ4


def tensor4_q(C2, A):

    N,M  = A.shape

    Q4 = np.zeros((N,N,N,N))

    AA = A.dot(A.T)
    ACA = A.dot(C2.dot(A.T))

    for a in range(N):
        for b in range(N):
            for c in range(N):
                for d in range(N):

                    Q4[a,b,c,d]+= ACA[a,b]*ACA[c,d]/3
                    Q4[a,b,c,d]+= (AA[a,b]*ACA[c,d]-T[a,b,c,d])*2/3
                    Q4[a,b,c,d]+= (AA[c,d]*ACA[a,b]-T[c,d,a,b])*2/3
                    Q4[a,b,c,d]+= (-1*AA[a,b]*AA[c,d]+R[a,b,c,d])*2/3

                    Q4[a,b,c,d]+= ACA[a,c]*ACA[b,d]/3
                    Q4[a,b,c,d]+= (AA[a,c]*ACA[b,d]-T[a,c,b,d])*2/3
                    Q4[a,b,c,d]+= (AA[b,d]*ACA[a,c]-T[b,d,a,c])*2/3
                    Q4[a,b,c,d]+= (-1*AA[a,c]*AA[b,d]+R[a,c,b,d])*2/3

                    Q4[a,b,c,d]+= ACA[a,d]*ACA[c,b]/3
                    Q4[a,b,c,d]+= (AA[a,d]*ACA[c,b]-T[a,d,c,b])*2/3
                    Q4[a,b,c,d]+= (AA[c,b]*ACA[a,d]-T[c,b,a,d])*2/3
                    Q4[a,b,c,d]+= (-1*AA[a,d]*AA[c,b]+R[a,d,c,b])*2/3

    return Q4


def reduce4(T4):

    N = T4.shape[0]

    S = 0
    for a in range(N):
        for c in range(N):
            S += T4[a,a,c,c]

    return S


def tensor4_h(C2):

    T4 = np.zeros((4,4,4,4))

    for j in range(4):
        for k in range(4):
            for l in range(4):
                for m in range(4):

                    T4[j,k,l,m] = (C2[j,k]*C2[l,m]+C2[j,l]*C2[k,m]+C2[j,m]*C2[k,l])/3
                    T4[j,k,l,m] += (kron2(j,k)+kron2(l,m)-kron2(j,k)*kron2(l,m))*(C2[j,k]*C2[l,m]-C2[k,m]*C2[j,l])*2/3
                    T4[j,k,l,m] += (kron2(j,l)+kron2(k,m)-kron2(j,l)*kron2(k,m))*(C2[j,l]*C2[k,m]-C2[j,m]*C2[k,l])*2/3
                    T4[j,k,l,m] += (kron2(j,m)+kron2(k,l)-kron2(j,m)*kron2(k,l))*(C2[j,m]*C2[k,l]-C2[j,k]*C2[l,m])*2/3

    return T4



def find_b(A,Cp):

    N, M = A.shape

    ATA = A.T.dot(A)
    dATA = np.diag(np.diag(ATA))

    Y1 = (ATA.dot(Cp)+Cp.dot(ATA))/2
    Y2 = -1*(dATA.dot(Cp)+Cp.dot(dATA))/2

    def obj(vb):

        b = vb.reshape(M,M)
        b = (b-b.T)/2

        X1 = ATA.dot(b) - b.dot(ATA)
        X2 = dATA.dot(b) - b.dot(dATA)

        dist = np.sum(np.abs(X1-Y1)**2)
        dist += np.sum(np.abs(X2-Y2)**2)

        return dist

    res = opt.minimize(obj,x0=np.zeros((M,M)).flatten())
    b = res.x.reshape(M,M)
    b = (b - b.T)/2

    return b

X = .1*np.random.rand(4,4)
X += X.T
X[:-1,:-1] = 0
X[-1,-1] = 0

#X = np.zeros((4,4))

A = np.random.rand(4,4)

C2, C4 = corr_h(X)

T = tensor_T(C2,A)
R = tensor_R(C2,A)

Q4 = tensor4_q(C2,A)

Q2 = 0
for k in range(4):
    Q2 += Q4[:,:,k,k]
