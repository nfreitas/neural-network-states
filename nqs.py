import utils
import numpy as np
import scipy as sp

import scipy.optimize as opt


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


def logcosh(x):
    return np.log(2*np.cosh(x))

class nqs:
    """Neural Quantum State"""

    def logPsi(self, st):
        """Evaluates the logarithm of Psi(S). S must be a column vector
        of size nv specifying an state of the computational basis"""

    def logPoP(self, st, iflips):
        """Returns the logarithm of Psi(st')/Psi(st)
        where st' is a state with a certain number of flipped spins
        the vector 'iflips' contains the indexes to be flipped"""

    def PoP(self, st, iflips):
        """Returns of Psi(st')/Psi(st) where st' is a state with a certain
        number of flipped spins the vector 'iflips' contains the indexes
        to be flipped"""

        lPoP = self.logPoP(st,iflips)
        return np.exp(lPoP)

    def initLT(self, st):
        """Initializes the lookup table used in logPoP. S must be a column
        vector of size nv specifying an state of the computational basis"""

    def updateLT(self, st, iflips):
        """Updates the lookup table used in logPoP. S must be a column
        vector of size nv specifying an state of the computational basis, and
        iflips must be a list of indexes where the spins are flipped"""

    def params(self):
        """return the parameters of the wave function"""

    def setParams(self, params):
        """set the parameters of the wave function"""

    def setRandomParams(self, scale=1):
        """set random parameters for the wave function"""

    def rmHiddenNodes(self, keep):

        if self.nh <= keep:
            return

        self.b = self.b[-1*keep:,:]
        self.X = self.X[-1*keep:,-1*keep:]
        self.W = self.W[-1*keep:,:]

        self.nh = keep


class brbm_nqs(nqs):
    """Neural Quantum States based on Binomial Restricted Boltzmann Machines"""

    @classmethod
    def load_file(cls, fname):
        """it load the nqs from a file in the format used by the code given in
        the supp. material of the Carleo and Troyer Science paper"""

        with open(fname) as fd:

            nv = int(fd.readline())
            nh = int(fd.readline())

            ins = cls(nv,nh)

            for k in range(nv):
                r,i = fd.readline()[1:-2].split(',')
                ins.a[k] = float(r)+1j*float(i)

            for k in range(nh):
                r,i = fd.readline()[1:-2].split(',')
                ins.b[k] = float(r)+1j*float(i)

            for j in range(nv):
                for k in range(nh):
                    r,i = fd.readline()[1:-2].split(',')
                    ins.W[k,j] = float(r)+1j*float(i)

            return ins

    def __init__(self, nv, nh):
        """nh/nv are the number of visible and hidden units, respectively"""

        self.nv = nv
        self.nh = nh
        self.a = np.zeros((nv,1), dtype=np.complex128)
        self.b = np.zeros((nh,1), dtype=np.complex128)
        self.W = np.zeros((nh,nv), dtype=np.complex128)
        # visible interactions
        self.Y = np.zeros((nv,nv), dtype=np.complex128)

        self.resetMem()

    def resetMem(self):

        self.logPsi_dict = {}

    def log_norm(self):

        max_log = np.max(np.real(list(self.logPsi_dict.values())))

        S = 0
        for k in self.logPsi_dict.keys():
            S += np.exp(2*(np.real(self.logPsi_dict[k])-max_log))

        return max_log + np.log(S)/2

    def logPsi(self, st):

        n = utils.regToInt(st)

        if n in self.logPsi_dict.keys():
            logpsi = self.logPsi_dict[n]
        else:
            sa = self.a.T.dot(st)
            # visible interactions
            sa += st.T.dot(self.Y.dot(st))/2

            lf =  logcosh(self.b+self.W.dot(st))
            logpsi = sa + np.sum(lf)
            self.logPsi_dict[n] = logpsi

        return logpsi

    def logPoP(self, st, iflips):

        logpop = -2*self.a[iflips].T.dot(st[iflips])
        logpop += -2*st[iflips].T.dot(self.Y[iflips,:].dot(st))
        dtheta = -2*self.W[:,iflips].dot(st[iflips])
        logpop += np.sum(logcosh(self.LT+dtheta)-logcosh(self.LT))

        return logpop

    def initLT(self, st):

        self.LT =  self.b+self.W.dot(st)

    def updateLT(self, st, iflips):

        self.LT -= 2*self.W[:,iflips].dot(st[iflips])

    def params(self):

        return {'a': self.a, 'b':self.b, 'W':self.W, 'Y':self.Y}

    def setParams(self, params):

        for par in params.keys():
            exec('self.' + par + ' = params[par].copy()')

        self.resetMem()

    def setRandomParams(self, scale=1):

        self.a = scale*(-1+2*np.random.rand(self.nv,1).astype(np.complex128))
        self.b = scale*(-1+2*np.random.rand(self.nh,1).astype(np.complex128))
        self.W = scale*(-1+2*np.random.rand(self.nh,self.nv).astype(np.complex128))
        self.Y = scale*(-1+2*np.random.rand(self.nv,self.nv).astype(np.complex128))

        self.resetMem()

    def varGradient(self, st):
        """return the variational derivatives with respect to the
        parameters of the wave function, of the coefficient corresponding to
        state |st>"""

        theta = self.b + self.W.dot(st)

        va = st
        vY = st.dot(st.T)/2
        vb = np.tanh(theta)
        vW = vb.dot(st.T)

        return {'a': va, 'b':vb, 'W':vW, 'Y':vY}
#        return {'a': va, 'b':vb, 'W':vW}

    def asUBM(self):

        ubm = bbm_nqs(self.nv,self.nh)
        ubm.setParams({'a':self.a, 'b':self.b, 'W':self.W, 'Y':self.Y})

        return ubm

    def rmVisibleInt(self):
        """Removes visible interactions and replaces them with new hidden
        nodes"""

        for j in range(self.nv):
            for k in range(j,self.nv):

                if np.abs(self.Y[j,k]) == 0:
                    self.Y[j,k] = 0
                    self.Y[k,j] = 0
                    continue

                w = np.arccosh(np.exp(2*np.abs(self.Y[j,k])))/2
                self.addHiddenNode()
                self.W[-1,j] = w
                self.W[-1,k] = w*np.sign(self.Y[j,k])

                self.Y[j,k] = 0
                self.Y[k,j] = 0

    def addHiddenNode(self):

        self.W = np.r_[self.W, np.zeros((1,self.nv))]
        self.b = np.r_[self.b, np.zeros((1,1))]

        self.nh += 1

    def rmHiddenNodes(self, keep):

        if self.nh <= keep:
            return

        self.b = self.b[-1*keep:,:]
        self.W = self.W[-1*keep:,:]

        self.nh = keep

class ti_brbm_nqs(brbm_nqs):
    """ Translation Invariant Neural Quantum States based on Binomial Restricted Boltzmann Machines"""

    def __init__(self, nv, nf):
        """nh is the number of visible units, and nf in the number of features"""

        nh = nv*nf
        super().__init__(nv, nh)

        self.nf = nf
        self.bf = np.zeros((nf,1))
        self.Wf = np.zeros((nf,nv))

        self.ones = np.ones((nv,1))

    def __expandParams__(self):
        """is takes the vector bf and matrix Wf with the feature's parameters and
        form the vector b and matrix W with the hidden node's parameters """

        for k in range(self.nf):
            self.b[k*self.nv:(k+1)*self.nv] = self.bf[k]*self.ones

        for k in range(self.nf):
            for n in range(self.nv):
                self.W[k*self.nv+n,:] = np.roll(self.Wf[k,:],n)

    def params(self):

        return {'bf': self.bf, 'Wf':self.Wf}

    def setParams(self, params):

        for par in params.keys():
            exec('self.' + par + ' = params[par]')

        self.__expandParams__()

    def setRandomParams(self, scale=1e-2):

        self.bf = scale*(-1+2*np.random.rand(self.nf,1).astype(np.complex128))
        self.Wf = scale*(-1+2*np.random.rand(self.nf,self.nv).astype(np.complex128))
        self.__expandParams__()

    def varGradient(self, st):

        theta = self.b+self.W.dot(st)
        tanh = np.tanh(theta)

        vbf = np.zeros_like(self.bf)
        vWf = np.zeros_like(self.Wf)

        for k in range(self.nf):
            vbf[k] = np.sum(tanh[k*self.nv:(k+1)*self.nv])

            for n in range(self.nv):
                sst = np.roll(st, -1*n)
                vWf[k,n] = tanh[k*self.nv:(k+1)*self.nv,:].T.dot(sst)

        return {'bf':vbf, 'Wf':vWf}

class bbm_nqs(nqs):
    """Neural Quantum States based on Binomial Boltzmann Machines"""

    def __init__(self, nv, nh):
        """nh/nv are the number of visible and hidden units, respectively"""

        self.nv = nv
        self.nh = nh
        self.a = np.zeros((nv,1), dtype=np.complex128)
        self.b = np.zeros((nh,1), dtype=np.complex128)
        self.W = np.zeros((nh,nv), dtype=np.complex128)
        self.X = np.zeros((nh,nh), dtype=np.complex128)
        self.Y = np.zeros((nv,nv), dtype=np.complex128)

        self.logPsi_dict = {}

    def resetMem(self):
        self.logPsi_dict = {}

    def log_norm(self):

        max_log = np.max(np.real(list(self.logPsi_dict.values())))

        S = 0
        for k in self.logPsi_dict.keys():
            S += np.exp(2*(np.real(self.logPsi_dict[k])-max_log))

        return max_log + np.log(S)/2

    def logPsi(self, st):

        nst = utils.regToInt(st)

        if nst in self.logPsi_dict.keys():
            logpsi = self.logPsi_dict[nst]
        else:
            p = self.a.T.dot(st) + st.T.dot(self.Y).dot(st)/2

            if self.nh == 0:
                f = 1
            else:
                f = 0
                Wst = self.W.dot(st)
                for n in range(2**self.nh):
                    h = utils.intToReg(n, self.nh)
                    exp = self.b.T.dot(h) + h.T.dot(Wst) + h.T.dot(self.X).dot(h)/2
                    f += np.exp(exp)

            logpsi = p + np.log(f)

            self.logPsi_dict[nst] = logpsi

        return logpsi


    def addHiddenNode(self):

        self.W = np.r_[self.W, np.zeros((1,self.nv))]

        self.X = np.r_[self.X, np.zeros((1,self.nh))]
        self.X = np.c_[self.X, np.zeros((self.nh+1,1))]

        self.b = np.r_[self.b, np.zeros((1,1))]

        self.nh += 1

    def params(self):

        return {'a': self.a, 'b':self.b, 'W':self.W, 'X':self.X, 'Y':self.Y}

    def setParams(self, params):

        for par in params.keys():
            exec('self.' + par + ' = params[par].copy()')

        self.resetMem()

    def setRandomParams(self, scale=1):

        self.a = scale*(-1+2*np.random.rand(self.nv,1).astype(np.complex128))
        self.b = scale*(-1+2*np.random.rand(self.nh,1).astype(np.complex128))
        self.W = scale*(-1+2*np.random.rand(self.nh,self.nv).astype(np.complex128))
        self.X = scale*(-1+2*np.random.rand(self.nh,self.nh).astype(np.complex128))
        self.Y = scale*(-1+2*np.random.rand(self.nv,self.nv).astype(np.complex128))

#        self.a += 1j*scale*(-1+2*np.random.rand(self.nv,1).astype(np.complex128))
#        self.b += 1j*scale*(-1+2*np.random.rand(self.nh,1).astype(np.complex128))
#        self.W += 1j*scale*(-1+2*np.random.rand(self.nh,self.nv).astype(np.complex128))
#        self.X += 1j*scale*(-1+2*np.random.rand(self.nh,self.nh).astype(np.complex128))
#        self.Y += 1j*scale*(-1+2*np.random.rand(self.nv,self.nv).astype(np.complex128))

        self.X = self.X + self.X.T
        self.Y = self.Y + self.Y.T

        np.fill_diagonal(self.X, 0)
        np.fill_diagonal(self.Y, 0)

        self.resetMem()

    def adjacencyMatrix(self):

        a1 = np.c_[self.X,self.W]
        a2 = np.c_[self.W.T,self.Y]
        A = np.r_[a1,a2]

        return A

    def asRBM(self):

        rbm = brbm_nqs(self.nv,self.nh)
        rbm.setParams({'a':self.a, 'b':self.b, 'W':self.W, 'Y':self.Y})

        return rbm

    def renormalize1(self):

        for k in range(self.nh-1):
            if self.X[k,-1] == 0:
                continue

            if np.sum(np.abs(self.W[k,:])) == 0:
                continue

            a2, beta, bp = utils.approx1(self.W[k,:], self.b[k], self.X[k,-1])
            #print('approx1', k, beta, bp)
            self.b[k] = bp
            self.W[k,:] = beta*self.W[k,:]

        for k in range(self.nh-1):

            if self.X[k,-1] == 0:
                continue

            if np.sum(np.abs(self.W[k,:])) == 0:
                self.X[k,-1] = 0
                self.X[-1,k] = 0
                continue

            alpha, beta = utils.approx2(self.W[k,:], self.b[k], self.X[k,-1])

            #print('approx2', k, alpha, beta)
            #print(self.W[-1,:])
            self.W[-1,:] += alpha*self.W[k,:]/2
            #print(self.W[-1,:])
            self.b[-1] += beta/2

            self.X[k,-1] = 0
            self.X[-1,k] = 0

    def renormalize2(self):

        for k in range(self.nh-1):
            if self.X[k,-1] == 0:
                continue

            self.W[-1,:] += 1*self.X[k,-1]*self.W[k,:]

            self.X[k,-1] = 0

        return

    def renormalize3(self):

        for k in range(self.nh-1):
            if self.X[k,-1] == 0:
                continue

            self.W[-1,:] += 1*np.tanh(self.X[k,-1])*self.W[k,:]

            self.W[k,:] = 0
            self.X[k,-1] = 0

        return

#    def renormalize3(self, nu=.735):
#
#        C = np.diag([1+0j]*self.nh)
#
#        for j in range(self.nh-1):
#            C[j,-1] = np.tanh(self.X[j,-1])
#            C[-1,j] = C[j,-1]
#            for k in range(j+1,self.nh-1):
#                C[j,k] = np.tanh(self.X[j,-1])*np.tanh(self.X[k,-1])
#                C[k,j] = C[j,k]
#
#        S, V = np.linalg.eigh(C)
#        S = np.sqrt(np.abs(S))
#
#        nW = V.dot(np.diag(S**(nu)).dot(V.T.conj().dot(self.W)))
##        nW = V.dot(np.diag(S-1)).dot(V.T.dot(self.W))
#
# #       print(np.max(np.abs(self.W-nW))/np.max(np.abs(self.W)))
#        self.W = nW
##        self.W += nu*nW
##       self.b = nb
#        self.X = np.zeros_like(self.X)
#
#        return C

    def rmHiddenNodes(self, keep):

        if self.nh <= keep:
            return

        self.b = self.b[-1*keep:,:]
        self.X = self.X[-1*keep:,-1*keep:]
        self.W = self.W[-1*keep:,:]

        self.nh = keep

class wrbbm_nqs(bbm_nqs):
    """Neural Quantum States based on Weakly Restricted Binomial Boltzmann Machines"""

    def logPsi(self, st):

        nst = utils.regToInt(st)

        if nst in self.logPsi_dict.keys():
            logpsi = self.logPsi_dict[nst]
        else:
            p = self.a.T.dot(st) + st.T.dot(self.Y).dot(st)/2

            Wst = self.W.dot(st)
            x = self.X[:,-1].reshape((self.nh,1))
            l1 = np.sum(np.log(2*np.cosh((Wst + self.b + x)[:-1])))
            l2 = np.sum(np.log(2*np.cosh((Wst + self.b - x)[:-1])))

            l1 += self.b[-1] + Wst[-1]
            l2 -= self.b[-1] + Wst[-1]

            if np.real(l1) > np.real(l2):
                L = l1
                l = l2-l1
            else:
                L = l2
                l = l1 - l2

            logpsi = p + L + np.log(1+np.exp(l))

            self.logPsi_dict[nst] = logpsi

        return logpsi

    def logPoP(self, st, iflips):

        nst = st.copy()
        nst[iflips] *= -1

        return self.logPsi(nst) - self.logPsi(st)
