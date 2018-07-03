import numpy as np

class nqo:
    """Neural Quantum Operation"""

    def applyTo(self, nqs, ind):
        return

class one_body(nqo):
    """A general one body Neural Quantum Operation"""

    def __init__(self, alpha, beta, omega):

        self.alpha = alpha
        self.beta = beta
        self.omega = omega

    def applyTo(self, nqs, ind):

        if self.omega == np.Inf:
            nqs.a[ind] -= self.alpha + self.beta
        else:
            nqs.addHiddenNode()

            nqs.X[0:-1,-1] = nqs.W[0:-1,ind]
            nqs.X[-1,0:-1] = nqs.W[0:-1,ind]

            nqs.W[0:-1,ind] = 0
            nqs.W[-1,:] = nqs.Y[ind,:]
            nqs.W[-1,ind] = self.omega

            nqs.Y[ind,:] = 0
            nqs.Y[:,ind] = 0

            nqs.b[-1] = self.beta + nqs.a[ind]
            nqs.a[ind] = self.alpha

class ID(nqo):
    """Identity"""

    def applyTo(self, nqs, ind):

        return nqs

class RZ(one_body):
    """A rotation around axis z of a single qubit: U = e^{-i angle/2 * sigma^z}"""

    def __init__(self, angle):

        self.angle = angle

    def applyTo(self, nqs, ind):

        nqs.a[ind] += -1j*self.angle/2


class RX(one_body):
    """A rotation around axis x of a single qubit: U = e^{-i angle/2 * sigma^x}"""

    @staticmethod
    def __real_angle__(angle):

        angle = angle%(2*np.pi)
        if angle > np.pi:
            angle = angle - 2*np.pi

        if angle ==  0:
            op = ID()
        else:
            if angle>0:
                a = b = 0
            elif angle<0:
                a = np.pi/2
                b = -a

            w = -1*np.log(np.tan(np.abs(angle)/2))/2

            op = one_body(1j*a,1j*b,w+1j*np.pi/4)

        return op

    @staticmethod
    def __imag_angle__(angle):

        if angle ==  0:
            op = ID()
        else:
            if angle>0:
                a = b = 0
            elif angle<0:
                a = np.pi/2
                b = -a

            w = -1*np.log(np.tanh(np.abs(angle)/2))/2

            op = one_body(1j*a,1j*b,w)

        return op

    def __new__(cls, angle):

        if angle.imag == 0:
            op = cls.__real_angle__(angle.real)
        elif angle.real == 0:
            op = cls.__imag_angle__(angle.imag)
        else:
            raise Exception("Complex valued angles not yet implemented")

        return op


class inf_RX():
    """An infinitesimal rotation around axis x of a single qubit:
    U = e^{-i angle/2 * sigma^x}"""

    def __init__(self, angle):

        self.angle = angle

    def applyTo(self, nqs, ind):

        A = -1j*self.angle/2

        C = 1
        for j in range(nqs.nh):
            C *= np.cosh(2*nqs.W[j,ind])

        # update visible offsets
        da = 2*A*C*nqs.a[ind]*nqs.Y[ind,:].reshape(nqs.nv,1)
        da[ind] -= 2*A*C*nqs.a[ind]

        nqs.a += da

        # update hidden offsets
        db = 2*A*C*nqs.a[ind]*np.tanh(2*nqs.W[:,ind]).reshape(nqs.nh,1)

        nqs.b += db

        # update visible links
        dY = 0*4*A*C*np.outer(nqs.Y[ind,:], nqs.Y[:,ind])
        #dY -= 4*A*C*np.outer(nqs.Y[ind,:]**2, nqs.Y[:,ind]**2)
        dY[ind,:] -= 2*A*C*nqs.Y[ind,:]
        dY[:,ind] -= 2*A*C*nqs.Y[:,ind]

        nqs.Y += dY

        # update filters

        dW = 2*A*C*np.outer(np.tanh(2*nqs.W[:,ind]),nqs.Y[ind,:])
        dW[:,ind] -= 2*A*C*np.tanh(2*nqs.W[:,ind])*(1/2+0*nqs.a[ind]**2+0*np.sum(nqs.Y[ind,:])**2)

        nqs.W += dW

class CRZ(nqo):
    """A controlled rotation around axis z: U = e^{-i angle/2 * sigma^z_1 * sigma^z_2}"""

    def __init__(self, angle):

        self.angle = angle

    def applyTo(self, nqs, ind):

        nqs.Y[ind[0],ind[1]] -= 1j*self.angle/2
        nqs.Y[ind[1],ind[0]] -= 1j*self.angle/2


class hidden_CRZ(nqo):
    """A controlled rotation around axis z: U = e^{-i angle/2 * sigma^z_1 * sigma^z_2}"""

    def __init__(self, angle):

        self.angle = angle

    def applyTo(self, nqs, ind):

        y = -1j*self.angle/2
        w = np.arccosh(np.exp(2*y))/2

        nqs.addHiddenNode()
        nqs.W[-1,ind[0]] = w
        nqs.W[-1,ind[1]] = w
