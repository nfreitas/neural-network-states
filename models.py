import nqo
import utils
import numpy as np
import scipy as sp

class TransIsing1D():

    def __init__(self, nspins, hfield, pbc=True):

        self.nspins = nspins
        self.hfield = np.complex128(hfield)
        self.pbc = pbc

        self.iflips = [[]] + [[k] for k in range(self.nspins)]
        self.mel = [np.complex128(0)] + [-1*self.hfield] * self.nspins

    def findConn(self, st):
        """for the state |st>, this function returns all the |st'> and the matrix
        elements <st'|H|st> such that <st'|H|st> !=0. Each |st'> is encoded as
        a series of spin flips to be performed on |st>"""

        en = np.complex128(0)
        for k in range(self.nspins-1):
            en -= st[k]*st[k+1]
        if self.pbc:
            en -= st[0]*st[-1]

        self.mel[0] = en[0]

        return self.iflips, self.mel

    def localEnergy(self, wf, st):
        """it returns the local energy of a given state of the
        computational basis |st> according to the wave function wf"""

        iflips, mel = self.findConn(st)

        en = 0
        for k in range(len(mel)):
            pop = wf.PoP(st,iflips[k])
            if pop == None:
                continue
            en += pop*mel[k].conj()

        return en

    def buildFullH(self):

        H = np.zeros((2**self.nspins, 2**self.nspins),dtype=complex)

        for k in range(2**self.nspins):
            st = utils.intToReg(k, self.nspins)

            I, M = self.findConn(st)

            for iflip, mel in zip(I,M):
                nst = st.copy()
                nst[iflip] *= -1
                j = utils.regToInt(nst)

                H[j,k] = mel

        return H


    def buildSparseH(self):

        H = sp.sparse.lil_matrix((2**self.nspins, 2**self.nspins),dtype=complex)

        for k in range(2**self.nspins):
#            print(k)
            st = utils.intToReg(k, self.nspins)

            I, M = self.findConn(st)

            for iflip, mel in zip(I,M):
#                nst = st.copy()
#                nst[iflip] *= -1
#                j = utils.regToInt(nst)
                j = k
                for index in iflip:
                    j = j^(1<<index)

                H[j,k] = mel

        return H

    def applyTrotterStep(self,nqs,tao,renorm=False,inf=False,hidden_crz=False):

        if inf:
            g1 = nqo.inf_RX(-1*2*tao*self.hfield)
        else:
            g1 = nqo.RX(-1*2*tao*self.hfield)

        if hidden_crz:
            g2 = nqo.hidden_CRZ(-1*2*tao)
        else:
            g2 = nqo.CRZ(-1*2*tao)

        for k in range(self.nspins-1):
            g2.applyTo(nqs, (k,k+1))
        if self.pbc:
            g2.applyTo(nqs, (0,-1))

        for k in range(self.nspins):
            g1.applyTo(nqs, k)
            if renorm:
                nqs.renormalize1()

        return g1, g2

    def runImagTrotterEvol(self, nqs, nsteps, learning_rate, decay_rate=.2):

        for step in range(nsteps):
            eps = learning_rate*(decay_rate**(step/nsteps))
            self.applyTrotterStep(nqs, -1j*eps,renorm=True)
