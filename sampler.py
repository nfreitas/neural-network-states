import numpy as np

class metropolis:

    def __init__(self, wf):

        self.wf = wf
        self.st = None  # current state
        self.resetRunVar()

        # function to call after each sweep
        self.afterSweep = None

    def resetRunVar(self):

        self.moves = 0
        self.accept = 0
        self.sweeps = 0
#        self.energy = []

    def initRandomState(self):

        self.st = -1 + 2*np.random.randint(0,2,self.wf.nv)
        self.st = np.array(self.st).reshape(self.wf.nv,1)

    def selRandSpins(self):

        return np.random.randint(0,self.wf.nv,1)

    def singleStep(self):

        iflips = self.selRandSpins()

        acceptance = np.abs(self.wf.PoP(self.st, iflips))**2

        if np.random.rand() < acceptance:

            self.wf.updateLT(self.st,iflips)
            self.st[iflips] *= -1
            self.accept += 1

        self.moves += 1

    def run(self, nsweeps, thermfactor=.1, sweepfactor=1, init_st=None, output=False):

        if output:
            print("# Starting Montecarlo sampling")
            print("# Number of sweeps to be performed is ", nsweeps)
            print("# Thermalization... ", end='')

        if type(init_st) == type(None):
            self.initRandomState()
        else:
            self.st = init_st

        self.wf.initLT(self.st)
        for j in range(int(nsweeps*thermfactor)):
            for k in range(self.wf.nv*sweepfactor):
                self.singleStep()

        if output:
            print("DONE")
            print("# Sweeping... ", end='')

        self.resetRunVar()
        for j in range(nsweeps):
            for k in range(self.wf.nv*sweepfactor):
                self.singleStep()
            self.sweeps += 1
            if self.afterSweep:
                self.afterSweep(self)

        if output:
            print("DONE")
