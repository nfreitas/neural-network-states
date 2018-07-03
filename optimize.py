import numpy as np
import utils

class energy_GD:

    def __init__(self, sampler, model):

        self.sampler = sampler
        self.model = model
        self.energy = []

    @staticmethod
    def __measure__(M,model,sampler):

        # measure energy
        E = model.localEnergy(sampler.wf,sampler.st)
        # update energy mean value
        M['E'] += (E-M['E'])/sampler.sweeps

        # measure variatonal derivatives
        O = sampler.wf.varGradient(sampler.st)
        # udpdate energy gradient
        for par in O.keys():
            EO = np.real(O[par]*E)
            M['EO'][par] += (EO-M['EO'][par])/sampler.sweeps
            M['O'][par] += (np.conj(O[par])-M['O'][par])/sampler.sweeps

    def opt(self, nsteps, nsweeps, learning_rate, decay_rate=.1):

        M = {'E':0, 'EO':{}, 'O':{}}
        self.sampler.afterSweep = lambda S: self.__measure__(M, self.model, S)

        for step in range(nsteps):

            # initialize energy and variatonal derivatives
            M['E'] = 0
            M['EO'] = {}.fromkeys(self.sampler.wf.params().keys(),0)
            M['O'] = {}.fromkeys(self.sampler.wf.params().keys(),0)

            # sample wave function
            self.sampler.run(nsweeps, thermfactor=0.1)

            # save energy for each step
            self.energy.append(M['E'])
            print(M['E'])

            # change wave function parameters
            params = self.sampler.wf.params()
#            print('\n----------------\nE',M['E'])
#            print('a',params['a'])
#            print('bf',params['bf'])
#            print('Wf',params['Wf'])
#            print(self.sampler.wf.LT)
            for par in params.keys():
                eps = learning_rate*(decay_rate**(step/nsteps))
                params[par] = (eps)*(M['EO'][par]-M['O'][par]*M['E'])
            self.sampler.wf.setParams(params)

            # corregir expresion del gradiente y poner decaimiento exponencial!

class fidelity_GA:

    def __init__(self, sampler, target):

        self.sampler = sampler
        self.target = target
        self.fidelity = []

    @staticmethod
    def __measure__(M, target, sampler):

        # measure D
#        l1 = np.real(target.logPsi(sampler.st)) # I don't care about the phase
        l1 = target.logPsi(sampler.st)
        l2 = sampler.wf.logPsi(sampler.st)
#        print(l1,l2)
        D = np.exp(l1-l2)

        # update D mean value
        M['D'] += (D-M['D'])/sampler.sweeps

        # measure variatonal derivatives
        O = sampler.wf.varGradient(sampler.st)

        # udpdate gradient
        for par in O.keys():
            DO = D*np.conj(O[par])
            M['DO'][par] += (DO-M['DO'][par])/sampler.sweeps
            M['O'][par] += (np.conj(O[par])-M['O'][par])/sampler.sweeps

    def opt(self, nsteps, nsweeps, learning_rate=1e-2, decay_rate=.1):

        M = {'D':0, 'DO':{}, 'O':{}}
        self.sampler.afterSweep = lambda S: self.__measure__(M, self.target, S)

        for step in range(nsteps):

            # initialize D and variatonal derivatives
            M['D'] = 0
            M['DO'] = {}.fromkeys(self.sampler.wf.params().keys(),0)
            M['O'] = {}.fromkeys(self.sampler.wf.params().keys(),0)

            # sample wave function
            self.sampler.run(nsweeps, thermfactor=0.1)

            for k in self.sampler.wf.logPsi_dict.keys():
                self.target.logPsi(utils.intToReg(k,self.target.nv))

#            print(self.sampler.wf.logPsi_dict)
#            print(self.target.logPsi_dict)

            # save fidelity for each step
#            f = np.abs(M['D'])*self.sampler.wf.norm()/self.target.norm()
            f = np.abs(M['D'])*np.exp(self.sampler.wf.log_norm()-self.target.log_norm())
            self.fidelity.append(f**2)
            print('opt', step, f**2)
            print(M['D'])
            print(self.sampler.wf.log_norm(), self.target.log_norm())
#            if self.fidelity[-1] > good_fid and step > .1*nsteps:
#                break

            # change wave function parameters
            params = self.sampler.wf.params()

            for par in params.keys():
                eps = learning_rate*(decay_rate**(step/nsteps))
                params[par] += (eps)*(M['DO'][par]-M['O'][par]*M['D'])/M['D']

            self.sampler.wf.setParams(params)
            self.target.resetMem()

class det_fidelity_GA:

    def __init__(self, wf, target):

        self.wf = wf
        self.target = target
        self.fidelity = []

        self.target_list = []
        self.st_list = []
        self.target.resetMem()
        for k in range(2**self.target.nv):
            st = utils.intToReg(k, self.target.nv)
            self.target_list.append(self.target.logPsi(st))
            self.st_list.append(st)
        self.target_log_norm = self.target.log_norm()

    def f_and_grad(self):

        D = 0
        par = self.wf.params().keys()
        G = {'O':{}.fromkeys(par,0), 'DO':{}.fromkeys(par,0)}

        K = 2**self.wf.nv

        for k in range(K):
            st = self.st_list[k]
            l2 = self.wf.logPsi(st)

        wf_log_norm = self.wf.log_norm()

        for k in range(K):
            st = self.st_list[k]

            # measure D
            #l1 = np.real(self.target_list[k]) # I don't care about the phase
            l1 = self.target_list[k] - self.target_log_norm
            l2 = self.wf.logPsi(st) - wf_log_norm
            d = np.exp(l1-l2)
            weight = np.exp(2*np.real(l2))

            # update D mean value
            D += d*weight

            # measure variatonal derivatives
            O = self.wf.varGradient(st)

            # udpdate gradient
            for par in O.keys():
                o = np.conj(O[par])
                G['DO'][par] += (d*o)*weight
                G['O'][par] += o*weight

        return D, G

    def opt(self, nsteps, learning_rate, decay_rate=.1):

        for step in range(nsteps):

            D, G = self.f_and_grad()

            fid = np.abs(D)#*np.exp(self.wf.log_norm()-self.target.log_norm())
            self.fidelity.append(fid**2)
            print('opt', step, fid**2)

            # change wave function parameters
            params = self.wf.params()
#            print(params['W'])
#            print(G['O']['W'])
            for par in params.keys():
                eps = learning_rate*(decay_rate**(step/nsteps))
                params[par] += (eps)*(G['DO'][par]-G['O'][par]*D)/D

            self.wf.setParams(params)


class det_rel_entropy_GD:

    def __init__(self, wf, target):

        self.wf = wf
        self.target = target
        self.rel_entropy = []

        self.target_list = []
        self.st_list = []
        self.target.resetMem()
        for k in range(2**self.target.nv):
            st = utils.intToReg(k, self.target.nv)
            self.target_list.append(self.target.logPsi(st))
            self.st_list.append(st)
        self.target_log_norm = self.target.log_norm()

    def f_and_grad(self):

        D = 0
        par = self.wf.params().keys()
        G = {}.fromkeys(par,0)

        K = 2**self.wf.nv

        for k in range(K):
            st = self.st_list[k]
            l2 = self.wf.logPsi(st)

        wf_log_norm = self.wf.log_norm()

        for k in range(K):
            st = self.st_list[k]

            # measure D
#            l1 = np.real(self.target_list[k] - self.target_log_norm)
            l1 = self.target_list[k] - self.target_log_norm
            l2 = self.wf.logPsi(st) - wf_log_norm
            print(l1,l2)
            weight1 = np.exp(2*np.real(l1))
            weight2 = np.exp(2*np.real(l2))

            D += 2*(l1-l2)*weight1

            # measure variatonal derivatives
            O = self.wf.varGradient(st)

            # udpdate gradient
            for par in O.keys():
                G[par] += 2*(weight2*np.real(O[par])-weight1*O[par])

        return D, G

    def opt(self, nsteps, learning_rate, decay_rate=.2):

        for step in range(nsteps):

            D, G = self.f_and_grad()

            rel_ent = np.abs(D)
            self.rel_entropy.append(rel_ent)
            print('opt', step, rel_ent)

            # change wave function parameters
            params = self.wf.params()

            for par in params.keys():
                eps = learning_rate*(decay_rate**(step/nsteps))
                params[par] -= (eps)*G[par]

            self.wf.setParams(params)

def opt_phases(logPsi_dict, N, nsteps, learning_rate, decay_rate=.1):

    L = np.real(list(logPsi_dict.values()))
    max_log = np.max(L)
    log_norm = max_log + np.log(np.sum(np.exp(2*(L-max_log))))/2

    phase = {}
    weight = {}

    for k in logPsi_dict.keys():
        phase[k] = np.imag(logPsi_dict[k])#%(2*np.pi)
        weight[k] = np.exp(2*(np.real(logPsi_dict[k])-log_norm))

#    return phase, weight

    def func(a, Y):
        f = 0
        for k in phase.keys():
            st = utils.intToReg(k, N)
            ph = a.T.dot(st) + st.T.dot(Y.dot(st))/2
            f += weight[k]*(ph-phase[k])**2
        return f

    def grad(a, Y):
        ga = 0
        gY = 0
        for k in phase.keys():
            st = utils.intToReg(k, N)
            ph = a.T.dot(st) + st.T.dot(Y.dot(st))/2
            ga += 2*weight[k]*(ph-phase[k])*st
            gY += 1*weight[k]*(ph-phase[k])*(st.dot(st.T))
        return ga, gY

    a = np.random.rand(N,1)
    Y = np.random.rand(N,N)
#    Y = Y - np.diag(np.diag(Y))

    for step in range(nsteps):
        ga, gY = grad(a,Y)
        eps = learning_rate*(decay_rate**(step/nsteps))
        a -= eps*ga
        Y -= eps*gY
#        Y -= np.diag(np.diag(Y))
        print(func(a,Y))

    return a, Y
