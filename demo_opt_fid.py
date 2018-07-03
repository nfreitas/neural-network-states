import numpy as np

import nqs
import models
import sampler
import optimize


ubm = nqs.wrbbm_nqs(nv=10,nh=10)
ubm.setRandomParams()
#ubm.X = np.zeros_like(ubm.X)

rbm = ubm.asRBM()
S = sampler.metropolis(wf=rbm)

O = optimize.fidelity_GA(S,ubm)
