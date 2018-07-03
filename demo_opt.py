import numpy as np

import nqs
import models
import sampler
import optimize


#wf = nqs.ti_brbm_nqs(nv=10,nf=1)
wf = nqs.brbm_nqs(nv=10,nh=10)
wf.setRandomParams()

TI = models.TransIsing1D(nspins=10,hfield=.1)

S = sampler.metropolis(wf=wf)

O = optimize.energy_GD(S,TI)
