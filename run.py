import numpy as np
from model import *
T = 200
r = .99
σ = 4
u = 4
n = 20
γ=.34
π=20
m = .05
f2 = .05
ω = .1
f = 0
β=10
state_space = np.round(np.linspace(-50,50,1001),1) 

def norm_distrib(x,loc,scale=np.sqrt(1-r**2)*σ):
    x = np.array(x)
    loc = np.array(loc)
    x = np.tile(x,(loc.size,1))
    loc = np.tile(loc,(loc.size,1))
    z = (scale/np.sqrt(2*np.pi))*np.exp(-(x-loc.T)**2/(2*scale**2))
    return z/z.sum(1,keepdims=1)

def probas(modif):
    z = (norm_distrib(x = state_space,loc = r*state_space + (1-r)*u + modif,scale = np.sqrt(1-r**2)*σ))
    return(z)

prob_matrixes = np.array([probas(0),probas(-β),probas(β),probas(-π)])
P = Population(10,4,10**5,prob_matrixes)
P.round(10)
