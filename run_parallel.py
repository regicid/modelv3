import pickle
import numpy as np
from model import Population
import multiprocess as mtp
MU = np.linspace(10,16,4)

def get_pop(mu):
        P = Population(mu)
        P.round(10)
        return(P)

l = mtp.Pool(processes=2)
runs = l.map_async(get_pop,MU)
l.close()
l.join()
Results = []
for run in runs.get():
    Results.append(run)


