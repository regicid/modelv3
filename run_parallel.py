import pickle
import numpy as np
from model import Population
import multiprocess as mtp
MU = np.linspace(10,16,20)

def get_pop(mu):
        P = Population(mu)
        P.round(10)
        pickle.dump(P,open(f"result_{mu}.pl","wb"))

l = mtp.Pool(processes=20)
l.map_async(get_pop,MU)
l.close()
l.join()
Results = []
for run in runs.get():
    Results.append(run)


