import pickle
import numpy as np
from model import *
import multiprocessing as mtp
MU = np.linspace(10,16,10)

def get_pop(mu):
        prob_matrixes = np.array([probas(0),probas(-10),probas(10),probas(-20)])
        P = Population(mu,4,10**5,prob_matrixes,update_rate = .1)
        P.round(10)
        pickle.dump(P,open(f"result_{mu}.pl","wb"))

l = mtp.Pool(processes=10)
runs = l.map_async(get_pop,MU)
l.close()
l.join()
Results = []
for run in runs.get():
    Results.append(run)


