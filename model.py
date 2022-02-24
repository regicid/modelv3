from scipy.stats import norm
from scipy.special import binom as binom_coef
import multiprocessing as mtp # Parts of the code are parallelized, make sure your configuration allows it
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import copy

def dyn_prog(T,p,v,prob_matrixes,n=10,r=.99,γ=1/3,m=.01,f2=.05,ω=.1,β=10,π=20,state_space = np.round(np.linspace(-50,50,1001),1)):
    
    return fitness, exp_fitness, decisions


class Population:
    def __init__(self,μ,σ=4,N=10**5,T = 200,n=10,r=.99,γ=1/3,β=10,π=20,m=.01,f2=.05,ω=.1,state_space = np.round(np.linspace(-50,50,1001),1),initial_v=0,update_rate = 1):
        self.μ = μ
        self.σ = σ
        self.N = N
        self.n = n
        self.T = T
        self.r = r
        self.β = β
        self.m = m
        self.f2 = f2
        self.ω = ω
        self.γ = γ
        self.update_rate = update_rate
        self.state_space = state_space
        self.states = np.round(np.random.normal(loc=self.μ,scale=self.σ,size=self.N),1).clip(self.state_space.min(),self.state_space.max())
        self.strategies = np.zeros(self.N,dtype="int8")
        self.p = 0
        self.v = initial_v
        def norm_distrib(x,loc,scale=np.sqrt(1-r**2)*σ):
            x = np.array(x)
            loc = np.array(loc)
            x = np.tile(x,(loc.size,1))
            loc = np.tile(loc,(loc.size,1))
            z = (scale/np.sqrt(2*np.pi))*np.exp(-(x-loc.T)**2/(2*scale**2))
            return z/z.sum(1,keepdims=1)

        def probas(modif):
            z = (norm_distrib(x = state_space,loc = r*state_space + (1-r)*μ + modif,scale = np.sqrt(1-r**2)*σ))
            return(z)
        self.prob_matrixes = np.array([probas(0),probas(-10),probas(10),probas(-20)])
        p = (1-(1-self.p)**self.n)/self.n
        if self.v==1: 
            v=.999
        else:
            v = self.v
        fitness = (100+self.state_space)/2
        decisions = [0,0]
        #Strategy order: submissive, discriminate violent, indiscriminate violent, steal
        prob_stolen= [np.clip(p*(1-v**self.n)/(1-v),0,1),np.clip(p*v**(self.n-1),0,1)]
        prob_success = (1-v**self.n/2)
        outcomes = [] #Two levels: strategy, outcome
        prob_fight = [0,prob_stolen[1]+(1-prob_stolen[1])*self.m*v,v**self.n + (1-v**self.n)*(prob_stolen[1]+(1-prob_stolen[1])*self.m*v)]
        outcomes.append(np.einsum('ijk,i->jk',self.prob_matrixes,np.array([1-prob_stolen[0],prob_stolen[0],0,0])))
        outcomes.append(np.einsum('ijk,i->jk',self.prob_matrixes,[1-prob_stolen[1]/2,prob_stolen[1]/2,0,0]))
        outcomes.append(np.einsum('ijk,i->jk',self.prob_matrixes,[(1-self.γ)*prob_success*prob_stolen[1]/2,(1-self.γ)*(1-prob_success)*prob_stolen[1]/2,(1-self.γ)*prob_success*(1-prob_stolen[1]/2),self.γ]))
        for i in range(self.T):
            temp_fitness = []
            exp_fitness = np.empty(shape=(3,len(state_space)),dtype = "float")
            for strat in range(3):
                exp_fitness[strat,:] = np.dot(outcomes[strat],fitness*(1-self.ω*(state_space<0))*(1 - prob_fight[strat]*self.f2/2))
            decisions = np.argmax(exp_fitness,axis=0)
            fitness = np.max(exp_fitness,axis=0)
        strategies = decisions
        positions = ((self.states-state_space.min())/(self.state_space[1]-self.state_space[0])).round().astype('int')
        self.strategies = strategies[positions]
        z = self.strategies < 2
        zz = np.random.random(z.sum()) < initial_v
        self.strategies[z] = zz
    def update_strategies(self):
        self.p = np.mean(self.strategies==2)
        self.v = np.mean(self.strategies>0)
        p = (1-(1-self.p)**self.n)/self.n
        if self.v==1: 
            v=.999
        else:
            v = self.v
        fitness = (100+self.state_space)/2
        decisions = [0,0]
        #Strategy order: submissive, discriminate violent, indiscriminate violent, steal
        prob_stolen= [np.clip(p*(1-v**self.n)/(1-v),0,1),np.clip(p*v**(self.n-1),0,1)]
        prob_success = (1-v**self.n/2)
        outcomes = [] #Two levels: strategy, outcome
        prob_fight = [0,prob_stolen[1]+(1-prob_stolen[1])*self.m*v,v**self.n + (1-v**self.n)*(prob_stolen[1]+(1-prob_stolen[1])*self.m*v)]
        outcomes.append(np.einsum('ijk,i->jk',self.prob_matrixes,np.array([1-prob_stolen[0],prob_stolen[0],0,0])))
        outcomes.append(np.einsum('ijk,i->jk',self.prob_matrixes,[1-prob_stolen[1]/2,prob_stolen[1]/2,0,0]))
        outcomes.append(np.einsum('ijk,i->jk',self.prob_matrixes,[(1-self.γ)*prob_success*prob_stolen[1]/2,(1-self.γ)*(1-prob_success)*prob_stolen[1]/2,(1-self.γ)*prob_success*(1-prob_stolen[1]/2),self.γ]))
        for i in range(self.T):
            temp_fitness = []
            exp_fitness = np.empty(shape=(3,len(self.state_space)),dtype = "float")
            for strat in range(3):
                exp_fitness[strat,:] = np.dot(outcomes[strat],fitness*(1-self.ω*(self.state_space<0))*(1 - prob_fight[strat]*self.f2/2))
            decisions = np.argmax(exp_fitness,axis=0)
            fitness = np.max(exp_fitness,axis=0)
        strategies = decisions
        z = np.random.random(self.N)<self.update_rate
        positions = ((self.states[z]-self.state_space.min())/(self.state_space[1]-self.state_space[0])).round().astype('int')
        self.strategies[z] = strategies[positions]
    def round(self,t):
        self.frequencies = np.zeros(shape = (3,t))
        for z in range(t):
            ### Choices
            self.update_strategies()
            ### Actions' consequences
            groups = []
            perm = np.random.permutation(self.N)     
            for i in range(np.int(self.N/self.n)+1):
                groups.append(np.array(perm[(i*self.n):(i+1)*self.n]))
            for group in groups:
                strat = self.strategies[group]
                if 3 in strat:
                    stealer = np.random.choice(group[strat==3])
                    targets = np.delete(group,np.where(group==stealer)[0])
                    target = np.random.choice(targets[(strat==0)+ (0 not in strat)])
                    caught = (np.random.random()<γ)
                    if self.strategies[target]>0:
                        fight_winner = np.random.random()>.5 #Symetric fight
                        self.states[target] -= self.β*fight_winner
                        self.states[stealer] += self.β*(1-fight_winner)*(1-caught) - self.π*caught
                    else: #If the target is non violent, the stealer just takes resources
                        self.states[target] -= self.β
                        self.states[stealer] += self.β*(1-caught) - self.π*caught
            
            #Shuffle the states (social mobility)
            fluctuations = np.random.normal(loc=self.μ,scale=np.sqrt(1-self.r**2)/(1-self.r)*self.σ,size=self.N)
            self.states = self.r*self.states + (1-self.r)*fluctuations
            self.states = self.states.clip(np.min(self.state_space),np.max(self.state_space)).round(1)
            
            #Record behaviours
            freq = np.unique(self.strategies,return_counts=True)
            self.frequencies[freq[0],z] = freq[1]/self.N
