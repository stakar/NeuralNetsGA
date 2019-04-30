import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import math

rin = np.random.randint
ran = np.random.random


def create_instance(placeholder,steady_state=ran()):
    shape_hold = placeholder.shape
    #Adding some noise
    
    for n in range(rin(0,5)):
        placeholder[rin(0,shape_hold[0]),
                    rin(0,shape_hold[1])] = ran()
    start_point1 = rin(1,shape_hold[0]-1)
    start_point2 = rin(1,shape_hold[1]-1)
    plus_state = steady_state
    placeholder[start_point1-1:start_point1+2,start_point2] = plus_state
    placeholder[start_point1,start_point2-1:start_point2+2] = plus_state

    return placeholder

def create_random_inst(placeholder):
    shape_hold = placeholder.shape
    for n in range(rin(0,10)):
        placeholder[rin(0,shape_hold[0]),
                    rin(0,shape_hold[1])] = 1
    return placeholder


#sigmoid and its derivative
def sigmoid(x, deriv = False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0,x)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import math

class GenAlArchNNv4(object):

    def __init__(self,n_pop=10,n_gen=17,n_hid = 8,
                 mut_prob=0.02,desired_fit=0.8,max_gen = 300):

        """
        Features selector that uses Genetic Algorithm.

        Parameters
        ----------
        n_pop : int
        number of individuals in pop

        mut_prob : float
        probability of mutation, default 0.02

        desired_fit : float
        accuracy that has to be achieved for algorithm to stop

        max_gen
        maximum number of generations. The threshold that is not supposed to cros
        sed, the limit of algorithm. It is safety limit, that one can moderate,
        so algorithm does not work forever.

        Attributes
        ----------

        pop : array [n_pop,n_gen]
        placeholder array for population with shape [number of individuals, num
        ber of features]

        pipeline : Pipeline
        Pipeline object, created using make_pipeline function from sklearn, taki
        ng as steps scaler and classifier passed as parameters

        """

        self.self = self
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.pop = np.zeros((n_pop,n_gen))
        self.mut_prob = mut_prob
        self.desired_fit = desired_fit
        self.max_gen = max_gen
        self.n_generation = 0
        self.n_hid = n_hid
        
    @staticmethod
    def get_gene():
        """ Returns positive value """
        return np.random.random()

    def fit(self,data,target):
        
        self.data = data
        self.target = target
        n_input = self.data.shape[1]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.33)
        self.pop = list()
        #in this version number of neurons in first layer is constant

        n_out = self.y_train.shape[1]
        
        #Create individuals
        for n in range(self.n_pop):
            #Generating random number of perceptrons in hidden layer
            n_hid = self.n_hid
            #Weights of each layer
            w0 = 2*np.random.uniform(0,1,size=(n_input,n_hid))-1 
            w1 = 2*np.random.uniform(0,1,size=(n_hid,n_out))-1
            #Biases weights
            bw1 = 2*np.random.uniform(0,1,size=(n_hid))-1
            bw2 = 2*np.random.uniform(0,1,size=(n_out))-1
            #merge into individual
            new = np.array((w0,w1,bw1,bw2))         
            #append to population
            self.pop.append(new)
        #create first population's fitness
        self.pop_fit = self._pop_fitness(self.pop)
        #add first population's fitness to the hall of glory
        #needed for later plotting
        self.past_pop = self.pop_fit.copy()
            
    def _check_fitness(self,gene):
        
        w0,w1,bw0,bw1 = gene
        
        #Feed forward
        layer0 = self.X_train
        
        #sum of weights plus bias for layer 1
        wsum0 = np.dot(layer0, w0) + bw0
        layer1 = sigmoid(wsum0)
        
        #sum of weights plus bias for layer 2
        wsum1 = np.dot(layer1, w1) + bw1
        layer2 = relu(wsum1)
        
        layer2_error = self.y_train - layer2
        error = np.mean(np.abs(layer2_error))
        
        accuracy = (1 - error) 
        
        return(accuracy)
        
    def validate(self):
        best = self.pop[np.argmax(self.pop_fit)]
        
        w0,w1,bw0,bw1 = best
        
        #Validate
        layer0 = self.X_test
        layer1 = sigmoid(np.dot(layer0, w0)+bw0)
        layer2 = relu(np.dot(layer1, w1)+bw1)

        layer2_error = self.y_test - layer2

        error = np.mean(np.abs(layer2_error))
        accuracy = (1 - error) * 100
        return accuracy
        
    def _pop_fitness(self,pop):
        """ Checks the fitness for each individual in pop, then returns
        it """
        return np.array([self._check_fitness(n) for n in pop])
    
    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendant
        s, array of characters with shape [2,n_gen] """
        try:
            if len(mother[1])<len(father[1]):
                parent = mother
            else:
                parent = father
            coef1 = parent[0].copy()
            coef2 = parent[1].copy()
            coef3 = parent[2].copy()
            coef4 = parent[3].copy()
            for coef,partner in zip([coef1,coef2],[0,1]):
                for n in range(len(parent[partner])):
                    for z in range(len(parent[partner][n])):
                        dec = rin(0,2)
                        if dec == 1:
                            coef[n][z] = parent[partner][n][z]
            for coef,partner in zip([coef2,coef3],[2,3]):
                for n in range(len(parent[partner])):
                    dec = rin(0,2)
                    if dec == 1:
                        coef[n] = parent[partner][n]
            return coef1,coef2,coef3,coef4
        except:
            dec = rin(0,2)
            if dec == 1:
                return father
            else:
                return mother

    @staticmethod
    def _pairing2(mother,father):
        """ Method for pairing chromosomes and generating descendant
        s, array of characters with shape [2,n_gen] """
        coef = mother.copy()
        for parent in range(4):
            arrlen = np.minimum(mother[parent].shape[0],
                                father[parent].shape[0])
            n_coef = np.random.randint(0,arrlen)
            n_coef2 = arrlen - n_coef
            dec = rin(0,2)
            if dec == 1:
                try:
                    coef[parent] = np.concatenate([father[parent][:n_coef],
                                                     mother[parent][-n_coef2:]])
                except:
                    return mother
            else:
                try:
                    coef[parent] = np.concatenate([mother[parent][:n_coef],
                                                     father[parent][-n_coef2:]])
                except:
                    return father
        return coef

    def transform(self):
        """ Transform, i.e. execute an algorithm.

        attributes
        ----------
        pop_fit : list
        list of population's fitness, i.e. fitness of each individual

        past_pop : array [n_generation,len(pop_fit)]
        past population's fitnesses

        best_ind : float
        best founded fitness

        n_generation : int
        number of generation already created

        """
        self.best_ind = np.max(self.past_pop)
        while self.best_ind < self.desired_fit:
            self.descendants_generation()
            self.pop_fit = self._pop_fitness(self.pop)
              #Comment out line below to print every population's fitness
#             if (self.n_generation % 1) == 0:
#                 print(self.pop.shape)
            self.best_ind = np.max(self.pop_fit)
            self.random_mutation()
            self.n_generation += 1
            if self.n_generation > self.max_gen:
                break


    def fit_transform(self,data,target):
        """ Fits the data to model, then executes an algorithm. """
        self.fit(data,target)
        self.transform()

    def descendants_generation(self):
        """ Selects the best individuals, then generates new pop, with
        half made of parents (i.e. best individuals) and half children(descendan
        ts of parents) """
        #Two firsts individuals in descendants generation are the best individua
        #ls from previous generation
        self.past_pop = np.vstack([self.past_pop,self.pop_fit])
        n_prev = round(self.n_pop/10)
        for n in range(n_prev):
              self.pop[n] = self.pop[np.argsort(self.pop_fit)[-(n_prev-n)]]
        #now,let's select best ones
        parents_pop = self.roulette()
        #Finally, we populate new generation by pairing randomly chosen best
        #individuals
        for n in range(n_prev,self.n_pop-1):
            father = parents_pop[np.random.randint(self.n_pop)]
            mother = parents_pop[np.random.randint(self.n_pop)]
            dec = rin(0,100)
            if dec == 1:
                children = self._pairing(mother,father)
            else:
                children = self._pairing2(mother,father)
            dec = rin(0,100)
            if dec == 2:
                children = self.reducing(children)
            elif dec == 3:
                children = self.growing(children)
            self.pop[(n)] = children

    
    @staticmethod
    def growing(gene):
        
        w0,w1,bw0,bw1 = gene
        w0 = np.hstack([w0,np.random.uniform(0,1,size=w0.shape[0]).reshape(w0.shape[0],1)])
        w1 = np.vstack([np.random.uniform(0,1,size=w1.shape[1]).reshape(1,w1.shape[1]),w1])
        bw0 = np.hstack([bw0,ran()])

        return w0,w1,bw0,bw1

    @staticmethod
    def reducing(gene):
        w0,w1,bw0,bw1 = gene
        try:
            worst = mode(np.where(np.abs(w0) < 0.05)[1])[0][0]
        except:
            return gene
        w0 = np.delete(w0,worst,1)
        w1 = np.delete(w1,worst,0)
        bw0 = np.delete(bw0,worst)
        return w0,w1,bw0,bw1
    
                
    def random_mutation(self):
        """ Randomly mutates the pop, for each individual it checks wheth
        er to do it accordingly to given probability, and then generates new cha
        racter on random locus """
        pop = self.pop.copy()
        for n in range(self.n_pop):
            decision = np.random.uniform(0,1)
            if decision < self.mut_prob:
#                 for n in range(np.random.randint(10)):
                which_layer = np.random.randint(0,2)
                which_locus = np.random.randint(0,len(pop[n][which_layer]))
                pop[n][which_layer][which_locus] == self.get_gene()
        self.pop = pop

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
        pop_fitness = self.pop_fit
        wheel = np.zeros((self.n_pop,3))
        prob = 0
        for n in range(self.n_pop):
            ind_prob = prob + (pop_fitness[n] / np.sum(pop_fitness))
            wheel[n] = [n,prob,ind_prob]
            prob = ind_prob
        return wheel

    def roulette_swing(self,wheel):
        """ This method takes as an input roulette wheel and returns an index of
        randomly chosen field """
        which = np.random.random()
        for n in range(len(wheel)):
            if which > wheel[n][1] and which < wheel[n][2]:
                return int(wheel[n][0])

    def roulette(self):
        """ This method performs selection of individuals, it takes the coeffici
        ent k, which is number of new individuals """
        wheel = self.roulette_wheel()
        return np.array([self.pop[self.roulette_swing(wheel)]
                         for n in range(self.n_pop)])

    def plot_fitness(self,title='Algorithm performance'):
        """ It checks the mean fitness for each passed pop and the fitnes
        s of best idividual, then plots it. It does not show the plotted figure,
        (unless last line is uncommented), but instead saves the plot under pass
        ed title

        Parameters
        ----------
        title : string
        title of plot, under which it is saved
        """
        N = self.past_pop.shape[0]
        t = np.linspace(0,N,N)
        past_fit_mean = [np.mean(self.past_pop[n]) for n in range(N)]
        past_fit_max = [np.max(self.past_pop[n]) for n in range(N)]
        plt.plot(t,past_fit_mean,label='pop mean fitness')
        plt.plot(t,past_fit_max,label='pop best individual\'s fitness')
        plt.xlabel('Number of generations')
        plt.ylabel('Fitness')
        plt.legend()
        plt.title(title)
        plt.savefig(title)
#         plt.show()
    
# ga = GenAlArchNN(n_pop=1000,max_gen=100,desired_fit=0.95)
# ga.fit(X_train,y_train)
# ga.transform()
# ga.plot_fitness()