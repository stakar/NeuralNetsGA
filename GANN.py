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

def create_instance(placeholder):
    shape_hold = placeholder.shape
    #Adding some noise
    for n in range(rin(0,5)):
        placeholder[rin(0,shape_hold[0]),
                    rin(0,shape_hold[1])] = ran()
    start_point1 = rin(1,shape_hold[0]-1)
    start_point2 = rin(1,shape_hold[1]-1)
    placeholder[start_point1-1:start_point1+2,start_point2] = 1
    placeholder[start_point1,start_point2-1:start_point2+2] = 1
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


class GenAlWeightsNN(object):

    def __init__(self,n_pop=10,n_gen=17,
                 mut_prob=0.02,desired_fit=0.6,max_gen = 300,
                 scaler = MinMaxScaler(),
                 clf = MLPClassifier(random_state=42,max_iter=800,
                                     tol=1e-2)):

        """
        Features selector that uses Genetic Algorithm.

        Parameters
        ----------
        n_pop : int
        number of individuals in pop

        n_gen : int
        length of genotype, i.e. range of features among which we can select in
        fact, it is determined by maximal length of Chromosome's attribute genot
        ype.

        scaler : class
        sklearn scaler used for scalling data

        clf : class
        sklearn classifier used for classification

        mut_prob : float
        probability of mutation, default 0.02

        desired_fit : float
        accuracy that has to be achieved for algorithm to stop

        max_gen
        maximum number of generation. The threshold that is not supposed to cros
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
        self.scaler = scaler
        self.clf = clf
        self.pipeline = make_pipeline(self.scaler,self.clf)
        # self.pipeline = self.clf
        self.mut_prob = mut_prob
        self.desired_fit = desired_fit
        self.max_gen = max_gen
        self.n_generation = 0
        
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

        #Creates individuals
        for n in range(self.n_pop):
            w0 = 2*np.random.random((n_input,5))-1 
            w1 = np.random.random((5,2))
            new = np.array((w0,w1))            
            self.pop.append(new)

        self.pop_fit = self._pop_fitness(self.pop)
        self.past_pop = self.pop_fit.copy()

    def _check_fitness(self,gene):
        
        w0,w1 = gene
        
        #Feed forward
        layer0 = self.X_train
        layer1 = sigmoid(np.dot(layer0, w0))
        layer2 = sigmoid(np.dot(layer1, w1))
        
        layer2_error = self.y_train - layer2
        error = np.mean(np.abs(layer2_error))
        
        accuracy = (1 - error) 
        
        return(accuracy)
        
    def validate(self):
        best = self.pop[np.argmax(self.pop_fit)]
#         print("Best individual accuracy on train dataset:{}".format(np.round(ga.best_ind,2)))
        
        w0,w1 = best
        
        #Validate
        layer0 = self.X_test
        layer1 = sigmoid(np.dot(layer0, w0))
        layer2 = sigmoid(np.dot(layer1, w1))

        layer2_error = self.y_test - layer2

        error = np.mean(np.abs(layer2_error))
        accuracy = (1 - error) * 100
        return(accuracy)
#         print("Best individual accuracy on test dataset:{}".format(np.round(accuracy,2)))
        
        
    def _pop_fitness(self,pop):
        """ Checks the fitness for each individual in pop, then returns
        it """
        return np.array([self._check_fitness(n) for n in pop])

    @staticmethod
    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendant
        s, array of characters with shape [2,n_gen] """
        n_coef1 = np.random.randint(0,len(mother[0]))
        n_coef2 = np.random.randint(0,len(mother[1]))
        coef1 = mother[0].copy()
        coef2 = mother[1].copy()
        for coef,parent in zip([coef1,coef2],[0,1]):
            for n in range(len(mother[parent])):
                for z in range(len(mother[parent][n])):
                    dec = rin(0,2)
                    if dec == 1:
                        coef[n][z] = father[parent][n][z]
        return coef1,coef2

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
        # print(pop_fit)
        parents_pop = self.roulette()
        #Finally, we populate new generation by pairing randomly chosen best
        #individuals
        for n in range(n_prev,self.n_pop-1):
                father = parents_pop[np.random.randint(self.n_pop)]
                mother = parents_pop[np.random.randint(self.n_pop)]
                children = self._pairing(mother,father)
                self.pop[(n)] = children

    def random_mutation(self):
        """ Randomly mutates the pop, for each individual it checks wheth
        er to do it accordingly to given probability, and then generates new cha
        racter on random locus """
        pop = self.pop.copy()
        for n in range(self.n_pop):
            decision = np.random.random()
            if decision < self.mut_prob:
#                 for n in range(np.random.randint(10)):
                which_layer = np.random.randint(0,2)
                which_locus = np.random.randint(0,len(pop[n][which_layer]))
                pop[n][which_layer][which_locus] == self.get_gene()
        self.pop = pop

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_populatio
        n, low_individual_probability,high_individual_probability]"""
#         pop_fitness = self._pop_fitness(self.pop)
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
        plt.show()
        
