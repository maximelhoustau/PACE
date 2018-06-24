# -*- coding: utf-8 -*-

import mxnet as mx
from mxnet import nd, autograd
import random
from math import floor
from itertools import permutations


def ampliOP(x):
    #return 0.01*(nd.relu(0.1*(x+10))-nd.relu(0.1*(x+0.5))) + 0.98*(nd.relu(x+0.5)-nd.relu(x-0.5)) + 0.01*(nd.relu(0.1*(x-1))-nd.relu(0.1*(x-11)))
    return nd.relu(x+0.5)-nd.relu(x-0.5)
    
def ampliOPSmooth(x):
    b1 = x<-0.4
    b2 = x>0.4
    return b1*0.1*(nd.exp(x+0.4)) + (1-b1)*(1-b2)*(x+0.5) + b2*(0.9+0.1*(1-nd.exp(-(x-0.4))))

def sigmoid(x):
    return nd.sigmoid(4*x)

def echelon(x):
    return x>0
    


def netS(net, input):
    L1 = net.fct(nd.dot(input,net.params[0])+net.params[1])
    L2 = net.fct(nd.dot(L1,net.params[2])+net.params[3])
    L3 = net.fct(nd.dot(L2,net.params[4])+net.params[5])
    L4 = net.fct(nd.dot(L3,net.params[6])+net.params[7])
    L5 = net.fct(nd.dot(L4,net.params[8][:,:4])+net.params[9][:4])
    L6 = net.fct(nd.dot(L5,net.params[10][:4,:]) + nd.dot(input[:,:4],net.params[10][4:,:]) + net.params[11])
    return net.fct(nd.dot(L6,net.params[12])+net.params[13])

def performancesS(net):
    wrong = []
    cpt = 0
    x = net.code.reachableWords
    zHat = nd.round(net.net(x))
    for i in range(len(net.code.E)):
        for j in range((net.code.n+1)*i,(net.code.n+1)*(i+1)):
            if (nd.sum(nd.equal(net.code.E[i],zHat[j])) != net.code.k):
                cpt+=1
                wrong.append((net.code.E[i],net.code.reachableWords[j]))
    return (cpt,len(net.code.reachableWords),wrong)

def syndromeMode():
    NeuralNetwork.net = netS
    NeuralNetwork.fct = echelon
    NeuralNetwork.computePerformances = performancesS
    net = NeuralNetwork.open("syndrome.txt")
    for param in net.params:
        param.attach_grad(grad_req = 'null')
        for ligne in param:
            for elt in ligne:
                if elt != 0:
                    elt.attach_grad()
    return net
    
 
    
class Code:
    """Represent a linear code
    if k is small, it computes all the code : G : E -> F"""
    kmax = 10
    nmax = 15
    
    def __init__(self,G, compute = True, ctx = mx.cpu(0)):
        #Peut etre rajouter H pour le calcul efficace de dmin
        self.k = len(G)
        self.n = len(G[0])
        self.G = nd.array(G, ctx = ctx)
        self.ctx = ctx

        
        if self.k<=self.kmax and compute:
            self.compute()
        
    
    
    def compute(self):
        self.size = 2**self.k
        self.E = [[0 for j in range(self.k)] for i in range(self.size)]
        for i in range(self.size):
            s = ("{:0"+str(self.k)+"b}").format(i)
            for j in range(len(s)):
                self.E[i][j] = s[j]
        self.E = nd.array(self.E, ctx = self.ctx)
        self.F = nd.dot(self.E,self.G)%2
        
        self.dmin = nd.sum(nd.equal(self.F[0],self.F[1])).asscalar()
        
        for i in range(2,len(self.F)):
            d = nd.sum(self.F[i]).asscalar()
            if d<self.dmin:
                self.dmin = d
        
        self.t = int(floor((self.dmin-1)/2))
        
        if self.n>self.nmax:
            return None
        
        ##Words with less than t mistakes
        self.corrigibleWords = []
        self.corrigibleE = []
        
        for nb in range(self.t+1):
            t = [1]*nb+[0]*(self.n-nb)
            combinations = []
            for perm in permutations(t):
                if not perm in combinations:
                    combinations.append(perm)
            for i in range(len(self.F)):
                for error in combinations:
                    corrigibleWord = []
                    for j in range(self.n):
                        corrigibleWord.append((self.F[i,j].asscalar()+ error[j])%2)
                    self.corrigibleWords.append(corrigibleWord)
                    self.corrigibleE.append(list(self.E[i].asnumpy()))
        self.corrigibleWords = nd.array(self.corrigibleWords,ctx = self.ctx)
        self.corrigibleE = nd.array(self.corrigibleE, ctx = self.ctx)
        
        ##Words with only one mistake
        self.reachableWords = []
        
        for word in self.F.asnumpy():
            self.reachableWords.append(list(word))
            for i in range(len(word)):
                word[i] = (word[i]+1)%2
                self.reachableWords.append(list(word))
                word[i] = (word[i]+1)%2
        self.reachableWords = nd.array(self.reachableWords, ctx = self.ctx)
        
        
        
        
        
    
    


class NeuralNetwork:
    """Represent a neural network able to correct a code.
    
    Can be trained and used on data.
    Can also be stored in file or create from a file"""
    
    ###########Training characteristics and methods####################
    nbIter = 50
    batchSize = 500
    
    def SE(yhat,y):
        return nd.sum((yhat - y) ** 2)
    SE = staticmethod(SE)
    
    ##pourrait etre une methode d'objet
    def adam(params, vs, sqrs, maximums, lr, batch_size, t):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        for param, v, sqr, maximum in zip(params, vs, sqrs, maximums):
            g = param.grad / batch_size

            v[:] = beta1 * v + (1. - beta1) * g
            sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

            v_bias_corr = v / (1. - beta1 ** t)
            sqr_bias_corr = sqr / (1. - beta2 ** t)

            div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
            param[:] = param - div
            param[:] = nd.where(param > maximum, maximum, param)
            param[:] = nd.where(param < -maximum, -maximum, param)
            
            
    
    adam = staticmethod(adam)
    ###############################################
    
    
    
        
    def __init__(self, code, insideLayersNumber, sizes, ctx = None, fct = ampliOP, maximum = 1):
        self.ctx = code.ctx
        if ctx:
            self.ctx = ctx
            if code.ctx != ctx:
                code = Code(code.G, ctx = ctx)
                
                
        self.fct = fct
        
        self.code = code
        self.layersNumber = insideLayersNumber
        self.sizes = [code.n] + sizes + [code.k]
        
        self.params = list()
        self.maximums = []
        
        for i in range(self.layersNumber+1):
            self.params.append(nd.random_normal(loc = 0,scale = 0.05,shape=(self.sizes[i],self.sizes[i+1]),ctx=self.ctx))
            self.params.append(nd.random.normal(loc = 0,scale = 0.05, shape = (self.sizes[i+1],),ctx=self.ctx))
            self.maximums.append(nd.array([[maximum for l in range(self.sizes[i+1])] for k in range(self.sizes[i])]))
            self.maximums.append(nd.array([maximum for l in range(self.sizes[i+1])]))
        
        
        self.t = 1
        self.vs = []
        self.sqrs = []
        
        for param in self.params:
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())
        
        self.lr = 0.001
    
    
    def size(self):
        S = 0
        for param in self.params:
            S+= param.size
        return S
    
    ##Peut etre rajouter une autre methode qui fait le round en sortie
    def net(self,input):
        L = input
        for i in range(self.layersNumber+1):
            L = self.fct(nd.dot(L,self.params[2*i])+self.params[2*i+1])
        return L
        
    
    
    def train(self,epochs):
        for i in range(epochs):
            efficiency = 0
            cumuLoss = 0
            for j in range(self.nbIter):
                z = nd.round(nd.random.uniform(0,1,(self.batchSize,self.code.k),ctx=self.ctx))
                x = nd.dot(z,self.code.G)%2

                noiseBSC = nd.random.uniform(0.01,0.99,(self.batchSize,self.code.n),ctx=self.ctx)
                noiseBSC = nd.floor(noiseBSC/nd.max(noiseBSC,axis=(1,)).reshape((self.batchSize,1)))
                actif = nd.array([[random.uniform(0,1)>0.125]*self.code.n for k in range(self.batchSize)], ctx = self.ctx)
                noiseBSC = noiseBSC * actif
                
                
                y = (x + noiseBSC)%2

                with autograd.record():
                    zHat = self.net(y)
                    loss = self.SE(zHat,z)
                loss.backward()

                self.adam(self.params,self.vs,self.sqrs, self.maximums, self.lr, self.batchSize, self.t)
                self.t+=1
        
                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()

                
            Pc = efficiency/(self.batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(self.batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))
    
    def train2(self,epochs):
        batchSize = self.code.size
        x = self.code.F
        z = self.code.E
        
        for i in range(epochs):
            efficiency = 0
            cumuLoss = 0
            for j in range(self.nbIter):
                noiseBSC = nd.random.uniform(0.01,0.99,(batchSize,self.code.n),ctx=self.ctx)
                noiseBSC = nd.floor(noiseBSC/nd.max(noiseBSC,axis=(1,)).reshape((batchSize,1)))
                actif = nd.array([[random.uniform(0,1)>0.125]*self.code.n for k in range(batchSize)], ctx = self.ctx)
                noiseBSC = noiseBSC * actif
                
                y = (x + noiseBSC)%2

                with autograd.record():
                    zHat = self.net(y)
                    loss = self.SE(zHat,z)
                loss.backward()

                self.adam(self.params,self.vs,self.sqrs, self.maximums, self.lr, batchSize, self.t)
                self.t+=1
        
                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()

               
            Pc = efficiency/(batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))
    
    def train3(self,epochs):
        batchSize = len(self.code.reachableWords)
        z = []
        for elt in self.code.E.asnumpy():
            z.extend([list(elt)]*(self.code.n+1))
        z = nd.array(z,ctx=self.ctx)
        x = self.code.reachableWords
        
        for i in range(epochs):
            efficiency = 0
            cumuLoss = 0
            for j in range(self.nbIter):
                with autograd.record():
                    zHat = self.net(x)
                    loss = self.SE(zHat,z)
                loss.backward()
                
                self.adam(self.params,self.vs,self.sqrs, self.maximums, self.lr, batchSize, self.t)
                self.t+=1
        
                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()
 
            Pc = efficiency/(batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))

            
            
    def computePerformances(self):
        wrong = []
        cpt = 0
        for i in range(len(self.code.E)):
            for word in self.code.reachableWords[(self.code.n+1)*i:(self.code.n+1)*(i+1)]:
                zhat = self.net(word)
                for diff in nd.round(zhat+self.code.E[i]):
                    if diff.asscalar()%2 != 0:
                        wrong.append((self.code.E[i],word))
                        cpt+=1
                        break
        
        return (cpt,len(self.code.reachableWords),wrong)
                
    
    ##overwrite the file ./file
    def save(self,file):
        with open(file,"w") as f:
            f.write(self.toString())
    
    def toString(self):
        s = "/Dimension :\n\n"
        s += str(self.layersNumber) + "\n"
        for size in self.sizes:
            s += str(size) + " "
    
        G = self.code.G.asnumpy()
        s += "\n\n/Code :\n"
        for ligne in G:
            s+="\n"
            for value in ligne:
                s+=str(value) + " "
            
        
        s+="\n\n/Parameters :\n"
        
        for param in self.params:
            s+="\n"
            for ligne in param:
                for value in ligne:
                    s+= str(value.asnumpy()[0]) + " "
                s+="\n"
        return s

    
    def open(file, ctx = mx.cpu(0)):
        s = ""
        with open(file,"r") as f:
            s = f.read()
        return NeuralNetwork.stringToNet(s, ctx = ctx)
    open = staticmethod(open)
    
    def stringToNet(s, ctx = mx.cpu(0)):
        tab = s.split("\n")
        tab2 = []
        for chain in tab:
            if "/" not in chain and chain != "":
                tab2.append(chain.strip(" "))
        
        insideLayersNumber = int(tab2.pop(0))
        sizes = []
        
        for chain in tab2.pop(0).split(" "):
            sizes.append(int(chain))
        k = sizes[-1]
        n = sizes[0]
        G = nd.array([[0 for j in range(n)] for i in range(k)],ctx=mx.cpu(0))
            
        for i in range(k):
            ligne = tab2.pop(0).split(" ")
            for j in range(n):
                G[i,j] = float(ligne[j])
        
        code = Code(G, ctx)
        net = NeuralNetwork(code,insideLayersNumber,sizes[1:-1], ctx)
        
        for param in net.params:
            for ligne in param:
                ligne_fichier = tab2.pop(0).split(" ")
                for i in range(len(ligne_fichier)):
                    ligne[i] = float(ligne_fichier[i])
        
        return net
    stringToNet = staticmethod(stringToNet)
                
#if __name__ == "__main__":
G= nd.array([[1, 0, 0, 0,1,0,1],
             [0, 1, 0, 0,1,1,0],
             [0, 0, 1, 0,1,1,1],
             [0, 0, 0, 1,0,1,1]],ctx=mx.cpu(0))
                
G2 = nd.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
               [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], 
               [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
               [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]], ctx = mx.cpu(0))
              
G3 = nd.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]], ctx = mx.cpu(0))

             
code = Code(G)
net = NeuralNetwork(code,3,[8,8,8])
