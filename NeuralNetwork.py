import mxnet as mx
from mxnet import nd, autograd       


             
             
class Code:
    """Represent a linear code"""
    
    def __init__(self,k,n,G):
        #A changer (k = nb de ligne de G n = nb de colonnes)
        #Peut etre rajouter H pour le calcul efficace de dmin
        self.k = k
        self.n = n
        self.G = G
        
        self.dmin = self.computeDmin()
        
    def computeDmin(self):
        #flemme
        return 3
        
        
    
    


class NeuralNetwork:
    """Represent a neural network able to correct a code.
    
    Can be trained and used on data.
    Can also be stored in file or create from a file"""
    
    ###########Training characteristics and methods####################
    ctx = mx.cpu(0)
    nbIter = 50
    batchSize = 100
    
    def SE(yhat,y):
        return nd.sum((yhat - y) ** 2)
    SE = staticmethod(SE)
    
    ##pourrait être une methode d'objet
    def adam(params, vs, sqrs, lr, batch_size, t):
        beta1 = 0.9
        beta2 = 0.999
        eps_stable = 1e-8

        for param, v, sqr in zip(params, vs, sqrs):
            g = param.grad / batch_size

            v[:] = beta1 * v + (1. - beta1) * g
            sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

            v_bias_corr = v / (1. - beta1 ** t)
            sqr_bias_corr = sqr / (1. - beta2 ** t)

            div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
            param[:] = param - div
    
    adam = staticmethod(adam)
    ###############################################
    
    
    
        
    def __init__(self, code, insideLayersNumber, sizes):
        self.code = code
        self.layersNumber = insideLayersNumber
        self.sizes = [code.n] + sizes + [code.k]
        
        self.params = list()
        
        for i in range(self.layersNumber+1):
            self.params.append(nd.random_normal(shape=(self.sizes[i],self.sizes[i+1]),ctx=self.ctx))
            self.params.append(nd.random.normal(shape = (self.sizes[i+1],),ctx=self.ctx))
        
        
        
        self.t = 1
        self.vs = []
        self.sqrs = []
        
        for param in self.params:
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())
        
        self.lr = 0.01
    
    
    
    ##Peut etre rajouter une autre methode qui fait le round en sortie
    def net(self,input):
        L = input
        for i in range(self.layersNumber+1):
            L = nd.sigmoid(nd.dot(L,self.params[2*i])+self.params[2*i+1])
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

                y = (x + noiseBSC)%2

                with autograd.record():
                    zHat = self.net(y)
                    loss = self.SE(zHat,z)
                loss.backward()

                self.adam(self.params,self.vs,self.sqrs, self.lr, self.batchSize, self.t)
                self.t+=1
        
                cumuLoss += loss.asscalar()
                zHat = nd.round(zHat)
                efficiency += nd.sum(nd.equal(zHat,z)).asscalar()

                
            Pc = efficiency/(self.batchSize*self.nbIter*self.code.k)
            Pe = 1 - Pc
            normCumuLoss = cumuLoss/(self.batchSize*self.nbIter*self.code.k)
            print("Epochs %d: Pe = %lf , loss = %lf" % (i,Pe,normCumuLoss))
        
    
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

                
if __name__ == "__main__":
    G= nd.array([[1, 0, 0, 0,1,0,1],
                [0, 1, 0, 0,1,1,0],
                [0, 0, 1, 0,1,1,1],
                [0, 0, 0, 1,0,1,1]],ctx=mx.cpu(0))
    code = Code(4,7,G)
    net = NeuralNetwork(code,2,[7,7])
                