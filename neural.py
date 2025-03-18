# import tensorflow as tf
import numpy as np

class Neural:
    def __init__(self, layers_sizes:list, activations:list, cost:str):
        
        self.layers = layers_sizes
        self.acts = activations
        self.cost = cost
        # self.params = self.parameters()

    def parameters(self, input_size):
        params = {}
        for i in range(len(self.layers)):
            if i == 0:
                params[f'W{i}'] = np.random.randn(self.layers[i], input_size)*0.01
            else:
                params[f'W{i}'] = np.random.randn(self.layers[i], self.layers[i-1])*0.01

            params[f'b{i}'] =np.zeros((self.layers[i], 1))

        return params
    

    def activation(self, z, act):
        if act == "sigmoid":
            a = 1/(1+np.exp(-z))
        elif act == "ReLU":
            a = np.maximum(0,z)
        elif act == "Leaky-ReLU":
            a = np.maximum(0.1*z,z)

        elif act == "softmax":
            exps = np.exp(z - np.max(z, axis=0, keepdims=True))
            a = exps / np.sum(exps, axis=0, keepdims=True)

        return a


    def forward(self, inputs):
        cache = []
        A = inputs
        for i in range(len(self.layers)):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = np.dot(W, A) + b
            c = [W, A, b, Z]# A here is actually Aprev
            A = self.activation(Z, act=self.acts[i])
            c.append(A)# A here is Acurr
            cache.append(c)
        return cache, A
    
    def cost_calc(self, loss, A, Y, m):
        if loss=="CCE":
            if self.acts[-1] == 'sigmoid':
                cost = -np.sum(Y*(np.log(A + 1e-8))+(1-Y)*(np.log(1-A + 1e-8)))/m
            elif self.acts[-1] == 'softmax':
                 # Avoid log(0) by adding a small epsilon
                epsilon = 1e-9
                pred = np.clip(pred, epsilon, 1 - epsilon)  

                cost = -np.sum(Y * np.log(pred)) / m
                
        return cost
    

    def backward(self, Y, pred, cache, alpha=0.01):
        dA=None
        for i in reversed(range(len(self.layers))):
            c =cache[i]
            W = c[0]
            Aprev = c[1]
            Z = c[3]
            A = c[4]

            if i == len(self.layers)-1:
                dZ = pred - Y
            else:
                if self.acts[i] == 'Leaky-ReLU':
                    g = np.where(Z>=0,1, 0.1)
                    
                elif self.acts[i] == 'ReLU':
                    g = np.where(Z>=0,1, 0)
                elif self.acts[i] == 'sigmoid':
                    g = A*(1-A)
                dZ = dA * g.astype(int)
                
            # if loss_type == 'CCE':
            # derivatives[f'dz{i}'] = dZ
            dW = np.dot(dZ, Aprev.T)/self.num_inputs
            db = np.sum(dZ, axis=1 ,keepdims=True)/self.num_inputs

            dA = np.dot(W.T, dZ)

            self.params[f'W{i}'] -= alpha*dW
            self.params[f'b{i}'] -= alpha*db


    

    def train(self, X_train, Y_train, epoch, learning_rate=0.01, X_dev=None, Y_dev=None,):
        self.input_size = X_train.shape[0]
        self.num_inputs = X_train.shape[1]
        dev = False
        if X_dev is not None:
            dev_input_size = X_dev.shape[1]
            dev = True


        self.params = self.parameters(self.input_size)
        J_hist = []
        dev_hist = []
        for i in range(epoch):
                
            cache, pred =self.forward(X_train)

            cost =self.cost_calc(loss='CCE', A=pred, Y=Y_train, m=self.num_inputs)

            self.backward(Y_train, pred, cache, alpha=learning_rate)
            J_hist.append(cost)

            if dev:
                c, dev_pred = self.forward(X_dev)
                dev_cost = self.cost_calc(loss='CCE', A=dev_pred, Y=Y_dev, m=dev_input_size)
                dev_hist.append(dev_cost)
                eval_percentage = self.evaluate(X_dev, Y_dev)
            if (i == 0) or (i % 100 == 0) or (i == epoch):
                print(f'epoch {i}: train cost {cost:.4f} | dev cost {dev_cost:.4f} | eval {eval_percentage:.4f}')
        if dev:
            return J_hist, dev_hist
        else:
            return J_hist

    def predict(self, X):
        cache, pred = self.forward(X)
        return np.where(pred > 0.5, 1 ,0)
        
    def evaluate(self, X, Y):
        pred = self.predict(X)
        total = np.sum(pred == Y)
        eval_percentage = total/self.num_inputs*100
        return eval_percentage









    

                






          
