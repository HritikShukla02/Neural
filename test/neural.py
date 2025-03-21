# import tensorflow as tf
import numpy as np

class Neural:
    def __init__(self,input_size:int, layers_sizes:list, activations:list, cost:str):
        
        self.layers = layers_sizes
        self.acts = activations
        self.cost = cost
        self.input_size = input_size
        
        # self.params = self.parameters()
        self.params = self.parameters(self.input_size)



    def parameters(self, input_size):
        params = {}
        for i in range(len(self.layers)):
            if i == 0:
                params[f'W{i}'] = np.random.randn(self.layers[i], input_size)*np.sqrt(2/input_size)*0.01
            else:
                params[f'W{i}'] = np.random.randn(self.layers[i], self.layers[i-1])*np.sqrt(2/self.layers[i-1])*0.01

            params[f'b{i}'] =np.zeros((self.layers[i], 1))

        return params
    

    def activation(self, z, act):
        z = np.clip(z, -500, 500)  # Prevents overflow
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
            c = {"W":W, 
                 "Aprev":A, 
                 "b":b,
                  "Z": Z}# A here is actually Aprev
            A = self.activation(Z, act=self.acts[i])
            c["A"] = A# A here is Acurr
            cache.append(c)
        return cache, A
    

    def cost_calc(self, loss, A, Y):
        m = Y.shape[1]
        if loss=="CCE":
            if self.acts[-1] == 'sigmoid':
                cost = -np.sum(Y*(np.log(A + 1e-8))+(1-Y)*(np.log(1-A + 1e-8)))/(2*m)
            elif self.acts[-1] == 'softmax':
                 # Avoid log(0) by adding a small epsilon
                epsilon = 1e-9
                A_clipped = np.clip(A, epsilon, 1 - epsilon)  

                cost = -np.sum(Y * np.log(A_clipped)) / (m)
        
        if self.reg_lambda is not None:        
            # Compute L2 Regularization Term
            L2_reg = 0
            for i in range(len(self.layers)):  # Sum over all layers
                W = self.params[f'W{i}']
                L2_reg += np.sum(np.square(W))

            L2_reg = (self.reg_lambda / (2 * m)) * L2_reg  # Apply L2 factor

            return cost + L2_reg  # Add regularization term to cost
        else:
            return cost
    
    def clip(self, mat, clip_val):
        grads = np.clip(mat, -clip_val, clip_val)
        return grads

    def backprop(self, Y, pred, cache, optimizer="gd", alpha=0.01, **kwargs ):
        m = Y.shape[1]
        dA=None
        for i in reversed(range(len(self.layers))):
            c =cache[i]
            W = c["W"]
            Aprev = c["Aprev"]
            Z = c["Z"]
            A = c["A"]

            if i == len(self.layers)-1:
                dZ = pred - Y
            else:
                if self.acts[i] == 'Leaky-ReLU':
                    g = np.where(Z>=0,1, 0.1)
                    
                elif self.acts[i] == 'ReLU':
                    g = np.where(Z>=0,1, 0)
                elif self.acts[i] == 'sigmoid':
                    g = A*(1-A)
                dZ = dA * g
                
           
            if self.reg_lambda is not None:
                dW = (np.dot(dZ, Aprev.T) + self.reg_lambda*W)/ m

            else:
                dW = np.dot(dZ, Aprev.T)/ m

            db = np.sum(dZ, axis=1 ,keepdims=True)/m

            dA = np.dot(W.T, dZ)

            if optimizer == "gd":
                self.params[f'W{i}'] = self.clip(self.params[f'W{i}'] - alpha*dW, 20)
                self.params[f'b{i}'] = self.clip(self.params[f'b{i}'] - alpha*db, 10)
            
            elif optimizer == "momentum":
                beta=kwargs.get('beta')
                if beta is None:
                    beta =0.9
                V= kwargs.get("update")

                Vdw = V[f"Vdw{i}"]
                Vdb = V[f"Vdb{i}"]

                
                Vdw = beta*Vdw + (1-beta)*dW
                Vdb = beta*Vdb + (1-beta)*db
                self.params[f'W{i}'] = self.clip(self.params[f'W{i}'] - alpha*Vdw, 20)
                self.params[f'b{i}'] = self.clip(self.params[f'b{i}'] - alpha*Vdb, 10)

                V[f"Vdw{i}"] = Vdw
                V[f"Vdb{i}"] = Vdb
                
                update = V
            
            elif optimizer == 'RMSprop':
                beta=kwargs.get('beta')
                if beta is None:
                    beta =0.999
                S= kwargs.get("update")

                Sdw = S[f"Sdw{i}"]
                Sdb = S[f"Sdb{i}"]
                e = 1e-8

                Sdw = beta*Sdw + (1-beta)*(dW**2)
                Sdb = beta*Sdb + (1-beta)*(db**2)
                self.params[f'W{i}'] = self.clip(self.params[f'W{i}'] - alpha*dW/np.sqrt(Sdw + e), 20)
                self.params[f'b{i}'] = self.clip(self.params[f'b{i}'] - alpha*db/np.sqrt(Sdb + e), 10)

                S[f"Sdw{i}"] = Sdw
                S[f"Sdb{i}"] = Sdb
                
                update = S
            elif optimizer == "Adam":
                
                beta1=kwargs.get('beta1')
                if beta1 is None:
                    beta1 =0.9

                beta2=kwargs.get('beta2')
                if beta2 is None:
                    beta2 =0.999
                
                t=kwargs.get('t')
                update = kwargs.get('update')
                S= update[0]
                V= update[1]



                Sdw = S[f"Sdw{i}"]
                Sdb = S[f"Sdb{i}"]
                Vdw = V[f"Vdw{i}"]
                Vdb = V[f"Vdb{i}"]
                e = 1e-8

                Vdw = beta1*Vdw + (1-beta1)*dW
                Vdb = beta1*Vdb + (1-beta1)*db

                Vdw /= (1-np.power(beta1,t+1))
                Vdb /= (1-np.power(beta1,t+1))
    
                Sdw = beta2*Sdw + (1-beta2)*(dW**2)
                Sdb = beta2*Sdb + (1-beta2)*(db**2)

                self.params[f'W{i}'] = self.clip(self.params[f'W{i}'] - alpha*Vdw/np.sqrt(Sdw + e), 20)
                self.params[f'b{i}'] = self.clip(self.params[f'b{i}'] - alpha*Vdb/np.sqrt(Sdb + e), 10)

                S[f"Sdw{i}"] = Sdw
                S[f"Sdb{i}"] = Sdb
                V[f"Vdw{i}"] = Vdw
                V[f"Vdb{i}"] = Vdb
                update = S, V
        if optimizer != "gd":
            return update


    def batches_generator(self, X, Y, seed=0, batch_size=None):
        num_samples = Y.shape[1]

        # Determine batch size and number of batches
        if batch_size is None:
            batch_size = num_samples

        num_batches = num_samples // batch_size
        if num_samples % batch_size != 0:
            num_batches += 1  # Add an extra batch for remaining samples

        # At the beginning of each epoch
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(num_samples)
        X_train = X[:, shuffle_indices]
        Y_train = Y[:, shuffle_indices]
        batches = []

        for j in range(num_batches):

            start_idx = j * batch_size
            end_idx = min((j + 1) * batch_size, num_samples)

        
            x_batch = X_train[:, start_idx:end_idx]
            y_batch = Y_train[:, start_idx:end_idx]

            batch = (x_batch, y_batch)
            batches.append(batch)
        
        return batches
    

    # GRADED FUNCTION: update_lr

    def update_lr(self, learning_rate0, epoch_num, decay_rate):
        """
        Calculates updated the learning rate using exponential weight decay.
        
        Arguments:
        learning_rate0 -- Original learning rate. Scalar
        epoch_num -- Epoch number. Integer
        decay_rate -- Decay rate. Scalar

        Returns:
        learning_rate -- Updated learning rate. Scalar 
        """
        #(approx. 1 line)
        # learning_rate = 
        # YOUR CODE STARTS HERE
        learning_rate = learning_rate0/(1+decay_rate*epoch_num)
        
        # YOUR CODE ENDS HERE
        return learning_rate
    

    def train(self, X_train, Y_train, epoch, learning_rate=0.01, optimizer="gd", batch_size=None, reg_lambda=None, X_dev=None, Y_dev=None, patience=None):

        self.reg_lambda = reg_lambda
        dev = False

        if X_dev is not None:
            dev_input_size = X_dev.shape[1]
            dev = True
        
        J_hist = []
        dev_hist = []
        dev_eval_hist =[]
        train_eval_hist = []

        if( optimizer == "momentum") or ( optimizer == "Adam"):
            V = {}
            for i in range(len(self.layers)):
                V[f'Vdw{i}'] = np.zeros(self.params[f'W{i}'].shape)
                V[f'Vdb{i}'] = np.zeros(self.params[f'b{i}'].shape)

        if (optimizer == "RMSprop" or(optimizer == "Adam")):
            S = {}
            for i in range(len(self.layers)):
                S[f'Sdw{i}'] = np.zeros(self.params[f'W{i}'].shape)
                S[f'Sdb{i}'] = np.zeros(self.params[f'b{i}'].shape)
        t=0
        seed = 10
        curr_lr = learning_rate
        train_pat = 5000
        if patience is not None:
            best_val_loss = np.inf
            patience_counter = 0

        for i in range(epoch):
            seed += 1
            batches = self.batches_generator(X_train, Y_train, batch_size=batch_size, seed=seed)
            num_batches = len(batches)
                
            cost=0
            for batch in batches:
                x_batch = batch[0]
                y_batch = batch[1]
            
                
                cache, pred =self.forward(x_batch)

                cost += self.cost_calc(loss='CCE', A=pred, Y=y_batch)

                if optimizer == "gd":
                    self.backprop(y_batch, pred, cache, alpha=curr_lr, optimizer=optimizer)
                elif optimizer == "momentum":
                    V = self.backprop(y_batch, pred, cache, alpha=curr_lr, optimizer=optimizer, update=V)
                elif optimizer == "RMSprop":
                    S = self.backprop(y_batch, pred, cache, alpha=curr_lr, optimizer=optimizer, update=S)
                elif optimizer == "Adam":
                    t += 1
                    S, V = self.backprop(y_batch, pred, cache, alpha=curr_lr, optimizer=optimizer, t=t, update=(S,V))

            cost /=num_batches

            curr_lr = self.update_lr(learning_rate, i, decay_rate=1 )
            
            if dev:
                c, dev_pred = self.forward(X_dev)
                dev_cost = self.cost_calc(loss='CCE', A=dev_pred, Y=Y_dev)
                dev_eval_percentage = self.evaluate(X_dev, Y_dev)

                

            train_eval = self.evaluate(x_batch, y_batch)
            train_eval_hist.append(train_eval)
            dev_hist.append(dev_cost)
            dev_eval_hist.append(dev_eval_percentage)
            J_hist.append(cost)

            if (i == 0) or (i % 100 == 0) or (i == epoch):
                if dev:
                    print(f'epoch {i}: train cost {cost:.8f} | dev cost {dev_cost:.8f} | train eval {train_eval:.8f} | dev eval {dev_eval_percentage:.8f}')
                else:
                    print(f'epoch {i}: train cost {cost:.8f} | train eval {train_eval:.8f}')

            if patience is not None:
                if (dev_cost < best_val_loss) and (i > train_pat):
                    best_val_loss = dev_cost
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Stop training if patience limit is reached
            if (i >train_pat) and (patience_counter >= patience):
                print(f"Early stopping at epoch {i+1}")
                break

        if dev:
            return J_hist, dev_hist, train_eval_hist, dev_eval_hist 
        else:
            return J_hist

    def predict(self, X):
        _, pred = self.forward(X)
        if self.acts[-1] == "softmax":
            return np.argmax(pred, axis=0)  # Multi-class prediction
        return np.where(pred > 0.5, 1, 0)  # Binary classification

        
    def evaluate(self, X, Y):
        pred = self.predict(X)
        total = np.sum(pred == Y)
        eval_percentage = total/X.shape[1]*100
        return eval_percentage









    

                






          
