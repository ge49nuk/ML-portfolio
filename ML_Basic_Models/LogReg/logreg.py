import numpy as np
import math

class LogReg:
    def __init__(self, lr=0.001,it = 1000):
        self.lr = lr
        self.it = it

    def fit(self, X, y):
        # Init weights and bias
        self.w = [0]*len(X[0])
        self.b = 0
        for _ in range(self.it):
            lin_model = np.dot(X,self.w) + self.b
            predicted = self._sigmoid(lin_model)
            
            # Calculate derivatives
            dw = predicted - y
            dw = np.dot(dw,X)
            db = predicted - y
            db = np.sum(db)

            # Normalize
            dw/=len(X)
            db/=len(X)
            
            # Update
            self.w-=self.lr*dw
            self.b-=self.lr*db




    def predict(self, X):
        lin_mod =  np.dot(X, self.w) + self.b
        return [1 if i>=0.5 else 0 for i in lin_mod]
    
    def _sigmoid(self, X):
        return 1/(1+np.exp(-X))
