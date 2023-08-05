import numpy as np

class Reg:
    def __init__(self, lr = 0.001, it = 1000):
        self.lr = lr
        self.it = it
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        for _ in range(self.it):
            dw = 0
            db = 0
            for i in range(len(X)):
                x=X[i]
                # Calculate derivatives
                dw+=2*x*(self.w*x+self.b - y[i])
                db+=2*(self.w*x+self.b - y[i])
            # Normalize
            dw/=len(X)
            db/=len(X)
            # Update
            self.w = self.w-self.lr*dw
            self.b = self.b-self.lr*db
            


    def predict(self,X):
        return np.array([self.w*xi + self.b for xi in X])
