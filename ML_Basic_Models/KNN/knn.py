import math
import numpy as np

class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        #returned predicted labels
        yOut = [-1]*len(X)

        #iterate through data to be labeled
        for i1 in range(len(X)):
            closestPoints = {}
            xi = X[i1]

            #go through train data
            for i in range(len(self.X)):
                x = self.X[i]
                dist = np.linalg.norm(x-xi)

                #For the first k points just add them
                if len(closestPoints.keys()) < self.k:
                    closestPoints[i] = dist
                    closestPoints = dict(sorted(closestPoints.items(), key=lambda x: x[1]))
                    continue

                #else look if new point is closer than the current k closest points
                furthest = list(closestPoints.values())[-1]
                if(dist<furthest):
                    closestPoints.popitem()
                    closestPoints[i] = dist
                    closestPoints = dict(sorted(closestPoints.items(), key=lambda x: x[1]))

            votes = [0]*self.k
            for i in closestPoints.keys():
                votes[self.y[i]]+=1

            max_value = max(votes)
            yOut[i1] = votes.index(max_value)
            

        return yOut
        




   