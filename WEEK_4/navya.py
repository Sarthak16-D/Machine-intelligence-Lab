from typing import final
import numpy as np


class KNN:

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        # TODO
        fin_dist = []
        p = self.p
        for i in self.data:
            for j in range(len(x)):
                fin_dist.append(minkowski_dist(i, x[j], p))

        fin_dist = np.array(fin_dist)

        fin_dist = fin_dist.reshape(len(self.data), len(x))
        fin_dist = fin_dist.transpose()
        fin_dist = fin_dist.tolist()
        return fin_dist

    def k_neighbours(self, x):

        dist = self.find_distance(x)
        min_dist = []
        ind = []
        mini = []

        for i in dist:
            s = sorted(i)
            mini = s[0:self.k_neigh]
            for j in mini:
                ind.append(i.index(j))
            min_dist.append(mini)

        ind = np.array(ind)
        ind = ind.reshape(len(dist), self.k_neigh)
        final_ind = ind.tolist()

        return([min_dist, final_ind])

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        l = self.k_neighbours(x)
        l1 = self.find_distance(x)

        retList = []

        if(self.weighted == True):
            tarDict = {}
            for i in range(len(x)):
                tarDict = {}

                for j in self.target:
                    if j not in tarDict:
                        tarDict[j] = 0

                w = []
                q = []
                for j in range(len(l[0][i])):
                    q.append(l[1][i][j])
                    p = l[0][i][j]
                    if(p==0):
                    	w.append(1/(p+0.000000001))
                    	continue
                    w.append(1/(p))

                maxval = 0
                maxpos = 0

                for j in range(len(w)):
                    tarDict[self.target[q[j]]
                            ] = tarDict[self.target[q[j]]]+w[j]

                for j in tarDict:
                    if(tarDict[j] > maxval):
                        maxval = tarDict[j]
                        maxpos = j
                # print(tarDict)
                retList.append(maxpos)
            #print(retList)
            return retList

        else:
            indices = self.k_neighbours(x)[1]
            r = []
            for i in range(len(indices)):
                f = {}
                for j in range(len(indices[i])):
                    if self.target[indices[i][j]] in f:
                        f[self.target[indices[i][j]]] += 1
                    else:
                        f[self.target[indices[i][j]]] = 1
                maxF = 0
                maxK = None
                for i in range(min(f), max(f)+1):
                    if f[i] > maxF:
                        maxF = f[i]
                        maxK = i
                r.append(maxK)
        return r

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        l = self.predict(x)
        correct = 0

        for i in range(len(l)):
            if l[i] == y[i]:
                correct += 1

        ret = (correct*100/len(l))
        return ret


def minkowski_dist(x, y, p):
    # print("x,y", x, y)
    dist = 0
    for i in range(len(x)):
        dist += (abs(x[i]-y[i]))**p
    return (dist**(1/p))

