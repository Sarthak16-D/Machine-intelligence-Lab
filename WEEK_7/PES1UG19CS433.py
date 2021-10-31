import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math
"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED 

class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def stump_error(self, y, y_pred, sample_weights):
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """
        temp = np.where(y_pred!=y,1,0)
        err = np.sum(sample_weights*temp)/np.sum(sample_weights)
        return err

        # TODO
        

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        eps = 1e-9
        # TODO
        alpha=(0.5)*math.log((1-error + eps )/(error + eps))
        return alpha

    def update_weights(self, y, y_pred, sample_weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """

        # TODO
        sterr=self.stump_error( y, y_pred, sample_weights)
        alpha=self.compute_alpha(sterr)
        # print(alpha)
        new_sample_weights=sample_weights
        for i in range(0,len(y)):
            if(y[i]==y_pred[i]):
                new_sample_weights[i]=(sample_weights[i])*math.exp(-alpha)
            else:
                new_sample_weights[i]=(sample_weights[i])*math.exp(alpha)
        # print(new_sample_weights)
        new_sample_weights=[ i/sum(new_sample_weights) for i in new_sample_weights]
        return new_sample_weights
        

    def predict(self, X):
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        # TODO
        spl = np.array([self.stumps[stump].predict(X) for stump in range(self.n_stumps)])
        N=spl[0]
        #print(spl)
        # for i in range(len(spl[0])):
        #     if(spl[0][i]<0):
        #         N[i]=-1
        #     elif(spl[0][i]>0):
        #         N[i]=1
        #     else:
        #         N[i]=0
        # return N
        return np.sign(spl[0])

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        pred = self.predict(X)
        # find correct predictions
        correct = (pred == y)

        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy