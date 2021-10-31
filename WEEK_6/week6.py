from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import *
import pandas as pd
import numpy as np

class SVM:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        data = pd.read_csv(self.dataset_path)

        # X-> Contains the features
        self.X = data.iloc[:, 0:-1]
        # y-> Contains all the targets
        self.y = data.iloc[:, -1]

    def solve(self):
        """
        Build an SVM model and fit on the training data
        The data has already been loaded in from the dataset_path

        Refrain to using SVC only (with any kernel of your choice)

        You are free to use any any pre-processing you wish to use
        Note: Use sklearn Pipeline to add the pre-processing as a step in the model pipeline
        Refrain to using sklearn Pipeline only not any other custom Pipeline if you are adding preprocessing

        Returns:
            Return the model itself or the pipeline(if using preprocessing)
        """

        # TODO
        pipeline = Pipeline([('normalizer', StandardScaler()),('svc', SVC(gamma='auto'))])
        clf=make_pipeline(StandardScaler(),SVC(gamma='auto'))
        models=[]
        pipeline.fit(self.X,self.y)
        models.append(pipeline)
        score=clf.predict
        # p2=Pipeline([('normalizer', StandardScaler()),('svc', SVC(gamma='auto',kernel='linear'))])
        # p2.fit(self.X,self.y)
        # models.append(p2)
        p3=Pipeline([('normalizer', StandardScaler()),('svc', SVC(gamma='auto',kernel='poly',degree=20))])
        p3.fit(self.X,self.y)
        models.append(p3)
        p4=Pipeline([('normalizer', StandardScaler()),('svc', SVC(gamma='auto',kernel='rbf'))])
        p4.fit(self.X,self.y)
        models.append(p4)
        # p5=Pipeline([('normalizer', StandardScaler()),('svc', SVC(gamma='auto',kernel='sigmoid'))])
        # p5.fit(self.X,self.y)
        # models.append(p5)
        print('polyd3')
        mym=[]
        l=[0.001,0.01,0.1,1,10,100]
        for i in l:
            for j in l:
                p=Pipeline([('normalizer', StandardScaler()),('svc', SVC(gamma=i,C=j,kernel='rbf'))])
                p.fit(self.X,self.y)
                mym.append(p)
        return mym
            
