'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    # TODO
    entropy = 0
    labelCol = df[[df.columns[-1]]].values
    labels,freq = np.unique(labelCol,return_counts=True)
    s = np.sum(freq)
    for i in freq:
        p = i/s
        if p != 0:
            entropy -= p*(np.log2(p))
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO
    avg_info=0
    l=np.unique(df[attribute].values)
    #print(l)
    for i in l:
        temp=df[df[attribute]==i]
        j=temp[[temp.columns[-1]]].values
        #print(j)
        v,f=np.unique(j,return_counts=True)
        s=np.sum(f)
        #print(v,f)
        e=0
        for k in f:
            prob=k/s
            if prob!=0:
                e-=prob*np.log2(prob)
        avg_info+=e*(np.sum(f)/df.shape[0])
    return (abs(avg_info))


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO
    information_gain=get_entropy_of_dataset(df)-get_avg_info_of_attribute(df,attribute)
    return information_gain




#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    d={}
    for a in df.columns[:-1]:
        d[a]=get_information_gain(df,a)
    #print(d)
    return (d,(max(d, key=d.get)))

