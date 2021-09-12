'''
import pandas as pd
df=pd.read_csv("example.csv")
print(df)

print(list(df.columns))
print(df['SL.No'])


#print(list(df.iloc[0]))
#print(list(df.iloc[:,:]))

print(list(df.iloc[:,1]))
print(list(df.iloc[1,:]))
print(list(df.iloc[1]))
#print 2ndlast col
print(list(df.iloc[:,-2]))


#print(df[0:2])

#print(df.mean(axis=0,skipna=True))
#print(df.mean(axis=0))
df.fillna 
'''
import numpy as np
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
	#note: shape is of the forst (x1,x2)
	#return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
	# where W1 is random matrix of shape shape1 with seed1
	# where W2 is random matrix of shape shape2 with seed2
	# where B is a random matrix of comaptible shape with seed3
	# if dimension mismatch occur return -1
	ans=None
	#TODO
	
	newshape1=(shape1[0],np.shape(X1**coef1)[1])
	newshape2=(shape2[0],np.shape(X2**coef2)[1])
	if(newshape1!=newshape2 or shape1[1]!=np.shape(X1**coef1)[0] or shape2[1]!=np.shape(X2**coef2)[0]): 
	        return -1
	np.random.seed(seed1)
	W1=np.random.rand(shape1[0],shape1[1])
	np.random.seed(seed2)
	W2=np.random.rand(shape2[0],shape2[1])
	t1=np.matmul(W1,X1**coef1)
	t2=np.matmul(W2,X2**coef2)
	np.random.seed(seed3)
	B=np.random.rand(newshape1[0],newshape1[1])
	ans=t1+t2+B
	return ans 
        
	        
X1=np.array([[1,2,3],[  4,5,6]])
X2=np.array([[4,5,6],[7,8,9]])
coef1=2
coef2=3
seed1=1
seed2=2
seed3=3

shape1=(3,2)
shape2=(3,2)
print(np.shape(X1))
print(np.shape(X2))
#print(f1(np.array([[1,2],[3,4]]),3,np.array([[1,2],[3,4]]),2,1,2,3,(3,2),(3,2)))
print(f1(np.array([[1,2],[3,4]]),3,np.array([[1,2],[3,4]]),2,1,2,3,(3,2),(3,2)).all()==np.array([[415.11116764, 604.9332781 ],[187.42695991 ,273.27266349],[112.57538713, 163.6775407 ]]).all())
