import matplotlib.pyplot as plt
from sklearn import preprocessing as prep
from scipy.special import expit, logit
import numpy as np
import pandas as pd

print("----------------------------------------------")
print("SOFT    : Logistics Regression")
print("VERSION : 1.0.0  2020/10/19")
print("AUTHOR  : TRINH VAN KHAI\n")
print("----------------------------------------------\n")

"""
-----------------------------------------
Abbreviated version of sigmoid function
Return vector of sigmoid values
k : Number of features
M : Number of samples
-----------------------------------------
w.T.shape     =   (1, k+1)
X.shape       =   (k+1, M)
z             =   np.dot(w.T,X)
expit( z )    :   1/[ 1 + np.exp( -z ) ]
-----------------------------------------
return = (1, M)

"""
def Sigmoid (w,X):
  z = np.dot(w.T,X)
  return expit( z )




"""
-----------------------------------------
Gradient function
Return the gradient of w for M samples
k : Number of features
M : Number of samples
-----------------------------------------
Btm.shape  = (k+1, 1)
Xm.shape   = (k+1, M)
ym.shape   = (1  , M)

-----------------------------------------
return (k+1, 1)
"""
def Gradient (Btm, Xm, ym):
  ym_hat = Sigmoid(Btm,Xm)
  Delta = ym_hat - ym
  return np.dot(Xm, Delta.T)


"""
-----------------------------------------
Gradient descent function
Return the final w vector
k : Number of features
N : Number of total samples in dataset
-----------------------------------------
Bt.shape  = (k+1, 1)  :  Beta
X.shape   = (k+1, N)  :  Features surface
y.shape   = (1  , N)  :  Expected outcome

-----------------------------------------
return (k+1, 1)
"""
def MiniBatchGradDescent(Bt,X,y, lr):
  Tt = Bt
  #Get dimensions and Size from features matrix
  d = X.shape[0]             # Get the dimensions, number of features
  N = X.shape[1]             # Get the size, number of examples in dataset
  Bsize = int(20)            # Determine the Mini batch size
  Dsize = int(N/Bsize)*Bsize 
  Dstep = int(Dsize/Bsize)
  Boff = np.arange(0,Dsize,Bsize,dtype=int)
  Vt = Tt
  for epoch in range(3000000):
    Told = Tt
    i = int(epoch%Dstep)

    #Mini batch extracted from Dataset
    Xm = X[ : , Boff[i] : Boff[i]+Bsize ] 
    ym = y[ : , Boff[i] : Boff[i]+Bsize ]
    #print("========= [ %d ] ==========" %i)
    #print(Xm)
    #print(ym)
    #print("=============================\n\n")
    #return Tt
    #Vt = 0.9*Vt + lr * Gradient ( (Tt - 0.9*Vt) ,Xm,ym)
    #Tt = Tt - Vt
    #print("Vt =", Vt)
    #print("Tt =", Tt)
    Tt = Tt - lr*Gradient(Tt, Xm, ym)
    if np.linalg.norm(Tt - Told) < 1e-15:
      print("Got result after %d iters" %epoch)
      print("Differ ")
      print(np.linalg.norm(Tt - Told))
      return Tt

  return Tt



"""
-----------------------------------------
Read dataset from csv file
k : Number of features (vd 16)
N : Number of total samples in dataset
-----------------------------------------
Bt.shape         = (k+1, 1)
Features.shape   = (k+1, N)
y.shape          = (1  , N)

-----------------------------------------

"""
filename = "bank.csv"
column_names = [ \
"age", "job", "marital", "education", "default", \
"balance", "housing", "loan", "contact", "day",  \
"month","duration","campaign","pdays","previous","poutcome", \
"y"] 

df        = pd.read_csv(filename, names=column_names, sep=';', header=0)
age       = df[[ column_names[0] ]].to_numpy().transpose()
job       = df[ column_names[1] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
marital   = df[ column_names[2] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
education = df[ column_names[3] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
default   = df[ column_names[4] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
balance   = df[[ column_names[5] ]].to_numpy().transpose()
housing   = df[ column_names[6] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
loan      = df[ column_names[7] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
contact   = df[ column_names[8] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
day       = df[[ column_names[9] ]].to_numpy().transpose()
month     = df[[ column_names[10] ]].to_numpy().transpose()
duration  = df[[ column_names[11] ]].to_numpy().transpose()
campaign  = df[[ column_names[12] ]].to_numpy().transpose()
pdays     = df[[ column_names[13] ]].to_numpy().transpose()
previous  = df[[ column_names[14] ]].to_numpy().transpose()
poutcome  = df[ column_names[15] ].astype('category').cat.codes.to_numpy().reshape(1,-1)
y         = df[ column_names[16] ].astype('category').cat.codes.to_numpy().reshape(1,-1)



"""
------------------------------------------------------------
Create feature surface
------------------------------------------------------------

Bt.shape       = (k+1, 1)
Features.shape = 
------------------------------------------------------------
"""
X = balance
Ones = np.ones( (1, X.shape[1]) )
Features = np.vstack((Ones,X))

"""
------------------------------------------------------------
Init Beta 
------------------------------------------------------------

Bt.shape    = (k+1, 1)
------------------------------------------------------------
"""
d = Features.shape[0]
Bt = np.random.randn(d,1)
print('Init Bt=\n', Bt)




"""
------------------------------------------------------------
Perform training
------------------------------------------------------------
Beta.shape = (k+1, 1)
"""
Beta = MiniBatchGradDescent(Bt, Features, y, 0.003333333)
print("Beta is : \n",Beta)



"""
------------------------------------------------------------
Valuate model, count number of fited data
------------------------------------------------------------
Beta.shape = (k+1, 1)
"""
Yres = Sigmoid(Beta,Features)
total = Yres.shape[1]
tcount = 0
for cmp in range(total):
  if (y[:,cmp] - Yres[:,cmp]) < 0.5 :
    tcount = tcount + 1 
print ('Total score = %f \n' %(tcount*100/total) )




"""
------------------------------------------------------------
Plot result, extract only data in range to show, or full set
------------------------------------------------------------
Bt.shape = (k+1, 1)
"""
Xdata = X[:,600:700]
Ydata = y[:,600:700]
Yhat  = Sigmoid(Beta,Features[:,600:700])



fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.plot(Xdata,Ydata, 'bo')
ax.plot(Xdata,Yhat, 'y*-')
ax.set_title('Scatter plot moi tuong quan giua Balance and y')
ax.set_xlabel('Balance')
ax.set_ylabel('Output')

plt.tight_layout()
plt.show()
