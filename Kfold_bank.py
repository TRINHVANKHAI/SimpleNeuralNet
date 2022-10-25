from sklearn.datasets import make_classification
from sklearn.model_selection import KFold,StratifiedKFold,RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep
from scipy.special import expit, logit
import numpy as np
import pandas as pd
import json
from json import JSONEncoder
from timeit import default_timer as timer


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
Back propagation derivative of weight 
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
nextDLA.shape     =   ( nex_K, 1 )
nextA.shape       =   ( nex_K, 1 )
nextW.shape       =   ( nex_K, cur_K )
DLA.shape         =   ( cur_K, 1 )
-----------------------------------------
return = ( cur_K, 1 )

"""
def DerivativeBackPropagation_DLA(nextDLA, nextA, nextW):
  dnextAtoA = nextA*(1-nextA)*nextW
  DLA = np.dot(dnextAtoA.T, nextDLA)
  return DLA





"""
-----------------------------------------
Back propagation derivative of weight 
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
curA.shape       =   ( cur_K, 1 )
preA.T           =   ( 1, pre_K )
curDLA.shape     =   ( cur_K, 1 )
DLW.shape        =   ( cur_K, pre_K ) 
-----------------------------------------
return = ( cur_K, pre_K )

"""
def DerivativeBackPropagation_DLW(curDLA, curA, preA):
  dcurAtoW = np.kron( curA*(1-curA), preA.T ) #Dot?
  DLW = curDLA * dcurAtoW
  return DLW



"""
-----------------------------------------
Back propagation derivative of bias 
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
curDLA.shape     =   ( cur_K, 1 )
curA.shape       =   ( cur_K, 1 )
DLB.shape        =   ( cur_K, 1 ) 
-----------------------------------------
return = ( cur_K, 1 )

"""
def DerivativeBackPropagation_DLB(curDLA, curA):
  dcurAtoB = curA * (1 - curA)  #Hadamard broadcasting
  DLB = curDLA * dcurAtoB
  return DLB

"""
-----------------------------------------
Feed forward progress 
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
curB.shape     =   ( cur_K, 1 )
curW.shape     =   ( cur_K, pre_K )
preA.shape     =   ( pre_K, 1 )

-----------------------------------------
return = ( cur_K, 1 )

"""
def NN_FeedForward(preA, curB, curW):
  pre_K = preA.shape[1]
  Ones  = np.ones( (pre_K, 1) )
  BW    = np.hstack( (curB, curW) )
  AO    = np.vstack( (Ones, preA) )
  z = np.dot(BW, AO)
  S = expit( z )
  return S





#############################################################INIT MODEL



class NumpyArrayEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(NumpyArrayEncoder, self).default(obj)
      
with open("tree_fixed_write.json", "r") as read_file:

    model = json.load(read_file)

    configuration = model['configurations']
    NumberOfFeatures = configuration['NumberOfFeatures']
    NumberOfLayers = configuration['NumberOfLayers']
    Layers = configuration['Layers']
    
    print('\n\n==================================================')    
    print('Model : %s' %(model['model']))
    print('ID    : %s' %(model['id']))
    print('--------------------------------------------')
    print('\n>  Sumary infomations : ')
    print('---------------------')
    print('    Accuracy score : %s' %(model['score']))
    
    print('\n>  Configurations : ')
    print('---------------------')
    print('   Number of features: %d' %(NumberOfFeatures))
    print('   Number of layers  : %d' %(NumberOfLayers))
    print('   Number of epochs  : %d' %(configuration['NumberOfEpochs']))
    print('   Learning rate  : %d' %(configuration['LearningRate']))
    
    print('\n>  Details : ')
    print('---------------------')
    print('    -> -> -> ->    -> -> -> ->   -> -> -> ->')


    for layerId in range(1, NumberOfLayers+1):
       print('\n\n    -------------------------------------------')
       layerName = 'layer@'+str(layerId)

       layerInfo = Layers[ layerName ]
       print('    %s' %(layerInfo['Name']))
       print('    -------------')
       print('        Number of nodes: %d' %(layerInfo['NumberOfNodes']))
       Weight = np.asarray(layerInfo['Weight'])
       Bias   = np.asarray(layerInfo['Bias'])
       print('Weight=\n', Weight)
       print('Bias=\n', Bias)

    print('\n\n    <- <- <- <-    <- <- <- <-    <- <- <- <-')

    read_file.close()




filename = "bank-full.csv"
column_names = [ \
"age", "job", "marital", "education", "default", \
"balance", "housing", "loan", "contact", "day",  \
"month","duration","campaign","pdays","previous","poutcome", \
"y"] 

df        = pd.read_csv(filename, names=column_names, sep=';', header=0)

count_class_no, count_class_yes = df.y.value_counts()
df_class_no = df[ df['y'] == 'no' ]
df_class_yes = df[ df['y'] == 'yes' ]
df_class_yes_over = df_class_yes.sample(count_class_no, replace=True)
#Over sample 1?
df_resampled = pd.concat([df_class_no, df_class_yes_over], axis=0)
#df_resampled = pd.concat([df_class_no, df_class_yes], axis=0)
df_resampled = df_resampled.sample(frac=1).reset_index()

age       = df_resampled[ column_names[0] ].to_numpy().reshape(-1,1)
job       = df_resampled[ column_names[1] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
marital   = df_resampled[ column_names[2] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
education = df_resampled[ column_names[3] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
default   = df_resampled[ column_names[4] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
balance   = df_resampled[ column_names[5] ].to_numpy().reshape(-1,1)
housing   = df_resampled[ column_names[6] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
loan      = df_resampled[ column_names[7] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
contact   = df_resampled[ column_names[8] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
day       = df_resampled[ column_names[9] ].to_numpy().reshape(-1,1)
month     = df_resampled[ column_names[10] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
duration  = df_resampled[ column_names[11] ].to_numpy().reshape(-1,1)
campaign  = df_resampled[ column_names[12] ].to_numpy().reshape(-1,1)
pdays     = df_resampled[ column_names[13] ].to_numpy().reshape(-1,1)
previous  = df_resampled[ column_names[14] ].to_numpy().reshape(-1,1)
poutcome  = df_resampled[ column_names[15] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
y         = df_resampled[ column_names[16] ].astype('category').cat.codes.to_numpy().reshape(-1,1)

Features = np.hstack((age, job, marital,education, default, balance,housing,loan,contact,day, month, duration,campaign,pdays,previous, poutcome))
count_class_no, count_class_yes = df.y.value_counts()





#########################################################################

lRate = configuration['LearningRate']
config_epochs = configuration['NumberOfEpochs']
node_num_l0 = Features.shape[1] 
node_num_l1 = Layers['layer@1']['NumberOfNodes']
node_num_l2 = Layers['layer@2']['NumberOfNodes']
node_num_l3 = Layers['layer@3']['NumberOfNodes']
node_num_l4 = y.shape[1]


W1 = Layers['layer@1']['Weight']
W2 = Layers['layer@2']['Weight']
W3 = Layers['layer@3']['Weight']
W4 = Layers['layer@4']['Weight']
B1 = Layers['layer@1']['Bias']
B2 = Layers['layer@2']['Bias']
B3 = Layers['layer@3']['Bias']
B4 = Layers['layer@4']['Bias']
"""
B1 = np.random.uniform(-2,2, size=(node_num_l1,1))
W1 = np.random.uniform(-2,2, size=( node_num_l1,node_num_l0))

B2 = np.random.uniform(-2,2, size=( node_num_l2,1))
W2 = np.random.uniform(-2,2, size=( node_num_l2,node_num_l1))


B3 = np.random.uniform(-2,2, size=( node_num_l3,1))
W3 = np.random.uniform(-2,2, size=( node_num_l3,node_num_l2))


B4 = np.random.uniform(-2,2, size=( node_num_l4,1))
W4 = np.random.uniform(-2,2, size=( node_num_l4,node_num_l3))
"""
print('>>B1=\n', B1)
print('>>W1=\n', W1)
print('>>B2=\n', B2)
print('>>W2=\n', W2)
print('>>B3=\n', B3)
print('>>W3=\n', W3)
print('>>B4=\n', B4)
print('>>W4=\n', W4)





print('\n\n==============   TRAINING DATA   ==============')
#Stochastics Gradient Descent
#Return rows number : number of samples
scaler = prep.MinMaxScaler()
scaledFeatures = scaler.fit_transform(Features)

#kf = KFold(10)
kf = StratifiedKFold(10)
count = 0
TotalN = Features.shape[0]
Yh     = np.random.rand(y.shape[0], y.shape[1])
test_passed=0

for epoch in range (2):
    count = count+1
    print('Round : %d ' %(count))

    for train_index, test_index in kf.split(Features,y):
        #Training process
        startTime = timer()
        for pIDX in train_index:
            Xinput = scaledFeatures[pIDX,:].reshape(-1,1)
            Yinput = y[pIDX,:].reshape(-1,1)

            #FEED FORWARD
            A1 = NN_FeedForward(Xinput, B1, W1)
            A2 = NN_FeedForward(A1, B2, W2)
            A3 = NN_FeedForward(A2, B3, W3)
            A4 = NN_FeedForward(A3, B4, W4)
 
            #BACK PRO
            DLA4 = ((A4 - Yinput)/(A4*(1-A4)))
            W4 = W4 - lRate * DerivativeBackPropagation_DLW(DLA4, A4, A3)
            B4 = B4 - lRate * DerivativeBackPropagation_DLB(DLA4, A4)

            DLA3 = DerivativeBackPropagation_DLA(DLA4, A4, W4)
            W3 = W3 - lRate * DerivativeBackPropagation_DLW(DLA3, A3, A2)
            B3 = B3 - lRate * DerivativeBackPropagation_DLB(DLA3, A3)

            DLA2 = DerivativeBackPropagation_DLA(DLA3, A3, W3)
            W2 = W2 - lRate * DerivativeBackPropagation_DLW(DLA2, A2, A1)
            B2 = B2 - lRate * DerivativeBackPropagation_DLB(DLA2, A2)

            DLA1 = DerivativeBackPropagation_DLA(DLA2, A2, W2)
            W1 = W1 - lRate * DerivativeBackPropagation_DLW(DLA1, A1, Xinput)
            B1 = B1 - lRate * DerivativeBackPropagation_DLB(DLA1, A1)
            
        #Testing process
        for pIDX in test_index:
            Xinput = scaledFeatures[pIDX,:].reshape(-1,1)
            Yinput = y[pIDX,:].reshape(-1,1)
            A1T = NN_FeedForward(Xinput, B1, W1)
            A2T = NN_FeedForward(A1T, B2, W2)
            A3T = NN_FeedForward(A2T, B3, W3)
            A4T = NN_FeedForward(A3T, B4, W4)
            if A4T[0] > 0.5:
                Yh[pIDX,0] = 1
            else:
                Yh[pIDX,0] = 0

        tacount = 0
        ttcount = 0
        tfcount = 0
        tzcount = 0
        tocount = 0

        for pIDX in test_index:
            if (y[pIDX,0] == 1):
                tocount = tocount+1
            else:
                tzcount = tzcount+1

            if (y[pIDX,0] == Yh[pIDX,0]):
                tacount = tacount + 1
                if (Yh[pIDX,0]==1):
                    ttcount = ttcount + 1
                else:
                    tfcount = tfcount + 1

        
        if((tacount/len(test_index)) > 0.9):
            print("Test passed")
            test_passed=1
            break;
        
     

        
        endTime = timer()
        print('Epoch time cost = %f'%(endTime - startTime))
        print('\n\n------------------------------------------------------------')
        print('Overal score       = %.4f%%   [ %d / %d ]\n' %( (tacount*100/len(test_index) ), tacount,len(test_index)  ) )
        print('------------------------------------------------------------')
        print('True values score  = %.4f%%   [ %d / %d ]\n' %( (ttcount*100/tocount), ttcount,tocount ) )
        print('False values score = %.4f%%   [ %d / %d ]\n' %( (tfcount*100/tzcount), tfcount,tzcount ) )
    if(test_passed==1):
        break
print('\n\n------------------------------------------------------------')
print('Overal score       = %.4f%%   [ %d / %d ]\n' %( (tacount*100/len(test_index) ), tacount,len(test_index)  ) )
print('------------------------------------------------------------')
print('True values score  = %.4f%%   [ %d / %d ]\n' %( (ttcount*100/tocount), ttcount,tocount ) )
print('False values score = %.4f%%   [ %d / %d ]\n' %( (tfcount*100/tzcount), tfcount,tzcount ) )




print('\n\n==============   WEIGHT RESULT   ==============')
print('>>  W1= \n', W1)
print('>>  B1= \n', B1)
print('------------------------------------------------')
print('>>  W2= \n', W2)
print('>>  B2= \n', B2)
print('------------------------------------------------')
print('>>  W3= \n', W3)
print('>>  B3= \n', B3)
print('------------------------------------------------')
print('>>  W4= \n', W4)
print('>>  B4= \n', B4)



print('\n\n==============   PREDICT RESULT   ==============')

age       = df[ column_names[0] ].to_numpy().reshape(-1,1)
job       = df[ column_names[1] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
marital   = df[ column_names[2] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
education = df[ column_names[3] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
default   = df[ column_names[4] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
balance   = df[ column_names[5] ].to_numpy().reshape(-1,1)
housing   = df[ column_names[6] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
loan      = df[ column_names[7] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
contact   = df[ column_names[8] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
day       = df[ column_names[9] ].to_numpy().reshape(-1,1)
month     = df[ column_names[10] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
duration  = df[ column_names[11] ].to_numpy().reshape(-1,1)
campaign  = df[ column_names[12] ].to_numpy().reshape(-1,1)
pdays     = df[ column_names[13] ].to_numpy().reshape(-1,1)
previous  = df[ column_names[14] ].to_numpy().reshape(-1,1)
poutcome  = df[ column_names[15] ].astype('category').cat.codes.to_numpy().reshape(-1,1)
y         = df[ column_names[16] ].astype('category').cat.codes.to_numpy().reshape(-1,1)


Features = np.hstack((age, job, marital,education, default, balance,housing,loan,contact,day, month, duration,campaign,pdays,previous, poutcome))


TotalN = Features.shape[0]
scaler = prep.MinMaxScaler()
scaledFeatures = scaler.fit_transform(Features)
Yh     = np.random.rand(TotalN, node_num_l4)
for idx in range (TotalN):
  Xi = scaledFeatures[idx,:].reshape(-1,1)
  A1 = NN_FeedForward(Xi, B1, W1)
  A2 = NN_FeedForward(A1, B2, W2)
  A3 = NN_FeedForward(A2, B3, W3)
  A4 = NN_FeedForward(A3, B4, W4)
  if A4[0] > 0.5:
    Yh[idx,0] = 1
  else:
    Yh[idx,0] = 0

tacount = 0
ttcount = 0
tfcount = 0
tzcount = 0
tocount = 0

for cmp in range(TotalN):
  if (y[cmp,0] == 1):
    tocount = tocount+1
  else:
    tzcount = tzcount+1

  if (y[cmp,0] == Yh[cmp,0]):
    tacount = tacount + 1
    if (Yh[cmp,0]==1):
      ttcount = ttcount + 1
    else:
      tfcount = tfcount + 1

print('\n\n------------------------------------------------------------')
print('Overal score       = %.4f%%   [ %d / %d ]\n' %( (tacount*100/TotalN ), tacount,TotalN  ) )
print('------------------------------------------------------------')
print('True values score  = %.4f%%   [ %d / %d ]\n' %( (ttcount*100/tocount), ttcount,tocount ) )
print('False values score = %.4f%%   [ %d / %d ]\n' %( (tfcount*100/tzcount), tfcount,tzcount ) )



#Update model
overalScore = tacount*100/TotalN
if overalScore > model['score']:
    with open("tree_fixed_write.json", "w") as write_file:
        configuration['NumberOfFeatures'] = node_num_l0
        Layers['layer@1']['Weight'] = W1
        Layers['layer@2']['Weight'] = W2
        Layers['layer@3']['Weight'] = W3
        Layers['layer@4']['Weight'] = W4
        Layers['layer@1']['Bias']   = B1
        Layers['layer@2']['Bias']   = B2
        Layers['layer@3']['Bias']   = B3
        Layers['layer@4']['Bias']   = B4
        model['score']              = overalScore
        modelUpdated = json.dumps(model, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)
        write_file.write(modelUpdated)
        print('Model weights has been updated')
        write_file.close()
else:
    print('Model weights update skipped')

"""
------------------------------------------------------------
Plot result, extract only data in range to show, or full set
------------------------------------------------------------
Bt.shape = (k+1, 1)
"""
Xax     = np.arange(100)  #0..99
Yax_org = y [2800:2900,:]
Yax_hat = Yh[2800:2900,:]



fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.plot(Xax, Yax_org, 'bo')
ax.plot(Xax, Yax_hat, 'y*')
ax.set_title('Predicted result yellow, expected blue')
ax.set_xlabel('Index')
ax.set_ylabel('y output')

plt.tight_layout()
plt.show()

