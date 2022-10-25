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
print("----------------------------------------------")
print("SOFT    : Neural Network Simplified Version   ")
print("VERSION : 1.0.0  2022/09/29                   ")
print("AUTHOR  : TRINH VAN KHAI                      ")
print("----------------------------------------------\n")

"""
-----------------------------------------
Abbreviated Sigmoid Derivative function
w.r.t biases
Return sigmoid derivative 
K : Number of features
M : Number of samples
-----------------------------------------
X.shape       =   (M, K)
-----------------------------------------
return = (M, K)

"""
def BackPropagationSigmoidDerivative (Xm):
  dSm = Xm*(1-Xm)
  return dSm



"""
-----------------------------------------
Back propagation derivative of output 
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
nextDLA.shape     =   ( M, nex_K )
nextA.shape       =   ( M, nex_K )
nextW.shape       =   ( cur_K, nex_K )
DLA.shape         =   ( M, cur_K )
-----------------------------------------
return = ( M, cur_K )

"""
def DerivativeBackPropagation_DLA(nextDLAm, nextAm, nextW):
  dSm = BackPropagationSigmoidDerivative (nextAm)
  DLAm = np.dot(nextDLAm*dSm, nextW.T)
  return DLAm





"""
-----------------------------------------
Back propagation derivative w.r.t weights
for minibatch, batch, stochastics
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
curA.shape       =   ( M, cur_K )
preA.shape       =   ( M, pre_K )
curDLA.shape     =   ( M, cur_K )
DLW.shape        =   ( pre_K, cur_K )
-----------------------------------------
return = ( pre_K, cur_K )

"""
def DerivativeBackPropagation_DLW( curDLAm, curAm, preAm ):
  DLW = np.dot( preAm.T, curDLAm * BackPropagationSigmoidDerivative (curAm) )
  return DLW



"""
-----------------------------------------
Back propagation derivative w.r.t biases 
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------
curDLA.shape     =   ( M, cur_K )
curA.shape       =   ( M, cur_K )
DLB.shape        =   ( 1, cur_K )
-----------------------------------------
return = ( 1, cur_K )

"""
def DerivativeBackPropagation_DLB( curDLAm, curAm ):
  DLB = np.sum( curDLAm * BackPropagationSigmoidDerivative (curAm) , axis=0 )
  return DLB

"""
-----------------------------------------
Feed forward progress 
M      : Batch size
cur_K  : Number of current layer's nodes
pre_K  : Number of prev layer's nodes
nex_K  : Number of next layer's nodes

-----------------------------------------

preA.shape     =   ( M, pre_K )
curW.shape     =   ( pre_K, cur_K )
curB.shape     =   ( 1, cur_K )

-----------------------------------------
return = ( M, cur_K )

"""
def NN_SigmoidFeedForward( preAm, curB, curW ):
  z = np.dot( preAm, curW ) + curB
  S = expit( z )
  return S




#############################################################MODEL TESTING
"""
M = 20
input_K = 5
output_K = 1
lRate = 0.005
Xinput = np.random.randn(M,input_K)
Yinput = np.random.randint(low=0, high=2 , size=(M, output_K))
WL1 = np.random.rand(input_K, output_K)
BL1 = np.random.rand(1,output_K)
AL1 = NN_SigmoidFeedForward(Xinput, BL1, WL1)
print('WL1  =\n', WL1)
print('BL1  =\n', BL1)
print('AL1  =\n', AL1)
AL1 = NN_SigmoidFeedForward(Xinput, BL1, WL1)
print('AL1  =\n', AL1)

for epoch in range(200):
    AL1 = NN_SigmoidFeedForward(Xinput, BL1, WL1)
    DLA1 = (AL1 - Yinput)/(AL1*(1-AL1))
    WL1 = WL1 - lRate * DerivativeBackPropagation_DLW(DLA1, AL1, Xinput)  #curDLAm, curAm, preAm
    BL1 = BL1 - lRate * DerivativeBackPropagation_DLB(DLA1, AL1)  #curDLAm, curAm


    #print('\n\n--------------------------------------------------------')
    #print('Output at epoch=%d'%(epoch))
    #print('AL1  =\n', AL1 )
    #print('DLA1 =\n', DLA1)
    #print('WL1  =\n', WL1)
    #print('BL1  =\n', BL1)
print('\n\n--------------------------------------------------------')
print("RESULT")
Yout = np.random.randint(low=0, high=2 , size=(M, output_K))
for idx in range (M):
  if AL1[idx] > 0.5:
    Yout[idx] = 1
  else:
    Yout[idx] = 0

print(Yinput.T)
print(Yout.T)
quit()

"""
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




filename = "bank.csv"
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



#print(Features.shape)
#quit()
#########################################################################

lRate = configuration['LearningRate']
config_epochs = configuration['NumberOfEpochs']
node_num_l0 = Features.shape[1] 
node_num_l1 = Layers['layer@1']['NumberOfNodes']
node_num_l2 = Layers['layer@2']['NumberOfNodes']
node_num_l3 = Layers['layer@3']['NumberOfNodes']
node_num_l4 = y.shape[1]


WL1 = np.asarray(Layers['layer@1']['Weight'])
WL2 = np.asarray(Layers['layer@2']['Weight'])
WL3 = np.asarray(Layers['layer@3']['Weight'])
WL4 = np.asarray(Layers['layer@4']['Weight'])
BL1 = np.asarray(Layers['layer@1']['Bias'])
BL2 = np.asarray(Layers['layer@2']['Bias'])
BL3 = np.asarray(Layers['layer@3']['Bias'])
BL4 = np.asarray(Layers['layer@4']['Bias'])


"""

BL1 = np.random.uniform(-2,2, size=( 1, node_num_l1))
WL1 = np.random.uniform(-2,2, size=( node_num_l0, node_num_l1))

BL2 = np.random.uniform(-2,2, size=( 1, node_num_l2))
WL2 = np.random.uniform(-2,2, size=( node_num_l1, node_num_l2))


BL3 = np.random.uniform(-2,2, size=( 1, node_num_l3))
WL3 = np.random.uniform(-2,2, size=( node_num_l2, node_num_l3))


BL4 = np.random.uniform(-2,2, size=( 1, node_num_l4))
WL4 = np.random.uniform(-2,2, size=( node_num_l3, node_num_l4))

"""


print('>>BL1:(%d,%d) =\n' %(BL1.shape[0],BL1.shape[1]), BL1)
print('>>WL1:(%d,%d) =\n' %(WL1.shape[0],WL1.shape[1]), WL1)
print('>>BL2:(%d,%d) =\n' %(BL2.shape[0],BL2.shape[1]), BL2)
print('>>WL2:(%d,%d) =\n' %(WL2.shape[0],WL2.shape[1]), WL2)
print('>>BL3:(%d,%d) =\n' %(BL3.shape[0],BL3.shape[1]), BL3)
print('>>WL3:(%d,%d) =\n' %(WL3.shape[0],WL3.shape[1]), WL3)
print('>>BL4:(%d,%d) =\n' %(BL4.shape[0],BL4.shape[1]), BL4)
print('>>WL4:(%d,%d) =\n' %(WL4.shape[0],WL4.shape[1]), WL4)




        


print('\n\n==============   TRAINING DATA   ==============')
i=0
TotalN      = Features.shape[0]
nFolds = 10
M = 200
scaler = prep.MinMaxScaler()
scaledFeatures = scaler.fit_transform(Features)

kf = StratifiedKFold(nFolds)
count       = 0


test_passed=0
beginTime = timer()
for epoch in range(10):

    for train_index, test_index in kf.split(Features,y):

        X_train, X_test = Features[train_index], Features[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        #TRAINING
        startTime = timer()
        idxM = int(len(train_index) / M)
        train_idxM = np.arange(0, idxM*M, M )
    
        for idx in range(idxM):
            M_idx = np.arange(train_idxM[idx], train_idxM[idx]+M )
            Xm, Ym = X_train[M_idx], Y_train[M_idx]
            
            #Feed forward
            AL1 = NN_SigmoidFeedForward(Xm , BL1, WL1)
            AL2 = NN_SigmoidFeedForward(AL1, BL2, WL2)
            AL3 = NN_SigmoidFeedForward(AL2, BL3, WL3)
            AL4 = NN_SigmoidFeedForward(AL3, BL4, WL4)
            
            #Back propagation process
            DLA4 = (AL4 - Ym)/(AL4*(1-AL4))
            WL4  = WL4 - lRate * DerivativeBackPropagation_DLW(DLA4, AL4, AL3) 
            BL4  = BL4 - lRate * DerivativeBackPropagation_DLB(DLA4, AL4)
              
            DLA3 = DerivativeBackPropagation_DLA(DLA4, AL4, WL4)
            WL3  = WL3 - lRate * DerivativeBackPropagation_DLW(DLA3, AL3, AL2) 
            BL3  = BL3 - lRate * DerivativeBackPropagation_DLB(DLA3, AL3)
              
            DLA2 = DerivativeBackPropagation_DLA(DLA3, AL3, WL3)
            WL2  = WL2 - lRate * DerivativeBackPropagation_DLW(DLA2, AL2, AL1) 
            BL2  = BL2 - lRate * DerivativeBackPropagation_DLB(DLA2, AL2)
              
            DLA1 = DerivativeBackPropagation_DLA(DLA2, AL2, WL2)
            WL1  = WL1 - lRate * DerivativeBackPropagation_DLW(DLA1, AL1, Xm) 
            BL1  = BL1 - lRate * DerivativeBackPropagation_DLB(DLA1, AL1)


    
        #TESTING PROCESS
        idxM      = int( len(test_index) / M )
        test_idxM = np.arange( 0, idxM*M, M )
        Yh_test   = np.random.rand(M,1)
        tacount = 0
        ttcount = 0
        tfcount = 0
        tzcount = 0
        tocount = 0
        for idx in range(idxM):
            M_idx = np.arange(test_idxM[idx], test_idxM[idx]+M )
            Xm, Ym = X_test[M_idx], Y_test[M_idx]

            AT1 = NN_SigmoidFeedForward(Xm , BL1, WL1)
            AT2 = NN_SigmoidFeedForward(AT1, BL2, WL2)
            AT3 = NN_SigmoidFeedForward(AT2, BL3, WL3)
            AT4 = NN_SigmoidFeedForward(AT3, BL4, WL4)

            for midx in range( M ):
                if AT4[midx] > 0.5:
                    Yh_test[midx] = 1
                else:
                    Yh_test[midx] = 0

                if (Ym[midx] == 1):
                    tocount = tocount+1
                else:
                    tzcount = tzcount+1

                if (Ym[midx] == Yh_test[midx]):
                    tacount = tacount + 1
                    if (Yh_test[midx]==1):
                        ttcount = ttcount + 1
                    else:
                        tfcount = tfcount + 1


        if((tacount/len(test_index)) > 0.7):
            print("Test passed")
            test_passed=1
            break;
    if(test_passed==1):
        break;

endTime = timer()
print('Total time= %f'%(endTime-beginTime))



print('\n\n------------------------------------------------------------')
print('Overal score       = %.4f%%   [ %d / %d ]\n' %( (tacount*100/len(test_index) ), tacount,len(test_index)  ) )
print('------------------------------------------------------------')
print('True values score  = %.4f%%   [ %d / %d ]\n' %( (ttcount*100/tocount), ttcount,tocount ) )
print('False values score = %.4f%%   [ %d / %d ]\n' %( (tfcount*100/tzcount), tfcount,tzcount ) )


quit()
print('\n\n==============   WEIGHT RESULT   ==============')
print('>>  W1= \n', WL1)
print('>>  B1= \n', BL1)
print('------------------------------------------------')
print('>>  W2= \n', WL2)
print('>>  B2= \n', BL2)
print('------------------------------------------------')
print('>>  W3= \n', WL3)
print('>>  B3= \n', BL3)
print('------------------------------------------------')
print('>>  W4= \n', WL4)
print('>>  B4= \n', BL4)

quit()

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

