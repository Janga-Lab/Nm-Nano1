# To set seed random number in order to reproducable results in keras
from numpy.random import seed
seed(4)
import tensorflow
tensorflow.random.set_seed(1234)
########################################
from gensim.models import Word2Vec  #for getting kmer embedding
########################################
import pandas as pd
from pandas import *
import numpy as np
import random
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#classifier=KNeighborsClassifier(n_neighbors=2)
#classifier= GaussianNB() #leads to best performance
#classifier=LogisticRegression(random_state=random.seed(1234))

classifier= xgb.XGBClassifier(colsample_bytree= 1,eta= 0.01,gamma= 0.1,max_depth= 15,min_child_weight= 3,scale_pos_weight= 1,silent= True)

#classifier =svm.SVC(gamma='scale',C=1,probability=True)
#classifier =RandomForestClassifier(n_estimators=30, max_depth=10, random_state = random.seed(1234))#random_state=0)
import plot_learning_curves as plc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #For feature normalization

scaler = MinMaxScaler()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]#.astype(np.float64)

def Union(lst1, lst2): 
    final_list = lst1 + lst2 
    return final_list 
    

#dataset = pd.read_csv('ps_hela_data.csv', index_col=0)  #Pandas: read csv file: https://realpython.com/pandas-read-write-files
dataset = pd.read_csv('Nm_hela_data.csv', index_col=0)  #Pandas: read csv file: https://realpython.com/pandas-read-write-files
dataset_test = pd.read_csv('Nm_hek_data.csv', index_col=0)

'''
#for getting samples of train and test datasets with the same uniform distributions 
dataset['freq'] = dataset.groupby('label')['label'].transform('count')  #https://stackoverflow.com/questions/55042334/pandas-sample-with-weights
dataset_test['freq'] = dataset_test.groupby('label')['label'].transform('count')
#print(dataset['freq'])
#print(dataset_test['freq'])

dataset = dataset.sample(n=2000,weights = dataset.freq)
dataset_test = dataset_test.sample(n=6000,weights = dataset_test.freq)
print("***************************")
print(dataset['label'].value_counts())
print(dataset_test['label'].value_counts())
print("***************************")
'''
dataset['mean_diff'] = (dataset['event_level_mean'] - dataset['model_mean']).astype(int)
dataset_test['mean_diff'] = (dataset_test['event_level_mean'] - dataset_test['model_mean']).astype(int)

dataset['kmer_match'] = np.where((dataset['reference_kmer'] == dataset['model_kmer']), 1, 0)
print(dataset.head())
dataset_test['kmer_match'] = np.where((dataset_test['reference_kmer'] == dataset_test['model_kmer']), 1, 0)
print(dataset_test.head())
    


#cols=['contig', 'position', 'reference_kmer', 'read_index', 'strand', 'event_index','event_level_mean', 'event_stdv',
#      'event_length', 'model_kmer', 'model_mean', 'model_stdv', 'standardized_level', 'kmer_match', 'label']

columns=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff','reference_kmer','label'] # with contig column
#columns=['event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff','reference_kmer','label']
#columns=['event_stdv','model_stdv','standardized_level','kmer_match','reference_kmer','label']



#dataset['label'] = dataset['label'].astype('category')
#dataset_test['label'] = dataset['label'].astype('category')

##scale training and testing data
dataset_train=dataset[columns]
dataset_train=clean_dataset(dataset_train)
dataset_test=dataset_test[columns]
dataset_test=clean_dataset(dataset_test)

print(dataset_train.shape)
print(dataset_test.shape)

#reference_kmer
#combine label_encoding of train and test
union_ref_kmer_set=set(dataset_train.iloc[:, 2]).union(set(dataset_test.iloc[:, 2]))
union_kmer=list(union_ref_kmer_set)
print(len(union_kmer))
dataset_train['reference_kmer']=pd.Categorical(dataset_train['reference_kmer'], categories=list(union_kmer))
dataset_test['reference_kmer']=pd.Categorical(dataset_test['reference_kmer'], categories=list(union_kmer))


'''
#combine label_encoding of train and test
union_contig_set=set(dataset_train.iloc[:, 0]).union(set(dataset_test.iloc[:, 0]))
union_contig=list(union_contig_set)
print(len(union_contig))
dataset_train['contig']=pd.Categorical(dataset_train['contig'], categories=list(union_contig))
dataset_test['contig']=pd.Categorical(dataset_test['contig'], categories=list(union_contig))
'''

print(dataset_train.shape)
print(dataset_test.shape)


##################################################

columns1=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff']
#####################columns1=['event_stdv','model_stdv','standardized_level','kmer_match']

#Feature importance
#columns1=['position']
#columns1=['event_level_mean']
#columns1=['event_stdv']
#columns1=['model_mean']
#columns1=['model_stdv']
#columns1=['kmer_match']
#columns1=['mean_diff']

X = dataset_train[columns1]
#To add label encoding of contig column to train data ()
#dataset_train['contig'] = dataset_train['contig'].astype('category')
dataset_train['reference_kmer'] = dataset_train['reference_kmer'].astype('category')
lb1 = dataset_train['reference_kmer'].cat.codes  #for label encoding
#lb2 = dataset_train['contig'].cat.codes 
lb1.reset_index(drop=True, inplace=True)
#lb2.reset_index(drop=True, inplace=True)
#X= pd.concat([X,lb1,lb2],axis=1)
#X= pd.concat([X,lb1],axis=1)

X_t = dataset_test[columns1]

#To add label encoding of contig column to test data (x_t)
dataset_test['reference_kmer'] = dataset_test['reference_kmer'].astype('category')
#dataset_test['contig'] = dataset_test['contig'].astype('category')
lb_test1= dataset_test['reference_kmer'].cat.codes  #for label encoding
#lb_test2 = dataset_test['contig'].cat.codes  #for label encoding
lb_test1.reset_index(drop=True, inplace=True)
#lb_test2.reset_index(drop=True, inplace=True)
#X_t= pd.concat([X_t,lb_test1,lb_test2],axis=1)
#X_t= pd.concat([X_t,lb_test1],axis=1)





print("shape of X=",X.shape)

print("shape of X_t=",X_t.shape)


#X=df3  #TO test the effect of embedding only
####################################
#only when test with random split  # 
####################################

#Y= dataset_train['label'] 
#X= scaler.fit_transform(X) 

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

##############################################
#only when test with independent cell line   # 
##############################################
X_train=X
X_test=X_t
#X_train= scaler.fit_transform(X_train)   
#X_test= scaler.fit_transform(X_test)   
y_train= dataset_train['label'] 
y_test = dataset_test['label']
print("shape of X_train=",X_train.shape)
print("shape of y_train=",y_train.shape)
print("shape of X_test=",X_test.shape)
print("shape of y_test=",y_test.shape)

#####################################

clf = classifier.fit(X_train,y_train)

#clf = classifier.fit(X_train,y_train.ravel())


y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
y_prob = y_prob[:,1]
# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix


print("TT",y_test.shape)
print("PP",y_pred.shape)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
 
print(classification_report(y_test, y_pred))
auc=roc_auc_score(y_test.round(),y_pred)
auc = float("{0:.3f}".format(auc))
print("AUC=",auc)
#true negatives c00, false negatives C10, true positives C11, and false positives C01 
#tn c00, fpC01, fnC10, tpC11 
print('CF=',confusion_matrix(y_test, y_pred))
l=confusion_matrix(y_test, y_pred)#https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
print('TN',l.item((0, 0)))
print('FP',l.item((0, 1)))
print('FN',l.item((1, 0)))
print('TP',l.item((1, 1)))
#print(type(X_train), type(y_train))


#plot learning curve: works with all classifier and all features except x(padded signal) as it leads to error with SVM 
#References:https://medium.com/@datalesdatales/why-you-should-be-plotting-learning-curves-in-your-next-machine-learning-project-221bae60c53
import matplotlib.pyplot as plt

# To address the error at https://stackoverflow.com/questions/55291667/getting-typeerror-slicenone-none-none-0-is-an-invalid-key
print(type(X_train))
print(type(X_test))
X_train = X_train.values
X_test = X_test.values

plc. plot_learning_curves(classifier, X_train, y_train, X_test, y_test)

# Create plot
#plt.title("Learning Curve")
plt.xlabel("Training Set (Size)"), plt.ylabel("Accuracy")#, plt.legend(loc="best")
plt.xticks(fontsize=18, weight = 'bold') 
plt.yticks(fontsize=18, weight = 'bold')
plt.tight_layout()
#plt.show()
#plt.savefig('Xgboost_LC_diff_cell_hek.png',dpi=300)
#plt.savefig('Xgboost_LC_diff_cell_hek.svg',dpi=300)
plt.savefig('Xgboost_LC_no_embedding_hela_validation.png',dpi=300)
plt.savefig('Xgboost_LC_no_embedding_hela_validation.svg',dpi=300)
plt.close()

#plot ROC curve: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt1

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt1.plot(fpr,tpr)
#plt1.title("ROC Curve")
# axis labels
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt.xticks(fontsize=12, weight = 'bold') 
plt.yticks(fontsize=12, weight = 'bold')
plt1.legend(loc="best")
#plt.show()
#plt1.savefig('Xgboost_ROC_diff_cell_hek.png',dpi=300)
#plt1.savefig('Xgboost_ROC_diff_cell_hek.svg',dpi=300) 
plt1.savefig('Xgboost_ROC_no_embedding_hela_validation.png',dpi=300)
plt1.savefig('Xgboost_ROC_no_embedding_hela_validation.svg',dpi=300)
plt1.close()

#############################################
#old code to plot learning curve: works only with RandomForest
#Reference: https://www.dataquest.io/blog/learning-curves-machine-learning/
##################

print("Features=",columns1)


