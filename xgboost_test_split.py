# To set seed random number in order to get reproducable results in keras
from numpy.random import seed
seed(4)
import tensorflow
tensorflow.random.set_seed(1234)
########################################
import pandas as pd
from pandas import *
import numpy as np
import random
import xgboost as xgb

classifier= xgb.XGBClassifier(colsample_bytree= 1,
 eta= 0.01,
 gamma= 0.1,
 max_depth= 15,
 min_child_weight= 3,
 scale_pos_weight= 1,
 silent= True)


from sklearn.model_selection import train_test_split
import plot_learning_curves as plc
from sklearn.preprocessing import MinMaxScaler #For feature normalization
scaler = MinMaxScaler()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]#.astype(np.float64)


df1 = pd.read_csv("Nm_Modification_coors_hek_complete.txt",sep=' ',skiprows=(0),header=(0))
df2 = pd.read_csv("hek-reads-ref.eventalign.txt",sep='\t',skiprows=(0),header=(0))
print(df2.shape)
print("&&&&&&&&")
print(df1.head())
print("***********************")
print(df2.head())
print("######################")


#print(df1['position'].iloc[0:5])
print(df1.iloc[0:5, 1])
print("@@@@@@@@@@@@@@@")
#print(df2['position'].iloc[0:5])
#print(df2.iloc[0:5, 1])
print(df2.iloc[0:5, 9])
print("######################")

#label the data
#kdf=df2
x=list(set(df1.iloc[:,1]).intersection(set(df2.iloc[:,1])))
print("length of intersection list",len(x))
df_Nm=df2[df2['position'].isin(x)]
listofones = [1] * len(df_Nm.index)
# Using DataFrame.insert() to add a column 
df_Nm.insert(13, "label", listofones, True)
df_non_Nm=df2[~df2['position'].isin(x)]
listofzeros=[0]*len(df_non_Nm.index)
df_non_Nm.insert(13, "label", listofzeros, True)
print(df_Nm.shape)
print(df_Nm.head())
print(df_non_Nm.shape)
print(df_non_Nm.head())
##########prepare dataset       
df_non_Nm = df_non_Nm.sample(n=len(df_Nm), replace=False) #try replace=false

# Create DataFrame from positive and negative examples
dataset = df_non_Nm.append(df_Nm, ignore_index=True)
dataset['label'] = dataset['label'].astype('category')
dataset.to_csv('hek_Nm_dataset_new.csv')
############################################################################################

dataset['mean_diff'] = (dataset['event_level_mean'] - dataset['model_mean']).astype(int)

#shuffle the test and train datasets
###from sklearn.utils import shuffle
###dataset = shuffle(dataset)
dataset['kmer_match'] = np.where((dataset['reference_kmer'] == dataset['model_kmer']), 1, 0)

print(dataset.head())
    

columns=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff','label']


# #scale training and testing data
dataset=dataset[columns]
dataset=clean_dataset(dataset)
##########################################################
##################################################

#columns1=['event_stdv','model_stdv','standardized_level','kmer_match']
columns1=['position','event_level_mean','event_stdv','model_mean','model_stdv','kmer_match','mean_diff']
#Feature importance
#columns1=['position']
#columns1=['event_level_mean']
#columns1=['event_stdv']
#columns1=['model_mean']
#columns1=['model_stdv']
#columns1=['kmer_match']
#columns1=['mean_diff']

X = dataset[columns1]
#######################################################
#########################################
print("#############",X.shape)
print(X.head())
print(type(X))
#scale training data
X= scaler.fit_transform(X)
Y = dataset['label'] 
print(",,,,,,,,",X.shape)
print(",,,,,,,,",Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3) for unblanced dataset


#clf = classifier.fit(X_train,y_train)
clf = classifier.fit(X_train,y_train.ravel())



y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
y_prob = y_prob[:,1]
# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix



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


from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt

ax =plot_learning_curves(X_train, y_train, X_test, y_test, clf,  print_model=False , style='classic')
# Adding axes annotations
plt.xlabel('Training set size(Percentage)',fontsize=18)
plt.ylabel('Performance(Misclassification error)',fontsize=18)
L=plt.legend(fontsize=20)
L.get_texts()[0].set_text('Training set')
L.get_texts()[1].set_text('Test set')
plt.gcf().set_size_inches(10, 8)
plt.grid(b=None) #for no grid
plt.savefig('XGBoost_loss_miss_test_split.png',dpi=300)
plt.close()



#plot learning curve: works with all classifier and all features except x(padded signal) as it leads to error with SVM 
#References:https://medium.com/@datalesdatales/why-you-should-be-plotting-learning-curves-in-your-next-machine-learning-project-221bae60c53
import matplotlib.pyplot as plt


plc. plot_learning_curves(classifier, X_train, y_train, X_test, y_test)

# Create plot
#plt.title("Learning Curve")
plt.xlabel("Training Set (Size)"), plt.ylabel("Accuracy")#, plt.legend(loc="best")
plt.tight_layout()
#plt.show()
plt.savefig('xgboost_LC_no_embedding_Nm_split.png',dpi=300)
plt.savefig('xgboost_LC_no_embedding_Nm_split.svg',dpi=300)
plt.close()

#plot ROC curve: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt1

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)


# Print ROC curve
plt1.plot(fpr,tpr)
plt1.title("ROC Curve")
# axis labels
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt1.legend(loc="best")
#plt.show() 
plt1.savefig('Xgboost_ROC_Nm_split.png',dpi=300)
plt1.savefig('xgboost_ROC_Nm_split.svg',dpi=300)
plt1.close()

#############################################
#old code to plot learning curve: works only with RandomForest
#Reference: https://www.dataquest.io/blog/learning-curves-machine-learning/
##################



