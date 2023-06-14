'''
1- import the liberaries
2- return_train_test_set: read the files and create the treain and test sets
3- confusion_AUC_Return : calculate the metrics "roc_auc, F1,recall,precision"
4- train_model: train the model
5- main : call the functions

'''
##########################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, f1_score,precision_score, recall_score,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import os
import sys

#############################################################################

def return_train_test_set(df_name,key):
    # read the database from the computer 
    data_final=pd.read_hdf(df_name,key=key)

    feature_list_for_training=[ 
       'sex_M',
       'age',
        
       'substance', 
       'mood', 
       'anxiety', 
       'psychotic', 
       'cognitive', 
       'otherpsych',
       'selfharm', 
    
       'visit_emr_MH_non_elect', 
       'visit_emr_NonMH',
       'visit_emr_visit', 
       
       'visit_hosp_visit',
       'visit_hospitalized_MH', 
       'visit_hospitalized_NonMH', 
    
       'visit_family_gp', 
       'visit_im',
       'visit_neurology', 
       'visit_other', 
       'visit_pharmacy', 
       'visit_psychiatry',
    
       'EX_CHF', 
       'EX_Arrhy', 
       'EX_VD', 
       'EX_PCD', 
       'EX_PVD', 'EX_HPTN_UC',
       'EX_HPTN_C', 'EX_Para', 'Ex_OthND', 'Ex_COPD', 'Ex_Diab_UC',
       'Ex_Diab_C', 'Ex_Hptothy', 'Ex_RF', 'Ex_LD', 'Ex_PUD_NB', 'Ex_HIV',
       'Ex_Lymp', 'Ex_METS', 'Ex_Tumor', 'Ex_Rheum_A', 'Ex_Coag', 'Ex_Obesity',
       'Ex_WL', 'Ex_Fluid', 'Ex_BLA', 'Ex_DA', 'Ex_Alcohol', 'Ex_Drug',
       'Ex_Psycho', 'Ex_Dep', 'Ex_Stroke', 'Ex_Dyslipid', 'Ex_Sleep', 'Ex_IHD',
       'EX_Fall', 'EX_Urinary', 'EX_Visual', 'EX_Hearing', 'EX_Tobacco',
       'EX_Delirium', 'Ex_MS', 'EX_parkinsons']

    # select the features and normalize them 
    X=data_final[feature_list_for_training]
    scaler_PT = PowerTransformer() 
    X = pd.DataFrame(scaler_PT.fit_transform(X), columns=X.columns)
    # target
    y=data_final['homeless']
    # split the data into test and training 
    x_train, x_test, y_train,  y_test=  train_test_split(X,y,test_size=0.1, random_state=42)
    # oversampling with Random over sampler  technique 
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(x_train, y_train)
    
    return X_res, y_res , x_test, y_test

##############################################################################################

# drawing the results for hyper-parameterization
def confusion_AUC_Return(x_test,y_test,model):
    # comparing original and predicted values of y
    y_pred = model.predict(x_test)
#     prediction = list(map(round, y_pred))

    # Find prediction to the dataframe applying threshold
    prediction = pd.Series(y_pred).map(lambda x: 1 if x > 0.5 else 0)
    
    # confusion matrix
    cm = confusion_matrix(y_test, prediction)


    # accuracy score of the model
    roc_auc = roc_auc_score(y_test, prediction)
    F1=f1_score(y_test, prediction)
    precision=precision_score(y_test, prediction)
    recall=recall_score(y_test, prediction)
    return (roc_auc, F1,recall,precision)

##################################################################################################


def train_model( bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators,criterion,X_res, y_res , x_test, y_test):
     # define the model with hyper parameters
    RF_model = RandomForestClassifier(
    
#         bootstrap=bootstrap,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        criterion=criterion
    
    )

    RF_model.fit(X_res, y_res, )

    # show the results
    roc_auc, F1,recall,precision=confusion_AUC_Return(x_test,y_test,RF_model)
    return roc_auc, F1,recall,precision

#####################################################################################################

def main(bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators,criterion,file_num,model_name):   
    
    df_name='data/data_final_homlessness.h5'
    key='data_final_homlessness'
    X_res, y_res , x_test, y_test=return_train_test_set(df_name,key)    
 

    roc_auc, F1,recall,precision=train_model (bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators,criterion,X_res, y_res , x_test, y_test)
    
    data='{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f}'.format(bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators,criterion,roc_auc, F1,recall,precision)
    with open (model_name+ str(file_num)+'.csv', 'w') as fp:
        fp.write(data)

###################################################################################

if __name__ == "__main__":
    
    bootstrap=sys.argv[1]
    max_depth=int(sys.argv[2])
    max_features=sys.argv[3]
    min_samples_leaf=int(sys.argv[4])
    min_samples_split=int(sys.argv[5])
    n_estimators=int(sys.argv[6])
    criterion=sys.argv[7]
    file_num=int(sys.argv[8])
    model_name='data/RF/'
    file_exist=[file for file in os.listdir(model_name) if file.split('.')[0]==str(file_num)]
    if len(file_exist)==0:
        main(bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators,criterion,file_num,model_name)
    
