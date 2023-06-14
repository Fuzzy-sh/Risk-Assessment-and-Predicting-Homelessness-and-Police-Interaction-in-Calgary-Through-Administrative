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
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import os
import sys

#############################################################################


def return_train_test_set(df_name,key):
    # read the database from the computer 
    data_final=pd.read_hdf(df_name,key)

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
       'EX_Delirium', 'Ex_MS', 'EX_parkinsons', 

]

    # select the features and normalize them 
    X=data_final[feature_list_for_training]
    scaler_PT = PowerTransformer() 
    X = pd.DataFrame(scaler_PT.fit_transform(X), columns=X.columns)
    # target
    y=data_final['police_interaction']
    # split the data into test and training 
    x_train, x_test, y_train,  y_test=  train_test_split(X,y,test_size=0.1, random_state=42)
    # oversampling with Random over sampler  technique 
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(x_train, y_train)
    
    return X_res, y_res , x_test, y_test




##############################################################################################

# drawing the results for hyper-parameterization
def confusion_AUC_Return(x_test,y_test,model,params):
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

def train_model( subsample,n_estimators,max_depth,learning_rate,colsample_bytree,colsample_bylevel,X_res, y_res , x_test, y_test):
     # define the model with hyper parameters
    xgboost_model = XGBClassifier(
    subsample= subsample,
    n_estimators= n_estimators,
    max_depth= max_depth,
    learning_rate= learning_rate,
    colsample_bytree= colsample_bytree,
    colsample_bylevel= colsample_bylevel)

    xgboost_model.fit(X_res, y_res, )

    # show the results
    roc_auc, F1,recall,precision=confusion_AUC_Return(x_test,y_test,xgboost_model,2)
    return roc_auc, F1,recall,precision
    #     print('AUC: {:.2f} | F1: {:.2f} | recall: {:.2f} | precision: {:.2f}'.format(roc_auc, F1,recall,precision))

#####################################################################################################

def main(subsample,n_estimators,max_depth,learning_rate,colsample_bytree,colsample_bylevel,file_num,model_name):   
    df_name='data/data_final_policing.h5'
    key='data_final_policing'
    X_res, y_res , x_test, y_test=return_train_test_set(df_name,key)    

    roc_auc, F1,recall,precision=train_model (subsample,n_estimators,max_depth,learning_rate,colsample_bytree,colsample_bylevel,X_res, y_res , x_test, y_test)
    
    data='{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f}'.format(max_depth,learning_rate,subsample,colsample_bytree,colsample_bylevel,n_estimators,roc_auc, F1,recall,precision)
    with open (model_name+ str(file_num)+'.csv', 'w') as fp:
        fp.write(data)


###################################################################################

if __name__ == "__main__":
    print(int(sys.argv[1]))
    max_depth=int(sys.argv[1])
    learning_rate=float(sys.argv[2])
    subsample=float(sys.argv[3])
    colsample_bytree=float(sys.argv[4])
    colsample_bylevel=float(sys.argv[5])
    n_estimators=int(sys.argv[6])
    file_num=int(sys.argv[7])
    model_name='data/xGBoost/'
    file_exist=[file for file in os.listdir(model_name) if file.split('.')[0]==str(file_num)]
    if len(file_exist)==0:
        main(subsample,n_estimators,max_depth,learning_rate,colsample_bytree,colsample_bylevel,file_num,model_name)