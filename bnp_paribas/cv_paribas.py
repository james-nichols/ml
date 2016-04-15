import numpy as np
import pandas as pd
import xgboost as xgb
import random
import csv

from sklearn.preprocessing      import LabelEncoder
from sklearn.preprocessing      import OneHotEncoder
import xgboost as xgb
from sklearn.cross_validation   import StratifiedKFold

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from ml_utilities import *

# --------------------------------------------------------------------------------
#
def evallogloss(preds, dtrain):
    
    labels = dtrain.get_label()
    
    return 'logloss', log_loss(labels, preds)

# --------------------------------------------------------------------------------
#
def do_train(X, Y, params, verbose=False):

    np.random.seed(1)
    random.seed(1)

    cv_scores = []
    train_scores = []
    
    split = StratifiedKFold(Y, 5, shuffle=True )
    fold = 0
    
    for train_index, cv_index in split:
    
        fold = fold + 1
                    
        X_train, X_valid    = X[train_index,:], X[cv_index,:]
        y_train, y_valid    = Y[train_index],   Y[cv_index]
    
    
        num_round       = params["n_estimators"]
        eta             = params["eta"]
        min_eta         = params["min_eta"]
        eta_decay       = params["eta_decay"]
        early_stop      = params["early_stopping_rounds"]
        max_fails       = params["max_fails"]
        
        params_copy     = dict(params)
        
        dtrain          = xgb.DMatrix( X_train, label=y_train ) 
        dvalid          = xgb.DMatrix( X_valid, label=y_valid )  
    
        total_rounds        = 0
        best_rounds         = 0
        pvalid              = None
        model               = None
        best_train_score    = None
        best_cv_score       = None
        fail_count          = 0
        best_rounds         = 0
        best_model          = None
        
        while eta >= min_eta:           
            
            model        = xgb.train( params_copy.items(), 
                                      dtrain, 
                                      num_round, 
                                      [(dtrain, 'train'), (dvalid,'valid')], 
                                      early_stopping_rounds=early_stop,
                                      feval=evallogloss )
    
            rounds          = model.best_iteration + 1
            total_rounds   += rounds
            
            train_score = log_loss( y_train, model.predict(dtrain, ntree_limit=rounds) )
            cv_score    = log_loss( y_valid, model.predict(dvalid, ntree_limit=rounds) )
    
            if best_cv_score is None or cv_score > best_cv_score:
                fail_count = 0
                best_train_score = train_score
                best_cv_score    = cv_score
                best_rounds      = rounds
                best_model       = model

                ptrain           = best_model.predict(dtrain, ntree_limit=rounds, output_margin=True)
                pvalid           = best_model.predict(dvalid, ntree_limit=rounds, output_margin=True)
                
                dtrain.set_base_margin(ptrain)
                dvalid.set_base_margin(pvalid)
            else:
                fail_count += 1

                if fail_count >= max_fails:
                    break
    
            eta                 = eta_decay * eta
            params_copy["eta"]  = eta
    
        train_scores.append(best_train_score)
        cv_scores.append(best_cv_score)

        print("Fold [%2d] %9.6f : %9.6f" % ( fold, best_train_score, best_cv_score ))
        
    print("-------------------------------")
    print("Mean      %9.6f : %9.6f" % ( np.mean(train_scores), np.mean(cv_scores) ) )
    print("Stds      %9.6f : %9.6f" % ( np.std(train_scores),  np.std(cv_scores) ) )
    print("-------------------------------")
      
    return best_model

# ----------------------------f----------------------------------------------------
#
def main():
   
    tr=pd.read_csv('train.csv')
    te=pd.read_csv('test.csv')

    target = tr['target']
    tr = tr.drop(['ID','target'],axis=1)
    IDs = te['ID'].values
    te = te.drop(['ID'],axis=1)

    # This seems to be a canonical list of data to avoid...
    #tr = tr.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
    #te = te.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

    tr = randomised_imputer(tr)
    te = randomised_imputer(te)
    
    Y = target.values  #train.target.values.astype(np.int32)
    X = tr.values      #train[ [ "VAR_0001", "VAR_0005", "VAR_0006", "VAR_0226"] ].values
   
    params = {
            "max_depth"             : 5, 
            "eta"                   : 0.1,
            "min_eta"               : 0.00001,
            "eta_decay"             : 0.5,
            "max_fails"             : 3,
            "early_stopping_rounds" : 20,
            "objective"             : 'binary:logistic',
            "subsample"             : 0.8, 
            "colsample_bytree"      : 1.0,
            "n_jobs"                : -1,
            "n_estimators"          : 5000, 
            "silent"                : 1,
            "gamma"                 : 0.1,
            "min_child_weight"      : 1.1
            }
    output_name = "pred_cv_{0}_{1}_{2}_{3}_{4}".format(params["eta"], params["n_estimators"], 
                                                    params["max_depth"], params["gamma"], params["min_child_weight"])

    print("\nWithout decay ...\n")
    best_model_nd = do_train(X, Y, params)

    dte = xgb.DMatrix(te.values)
    te_pred_nd = best_model_nd.predict(dte, ntree_limit=best_model_nd.best_iteration+1)
    
    # Save results
    predictions_file = open(output_name + ".csv", "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "PredictedProb"])
    open_file_object.writerows(zip(IDs, te_pred_nd))
    predictions_file.close()


# --------------------------------------------------------------------------------
#
if __name__ == '__main__':

    main()
