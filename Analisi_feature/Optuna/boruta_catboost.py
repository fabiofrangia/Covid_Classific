import os
import pandas as pd
import numpy as np
import statistics
import optuna 
import json 
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from BorutaShap import BorutaShap
from Utils.imputer import retrieve_test, retrieve_train_val

class Boruta_CatBoost():
    def __init__(self, filename, filename_valid, filename_test, epoch, trials, pca = False, pca_component = 0):
        self.filename = filename
        self.filename_test = filename_test
        self.filename_valid = filename_valid

        df = pd.read_csv(self.filename, header = 0)
        df = df.set_index('id_paziente')
        df_test = pd.read_csv(self.filename_test, header = 0)
        df_test = df_test.set_index('id_paziente')
        df_valid = pd.read_csv(self.filename_valid, header = 0)
        df_valid = df_valid.set_index('id_paziente')

        self.y = df.pop('ESITO').values
        self.X = df
        self.y_test = df_test.pop('ESITO').values
        self.X_test = df_test
        self.y_validation = df_valid.pop('ESITO').values
        self.X_validation = df_valid
        self.epoch = epoch
        self.trials = trials
        self.pca = pca
        self.pca_component = pca_component

        try:
            os.makedirs('Features/Boruta_CB')
        except Exception as e:
            print(e)

        try:
            if self.pca == True:
                self.result_folder = 'Features/Boruta_CB/epoch_{}_pca_{}'.format(self.epoch, self.pca_component)   
            else:
                self.result_folder = 'Features/Boruta_CB/epoch_{}'.format(self.epoch)
            os.makedirs(self.result_folder)
        except Exception as e:
            print(e)

    ##################################################################
    ######################   OPTUNA_CB   #############################
    ###                                                            ###
    ### VALUTAZIONE DELLA MIGLIORE CONFIGURAZIONE DI IPERPARAMETRI ###
    ###                                                            ###
    ##################################################################

    def optuna_cb(self):

        def objective(trial):

            train_X, val_X, train_y, val_y = retrieve_train_val(self.X, self.y, val_size = 0.2)
            test_X = retrieve_test(self.X_validation)
            
            list_eta = [0.001,0.005, 0.01,0.015,0.02]
            list_iterations = [250, 400, 550, 700, 850]
            
            param={
                'verbose' : False,
                'eval_metric':'AUC',
                'depth' : trial.suggest_int('depth', 4, 10),
                'objective' : trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
                'l2_leaf_reg':trial.suggest_loguniform('l2_leaf_reg', 3, 50),
                'boosting_type':'Ordered',
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'per_float_feature_quantization':trial.suggest_categorical('per_float_feature_quantization', ['33:border_count=1024', '33:border_count=256']),                                           
                'eta':trial.suggest_categorical('eta', list_eta),
                'iterations':trial.suggest_categorical('iterations', list_iterations),
            }
            if param['bootstrap_type'] == 'Bayesian':
                param['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0, 4)
            elif param['bootstrap_type'] == 'Bernoulli':
                param['subsample'] = trial.suggest_uniform('subsample', 0.1, 1)
            if param['objective'] == 'Logloss':
                param['random_strength'] = trial.suggest_uniform('random_strength', 0.5, 5)
            if param['objective'] == 'Logloss':
                param['scale_pos_weight'] = 0.46
        
            cbmodel = CatBoostClassifier(**param)
                                                                                                                
            cbmodel.fit(train_X, train_y)
            predictioncb = cbmodel.predict_proba(val_X)
            auccb = roc_auc_score(val_y, cbmodel.predict_proba(val_X) [:,1])

            
            print("AUC TEST")
            roc = roc_auc_score(self.y_validation, cbmodel.predict_proba(test_X)[:,1])
            print(roc)

            return auccb
        

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = 200)
        print(study.best_trial)

        value = []

        for i in range(0, len(study.trials)):
            value_dict = study.trials[i].params
            value_dict['value'] = study.trials[i].value
            value.append(value_dict)
        value.sort(key = lambda x: x.get('value'))
        value = value[::-1][0:10]



        with open(self.result_folder + '/param_CB_{}.json'.format(self.epoch), 'w') as f:
            json.dump(value, f)

        return study

    ######################################################################
    ######################################################################

    def evaluate_model(self):

        with open(self.result_folder + '/param_CB_{}.json'.format(self.epoch)) as f:
            dati = json.load(f)

            for data in dati:

                del data['value']

                cb_model = CatBoostClassifier(**data)

                cb_auc = []

                for i in tqdm(range(self.trials)):
                    
                    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = i+187462)
                    
                    for train_index, test_index in cv.split(self.X, self.y):
        
                        trainX = self.X.iloc[lambda x: train_index]
                        testX = self.X.iloc[lambda x: test_index]
                        
                        trainy = np.take(self.y, train_index)
                        testy = np.take(self.y, test_index)

                        median_imputer = SimpleImputer(missing_values = np.NaN,strategy = 'median')
                        imputer = median_imputer.fit(trainX)
                        vtrainX = imputer.transform(trainX)
                        imputertest = median_imputer.fit(testX)
                        vtestX = imputertest.transform(testX)
                        trainX = pd.DataFrame(vtrainX, columns = trainX.columns, index = trainX.index)
                        testX = pd.DataFrame(vtestX, columns = testX.columns, index = testX.index)
                        
                        # Calcolo AUC per migliori risultati da CatBoost
                    
                        cb_model.fit(trainX, trainy)
                        roc_cb = roc_auc_score(testy, cb_model.predict_proba(testX)[:,1])
                        cb_auc.append(roc_cb)

                print(statistics.mean(cb_auc))
        return cb_auc

    ########################################################################
    ########################################################################

    def random_boruta(self):
        with open(self.result_folder + '/param_CB_{}.json'.format(self.epoch)) as f:
            dati = json.load(f)

            for data in dati:
                del data['value']

                cb_model = CatBoostClassifier(**data)

                cv = StratifiedKFold(n_splits=5, shuffle=True)


                for train_index, test_index in cv.split(self.X, self.y):

                    X_train = self.X.iloc[lambda x: train_index]
                    X_test = self.X.iloc[lambda x: test_index]
                    y_train = np.take(self.y, train_index)
                    y_test = np.take(self.y, test_index)
                    
                    median_imputer = SimpleImputer(missing_values = np.NaN,strategy = 'median')
                    imputer = median_imputer.fit(X_train)
                    vX_train = imputer.transform(X_train)
                    imputertest = median_imputer.fit(X_test)
                    vX_test = imputertest.transform(X_test)
                    
                    X_train = pd.DataFrame(vX_train, columns = X_train.columns,index = X_train.index)
                    X_test = pd.DataFrame(vX_test, columns = X_test.columns,index = X_test.index)
                    Feature_Selector = BorutaShap(model = cb_model,
                                            importance_measure = 'shap',
                                            percentile = 90, 
                                            pvalue = 0.1,
                                            classification = True)
                    
                    Feature_Selector.fit(X_train, y_train, n_trials = 500, random_state = 0)
                    Feature_Selector.TentativeRoughFix()
                    Feature_Selector.plot(X_size=12, figsize=(12,8),
                            y_scale='log', which_features='all')
                    
                    Xstrain = Feature_Selector.Subset()
                    selected = [x for x in Xstrain.columns]
                    print('features selected',selected)

                    v_test_X = median_imputer.fit_transform(self.X_test)
                    test_X = pd.DataFrame(v_test_X, columns=self.X_test.columns,index=self.X_test.index)

                    cb_model.fit(Xstrain,y_train)

                    
                    print('AUC')
                    cb_model.fit(X_train,y_train)
                    roc = roc_auc_score(y_test, cb_model.predict_proba(X_test)[:,1])
                    
                    print(roc)

                    print('AUC TEST')
                    roc_test = roc_auc_score(self.y_test, cb_model.predict_proba(test_X)[:,1])
                    
                    print(roc_test)

        
        
 