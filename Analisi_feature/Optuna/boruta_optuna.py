import pandas as pd
import numpy as np
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib import pyplot as plt
import shap
import statistics
import optuna 
import json 
import os
from sklearn.metrics import accuracy_score
from BorutaShap import BorutaShap
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt 

class Boruta_optuna():
    def __init__(self, filename, filename_validation, filename_test, epoch, pca = False, pca_component = 0, trials = 20):

        df = pd.read_csv(filename, header = 0)
        df = df.set_index('id_paziente')

        df_validation = pd.read_csv(filename_validation, header = 0)
        df_validation = df_validation.set_index('id_paziente')

        df_test = pd.read_csv(filename_test, header = 0)
        df_test = df_test.set_index('id_paziente')

        self.y = df.pop('ESITO').values
        self.X = df

        self.X_validation = df_validation
        self.y_validation = df_validation.pop('ESITO').values

        self.X_test = df_test
        self.y_test = df_test.pop('ESITO').values

        self.epoch = epoch
        self.pca = pca
        self.pca_component = pca_component
        self.trials = trials

        try:
            os.makedirs('Features/Boruta_RF')
        except Exception as e:
            print(e)

        try:
            if self.pca == True:
                self.result_folder = 'Features/Boruta_RF/epoch_{}_pca_{}'.format(self.epoch, self.pca_component)   
            else:
                self.result_folder = 'Features/Boruta_RF/epoch_{}'.format(self.epoch)
            os.makedirs(self.result_folder)
        except Exception as e:
            print(e)

    
    def optuna_RF(self):

        def objective(trial):

            train_X, val_X, train_y, val_y = train_test_split(self.X, self.y, test_size = 0.2)
            median_imputer = SimpleImputer(missing_values = np.NaN,strategy = 'median')
            v_train_X = median_imputer.fit_transform(train_X)
            v_val_X = median_imputer.fit_transform(val_X)
            train_X = pd.DataFrame(v_train_X, columns = train_X.columns,index = train_X.index)
            val_X = pd.DataFrame(v_val_X, columns = val_X.columns, index = val_X.index)

            v_test_X = median_imputer.fit_transform(self.X_validation)
            test_X = pd.DataFrame(v_test_X, columns = self.X_validation.columns,index = self.X_validation.index)
        
            list_trees = [250, 500, 1000, 1500, 3000, 3500, 4000]

            brf_n_estimators = trial.suggest_categorical('n_estimators', list_trees)
            brf_max_features = trial.suggest_uniform('max_features', 0.15, 1.0)
            brf_min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
            brf_min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)
            brf_min_weight_fraction_leaf = trial.suggest_uniform('min_weight_fraction_leaf', 0, 0.5)
            brf_max_depth = trial.suggest_int('max_depth', 2, 32)


            brfmodel = BalancedRandomForestClassifier(
                                    n_estimators =  brf_n_estimators,
                                    max_features =  brf_max_features,
                                    min_samples_split = brf_min_samples_split,
                                    min_samples_leaf = brf_min_samples_leaf, 
                                    max_depth = brf_max_depth,
                                    min_weight_fraction_leaf = brf_min_weight_fraction_leaf,              
                                    bootstrap = True)

            brfmodel.fit(train_X, train_y)

            aucbrf = roc_auc_score(val_y, brfmodel.predict_proba(val_X) [:,1])
            aucbrf_test = roc_auc_score(self.y_validation, brfmodel.predict_proba(test_X) [:,1])
            print('Accuracy test ' + str(accuracy_score(self.y_validation, brfmodel.predict(test_X))))

            plt.figure()
            plot_confusion_matrix(brfmodel, test_X, self.y_validation,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
            plt.show()
            print(aucbrf_test)
                        
            return aucbrf

        study = optuna.create_study(direction = "maximize")
        study.optimize(objective, n_trials = self.trials)

        print(study.best_trial)

        value = []

        for i in range(0, len(study.trials)):
            value_dict = study.trials[i].params
            value_dict['value'] = study.trials[i].value
            value.append(value_dict)
        value.sort(key = lambda x: x.get('value'))
        value = value[::-1][0:10]
    


        with open(self.result_folder + '/param_RF_{}.json'.format(self.epoch), 'w') as f:
            json.dump(value, f)

        return study


#################################################################
#################################################################

    def evaluate_model(self):

        with open(self.result_folder + '/param_RF_{}.json'.format(self.epoch)) as f:
            dati = json.load(f)

            for data in dati:

                del data['value']

                rf_model = BalancedRandomForestClassifier(**data)

                rf_auc = []

                for i in tqdm(range(20)):
                    
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
                    
                        rf_model.fit(trainX, trainy)
                        roc_rf = roc_auc_score(testy, rf_model.predict_proba(testX)[:,1])
                        rf_auc.append(roc_rf)

                        print(roc_rf)

            print(statistics.mean(rf_auc))
        return rf_auc

################################################################################
################################################################################

    def evaluate_on_validation_or_test(self, test = False):


        with open(self.result_folder + '/param_RF_{}.json'.format(self.epoch)) as f:
            dati = json.load(f)
            for data in dati:

                del data['value']

                rf_model = BalancedRandomForestClassifier(**data)

                trainX = self.X
                trainy = self.y
                valx = self.X_validation
                valy = self.y_validation
                if test == True:
                    testx = self.X_test
                    testy = self.y_test

                median_imputer = SimpleImputer(missing_values = np.NaN,strategy = 'median')
                imputer = median_imputer.fit(trainX)
                vtrainX = imputer.transform(trainX)
                trainX = pd.DataFrame(vtrainX, columns = trainX.columns, index = trainX.index)

                vvalX = imputer.transform(valx)
                valx = pd.DataFrame(vvalX, columns = valx.columns, index = valx.index)

                if  test == True:
                    vtest = imputer.transform(testx)
                    testx = pd.DataFrame(vtest, columns = testx.columns, index = testx.index)
                    trainX = pd.concat([trainX, valx])
                    trainy = np.concatenate((trainy, valy))
                
        
                rf_model.fit(trainX, trainy)
                
                if test == True:
                    roc_rf = roc_auc_score(testy, rf_model.predict_proba(testx)[:,1])
                else:
                    roc_rf = roc_auc_score(valy, rf_model.predict_proba(valx)[:,1])
                
                if test == False:
                    print("Validation AUC: {}".format(str(roc_rf)))
                else:
                    print("Test AUC: {}".format(str(roc_rf)))

################################################################
################################################################

    def random_boruta(self):

        with open(self.result_folder + '/param_RF_{}.json'.format(self.epoch)) as f:

            dati = json.load(f)
            for data in dati:
                del data['value']

                brfmodel = BalancedRandomForestClassifier(**data)

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
                    Feature_Selector = BorutaShap(
                                            model = brfmodel,
                                            importance_measure = 'shap',
                                            percentile = 85, 
                                            pvalue = 0.08,
                                            classification = True)
                    
                    Feature_Selector.fit(X_train, y_train, n_trials=200, random_state=0)
                    Feature_Selector.TentativeRoughFix()

                    Feature_Selector.plot(
                            X_size=12, 
                            figsize=(12,8),
                            y_scale='log', 
                            which_features='all')
                    
                    Xstrain = Feature_Selector.Subset()
                    selected = [x for x in Xstrain.columns]
                    print('features selected', selected)

                    v_test_X = median_imputer.fit_transform(self.X_test)
                    test_X = pd.DataFrame(v_test_X, columns=self.X_test.columns, index=self.X_test.index)

                    valx = self.X_validation
                    valy = self.y_validation
                    vvalX = imputer.transform(valx)
                    valx = pd.DataFrame(vvalX, columns = valx.columns, index = valx.index)

                    print('AUC')
                    brfmodel.fit(X_train,y_train)
                    roc = roc_auc_score(y_test, brfmodel.predict_proba(X_test)[:,1])
                    print(roc)

                    print('AUC Validation')
                    roc_test = roc_auc_score(self.y_validation, brfmodel.predict_proba(valx)[:,1])
                    
                    print(roc_test)

                    print('AUC ridotte')
                    brfmodel.fit(Xstrain,y_train)
                    roc = roc_auc_score(y_test, brfmodel.predict_proba(X_test[selected])[:,1])
                    
                    print(roc)
                    roc_test = roc_auc_score(self.y_validation, brfmodel.predict_proba(valx[selected])[:,1])
                    
                    print(roc_test)



