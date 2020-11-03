import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.metrics import roc_curve, auc, roc_auc_score
import json

with open('Utils/direc.json') as f:
    data = json.load(f)
    FEATURE_TRAIN = data["FEATURE_TRAIN"]
    FEATURE_VALIDATION = data["FEATURE_VALIDATION"]
    FEATURE_TEST = data["FEATURE_TEST"]
    WORKING_DIR = data["WORKING_DIR"]


class Random_Forest:
    def __init__(self):
        df_train = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_TRAIN)
        df_validation = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_VALIDATION)
        df_test = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_TEST)

        df_train = df_train.drop(columns=['Unnamed: 0'])
        df_validation = df_validation.drop(columns=['Unnamed: 0'])
        df_test = df_test.drop(columns=['Unnamed: 0'])

        df_train_media = df_train.groupby('0').mean()
        df_validation_media = df_validation.groupby('0').mean()
        df_test_media = df_test.groupby('0').mean()

        df_train_media = df_train_media.reset_index()
        df_validation_media = df_validation_media.reset_index()
        df_test_media = df_test_media.reset_index()

        self.df_train_media = df_train_media.drop(columns=['0'])
        self.df_validation_media = df_validation_media.drop(columns=['0'])
        self.df_test_media = df_test_media.drop(columns=['0'])


    def get_best_parameters(self, num_trials = 200):

        def objective(trial):
            train_X, val_X, train_y, val_y = self.df_train_media.loc[:, self.df_train_media.columns != '41'].values, self.df_validation_media.loc[:, self.df_validation_media.columns != '41'].values, self.df_train_media['41'].values, self.df_validation_media['41'].values 
            test_X, test_y = self.df_test_media.loc[:, self.df_test_media.columns != '41'].values, self.df_test_media['41'].values
            list_trees = [250, 500, 1000, 1500, 3000, 3500, 4000]

            n_estimators = trial.suggest_categorical('n_estimators', list_trees)
            max_features = trial.suggest_uniform('max_features', 0.15, 1.0)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 16)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 16)
            min_weight_fraction_leaf = trial.suggest_uniform('min_weight_fraction_leaf', 0, 0.5)
            max_depth = trial.suggest_int('max_depth', 2, 32)


            brfmodel = BalancedRandomForestClassifier(n_estimators = n_estimators,
                                    max_features = max_features,
                                    min_samples_split = min_samples_split,
                                    min_samples_leaf = min_samples_leaf, 
                                    max_depth = max_depth,
                                    min_weight_fraction_leaf = min_weight_fraction_leaf,              
                                    bootstrap = True)
            brfmodel.fit(train_X, train_y)
            aucbrf = roc_auc_score(val_y, brfmodel.predict_proba(val_X)[:, 1])
            print("Test AUC: " + str(roc_auc_score(test_y, brfmodel.predict_proba(test_X)[:, 1])))
            
            return aucbrf

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials = num_trials)
        print(study.best_trial)

        return study.best_trial


