import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import json

with open('Utils/direc.json') as f:
    data = json.load(f)
    FEATURE_TRAIN = data["FEATURE_TRAIN"]
    FEATURE_VALIDATION = data["FEATURE_VALIDATION"]
    FEATURE_TEST = data["FEATURE_TEST"]
    FEATURE_LAB = data["FEATURE_LAB"]
    WORKING_DIR = data["WORKING_DIR"]

class Merge_faeture():
    def __init__(self, epoch, pca_component, pca = False):

        self.epoch = epoch
        self.pca = pca
        self.pca_component = pca_component

        try:
            os.makedirs('Features/Merged/')
        except Exception as e:
            print(e)

        df_train = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_TRAIN)
        df_validation = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_VALIDATION)
        df_test = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_TEST)

        df_lab = pd.read_csv(FEATURE_LAB, header = 0)

        df_lab = df_lab.replace({'ESITO':{'good':0,'bad':1}})
        df_lab = df_lab.drop(columns=['Coronaropatia'])

        df_train_val = df_train.reset_index()
        df_train_val = df_train_val.drop(columns=['Unnamed: 0', '41']).rename(columns = {'0':'id_paziente'})

        df_validation = df_validation.reset_index()
        df_validation = df_validation.drop(columns=['Unnamed: 0', '41']).rename(columns = {'0':'id_paziente'})

        df_test = df_test.reset_index()
        df_test = df_test.drop(columns=['Unnamed: 0', '41']).rename(columns = {'0':'id_paziente'})


        if self.pca == True:

            scaler = MinMaxScaler()
            pca = PCA(n_components = pca_component)
            df_train_pca = df_train_val.loc[:, df_train_val.columns != 'id_paziente']
            df_train_pca = scaler.fit_transform(df_train_pca)
                  
            
            # PCA TRAINING DATASET
            pca.fit(df_train_pca)   
            df_train_pca = pd.DataFrame(pca.transform(df_train_pca))
            df_train_pca.columns = list(map(str, list(range(0, pca_component))))
            df_train_pca.insert(pca_component, "id_paziente", df_train_val['id_paziente'].values, True) 

            # PCA VALIDATION DATASET
            df_validation_pca = df_validation.loc[:, df_validation.columns != 'id_paziente']
            df_validation_pca = scaler.transform(df_validation_pca)
            df_validation_pca = pd.DataFrame(pca.transform(df_validation_pca))
            df_validation_pca.columns = list(map(str, list(range(0, pca_component))))
            df_validation_pca.insert(pca_component, "id_paziente", df_validation['id_paziente'].values, True) 
            
            # PCA TEST DATASET
            df_test_pca = df_test.loc[:, df_test.columns != 'id_paziente']
            df_test_pca = scaler.transform(df_test_pca)
            df_test_pca = pd.DataFrame(pca.transform(df_test_pca))
            df_test_pca.columns = list(map(str, list(range(0, pca_component))))
            df_test_pca.insert(pca_component, "id_paziente", df_test['id_paziente'].values, True) 

            df_train_val = df_train_pca
            df_validation = df_validation_pca
            df_test = df_test_pca

        df_lab = df_lab.sort_values(by=['id_paziente'])
        df_lab = df_lab.reset_index(drop=True)

        try:
            if self.pca == True:
                os.makedirs('Features/Merged/epoch_{}_pca_{}'.format(self.epoch, self.pca_component))
            else:
                os.makedirs('Features/Merged/epoch_{}'.format(self.epoch))
        except Exception as e:
            print(e)


        if self.pca == True:
            merged_folder = 'Features/Merged/epoch_{}_pca_{}'.format(self.epoch, self.pca_component)
        else:
            merged_folder = 'Features/Merged/epoch_{}'.format(self.epoch)

        df_merged_train = pd.merge(df_lab, df_train_val, on='id_paziente', how='inner')
        df_merged_train = df_merged_train.set_index('id_paziente')
        df_merged_train.to_csv (merged_folder + '/Mergedmean_epoch_{}.csv'.format(self.epoch), index = True, header=True)

        df_merged_validation = pd.merge(df_lab, df_validation, on='id_paziente', how='inner')
        df_merged_validation = df_merged_validation.set_index('id_paziente')
        df_merged_validation.to_csv (merged_folder + '/Mergedmean_validation_epoch_{}.csv'.format(self.epoch), index = True, header=True)

        df_merged_test = pd.merge(df_lab, df_test, on='id_paziente', how='inner')
        df_merged_test = df_merged_test.set_index('id_paziente')
        df_merged_test.to_csv (merged_folder + '/Mergedmean_test_epoch_{}.csv'.format(self.epoch), index = True, header=True)

