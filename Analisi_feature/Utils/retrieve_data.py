from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import shutil
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cambiare la Working_dir prima di iniziare
WORKING_DIR = '/media/fabio/Disco locale1/Fabio/Programmazione/Python/Poliambulanza/Analisi_feature/'
FEATURE_TRAIN  = 'feature_estratte_train.csv'
FEATURE_VALIDATION  = 'feature_estratte_validation.csv'
FEATURE_TEST  = 'feature_estratte_test.csv'
FEATURE_FOLDER = '/media/fabio/Disco locale1/Fabio/Programmazione/Python/Poliambulanza/Analisi_feature/Features'

# Metodo per recuperare da GDrive le feature estratte dai dataset
# di training, validation e test e il modello aggiornato dalla CNN.
#

# Metodo per la creazione di cartelle
def create_folder(folder_name):
        try:
            os.makedirs(WORKING_DIR + folder_name)
        except Exception:
            print() 

class Retrieve_data:
    def __init__(self, epoch):
        self.epoch = epoch
        
    def get_model(self):
        FEATURE = [FEATURE_TRAIN,
                   FEATURE_VALIDATION,
                   FEATURE_TEST]
        create_folder('Features')

        for file in FEATURE:
            
            shutil.copyfile(FEATURE_FOLDER + 'epoch_{}'.format(self.epoch) + '/' + file, WORKING_DIR + 'Features/' + file)


       
def plot_distribution():
    create_folder('Output/Plot/Distribuzioni')
    create_folder('Output/Plot/Correlazioni')
    df_train = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_TRAIN)
    df_validation = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_VALIDATION)
    df_test = pd.read_csv(WORKING_DIR + 'Features/' + FEATURE_TEST)

    for i in range(1,41):
        plt.figure(figsize=(20,10))
        plt.grid()
        for j,k in df_train.groupby(['41']):
            sns.distplot(k[str(i)], bins=20, label=str(j)+ ' train')
        for j,k in df_validation.groupby(['41']):
            sns.distplot(k[str(i)], bins=20, label=str(j)+ ' validation')    
        for j,k in df_test.groupby(['41']):
            sns.distplot(k[str(i)], bins=20, label=str(j)+ ' test') 
        plt.legend()
        plt.savefig(WORKING_DIR + '/Output/Plot/Distribuzioni/distribuzione_{}'.format(str(i)))

    df_train = df_train.drop(columns=['Unnamed: 0', '41'])
    corr_matrix = df_train.corr()
    plt.figure(figsize=(30, 30))

    heatmap = sns.heatmap(corr_matrix,
                        square = True,
                        linewidths = .5,
                        cmap = 'coolwarm',
                        cbar_kws = {'shrink': .4,
                                    'ticks' : [-1, -.5, 0, 0.5, 1]},
                        vmin = -1,
                        vmax = 1,
                        annot = True,
                        annot_kws = {"size": 12})


    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.savefig(WORKING_DIR + '/Output/Plot/Correlazioni/Matrice_correlazione.pdf')




 