#!/usr/bin/env python
# coding: utf-8

# In[1]:

import cv2
import shap
import pickle 
from PIL import Image 
import streamlit as st
import numpy as np 
import pandas as pd
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing._label import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
matplotlib.use('Agg')
plt.style.use('ggplot')

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

st.title("Application de prédiction de la Prime d'une couverture bâtiment d'une société d'assurance en utilisant les méthodes de Machine Learning  et incluant les interactions de variables")
st.write('Cette application prédit les ** Prix des Primes Batiment et Contenue **' )
st.sidebar.header('Caractéristiques Assurés/Clients')
st.write('---')

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# Loads the PRIME BATIMENT Dataset
# Faire une liste de types de valeurs manquantes
#path_B = 'C:/Users/emera/Web_app1/' 
path_model_B = 'model_pkl_B/'
path_image_B='Images_B/'
missing_values = ["n/a", "na", "--"]
data_B = pd.read_csv('data_B0.csv', sep=';',decimal=",",encoding = "ISO-8859-1",engine='python', na_values = missing_values)
data_B['Date_naissance'] = data_B["Date_naissance"].astype(str).apply(lambda x: int(x.split("/")[-1])if x!='nan'else np.nan)
data_B["age"] = data_B["Date_naissance"].apply(lambda x: 2020-(x-100) if x>2020 else 2020-x)
# on supprime les variables non necessaires
data_B.drop(['Numéro_sim','Date_naissance'],axis='columns',inplace=True)  # without the option inplace=True
#Créer une copie de la dataframe
data_B_moy = data_B.copy()
#calcul de la moyenne de la variable age  et imputer aux valeurs manquantes la moyenne obtenue 
mean_B= data_B_moy['age'].mean()
data_B_moy['age'].fillna(mean_B, inplace=True)
#calcul du mode de la variable CDPISEV  imputer aux valeurs manquantes le mode obtenue 
mode_B=data_B_moy['CDPISEV'].value_counts()
CDPISEV_mode_B =data_B_moy['CDPISEV'].value_counts().index[0]
data_B_moy['CDPISEV'].fillna(CDPISEV_mode_B,inplace=True)
# separate featues and labels
datae_B=data_B_moy.copy(deep=True)
data_B_X = datae_B.drop(['primebat','primecont'], axis=1)

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Spécifier les paramètres d\'entrée')

def input_datae_B_features():
    #selecbox variable
    typpos= st.sidebar.selectbox('typpos',('pro','loc'))
    abex = st.sidebar.selectbox('abex', (809, 819, 833))
    couverture = st.sidebar.selectbox('couverture',('B'))
    typhab = st.sidebar.selectbox('typhab',('App','M2','M3','M4'))
    CDPISEV = st.sidebar.selectbox('CDPISEV',('E','I'))
    tyhabri2 = st.sidebar.selectbox('tyhabri2',('BO','CI','CL','LU','MA','MO','SO','ST','VI')) 
    franchise_amount = st.sidebar.selectbox('franchise_amount', (261.73, 261.85, 262.38,262.39, 261.85, 263.33,263.38, 263.43, 263.44,263.5, 263.52,263.57 ,263.6, 263.61, 263.64,263.85,265.41,788.54,790.46,1577.1,1580.94,3154.2,3161.87))
    #franchise_amount = st.sidebar.slider('franchise_amount', data_B_X.franchise_amount.min(), data_B_X.franchise_amount.max(), data_B_X.franchise_amount.mean())
    
    #number input variable
    nbpiece_Eth = st.sidebar.number_input('nbpiece_Eth', data_B_X.nbpiece_Eth.min(), data_B_X.nbpiece_Eth.max(), data_B_X.nbpiece_Eth.min())
    age = st.sidebar.number_input('age',int(data_B_X.age.min()) ,int(data_B_X.age.max()),int(data_B_X.age.mean()))
    classgeo = st.sidebar.number_input('classgeo',  data_B_X.classgeo.min(), data_B_X.classgeo.max(), data_B_X.classgeo.min())
    garages = st.sidebar.number_input('garages', data_B_X.garages.min(), data_B_X.garages.max(), data_B_X.garages.min())
    daconri2 = st.sidebar.number_input('daconri2', data_B_X.daconri2.min(), data_B_X.daconri2.max(), data_B_X.daconri2.min())
    zip = st.sidebar.number_input('zip',data_B_X.zip.min(), data_B_X.zip.max(), data_B_X.zip.min())
    
    #slider variable
    capbat = st.sidebar.slider('capbat', data_B_X.capbat.min(), data_B_X.capbat.max(), data_B_X.capbat.mean())
    capcont = st.sidebar.slider('capcont', data_B_X.capcont.min(), data_B_X.capcont.max(), data_B_X.capcont.mean())
      
    columns_datae_B = {'typpos':typpos,
            'nbpiece_Eth': nbpiece_Eth,
            'capbat': capbat,
            'capcont': capcont,
            'franchise_amount': franchise_amount,
            'abex': abex,
            'zip': zip,
            'garages': garages,
            'couverture': couverture,
            'typhab': typhab,
            'classgeo': classgeo,
            'CDPISEV': CDPISEV,
            'daconri2': daconri2,
            'tyhabri2': tyhabri2,
            'age': age}
    datae_B_features = pd.DataFrame(columns_datae_B, index=[0])
    return datae_B_features

datae_B = input_datae_B_features()

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

dataB_primebat_moy=data_B_moy.drop(['primecont'], axis=1)
dataB_primecont_moy=data_B_moy.drop(['primebat'], axis=1)


############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# random state
SEED=1234
from pycaret.regression import * # on importe toute les fonction associée à la regression depuis pycaret step1
my_setup_primebatB = setup(data =dataB_primebat_moy, target = 'primebat', train_size=0.8, silent=True, session_id=SEED) # setup permet d'obtenir une description generale du dataset 
my_setup_primecontB = setup(data =dataB_primecont_moy, target = 'primecont', train_size=0.8, silent=True, session_id=SEED) # setup permet d'obtenir une description generale du dataset 


#Extracting functions data created by setup in Pycaret
X_B=my_setup_primebatB[0]
X_train_B=my_setup_primebatB[2]
X_test_B=my_setup_primebatB[3]
#################################
Y_primebatB=my_setup_primebatB[1]
Y_train_primebatB=my_setup_primebatB[4]
Y_test_primebatB=my_setup_primebatB[5]
setup_primebatB_desc=my_setup_primebatB[9]
#################################
Y_primecontB=my_setup_primecontB[1]
Y_train_primecontB=my_setup_primecontB[4]
Y_test_primecontB=my_setup_primecontB[5]
setup_primecontB_desc=my_setup_primecontB[9]

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# LOAD THE PRIMEBAT MODEL FROM PICKLE
load_lm_primebatB = pickle.load(open(path_model_B+'lm_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)
load_cbr_primebatB = pickle.load(open(path_model_B+'cbr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)
load_cbr_tuned_primebatB = pickle.load(open(path_model_B+'cbr_tuned_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)
load_lgbmr_primebatB = pickle.load(open(path_model_B+'lgbmr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)

load_gbmr_primebatB = pickle.load(open(path_model_B+'gbmr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)
load_rfr_primebatB = pickle.load(open(path_model_B+'rfr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)

load_xgb_primebatB = pickle.load(open(path_model_B+'xgb_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)
load_svmr_primebatB = pickle.load(open(path_model_B+'svmr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)

load_mlpr_primebatB = pickle.load(open(path_model_B+'mlpr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)
load_dtr_primebatB = pickle.load(open(path_model_B+'dtr_primebatB.pkl','rb')).fit(X_train_B,Y_train_primebatB)

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# LOAD THE PRIMECONT MODEL FROM PICKLE
load_lm_primecontB = pickle.load(open(path_model_B+'lm_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)
load_cbr_primecontB = pickle.load(open(path_model_B+'cbr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)
load_cbr_tuned_primecontB = pickle.load(open(path_model_B+'cbr_tuned_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)

load_lgbmr_primecontB = pickle.load(open(path_model_B+'lgbmr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)

load_gbmr_primecontB = pickle.load(open(path_model_B+'gbmr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)
load_rfr_primecontB = pickle.load(open(path_model_B+'rfr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)

load_xgb_primecontB = pickle.load(open(path_model_B+'xgb_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)
load_svmr_primecontB = pickle.load(open(path_model_B+'svmr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)

load_mlpr_primecontB = pickle.load(open(path_model_B+'mlpr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)
load_dtr_primecontB = pickle.load(open(path_model_B+'dtr_primecontB.pkl','rb')).fit(X_train_B,Y_train_primecontB)

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
df0_primebatB = dataB_primebat_moy.copy(deep=True)
#df0_primecontB = dataB_primecont_moy.copy(deep=True)
###############################################################
df0_primebatB_E= df0_primebatB.drop(columns=['primebat'],axis=0)
#df0_primecontB_E= df0_primecontB.drop(columns=['primecont'],axis=0)
###############################################################
df1_primebatB = pd.concat([datae_B,df0_primebatB_E],axis=0)
#df1_primecontB = pd.concat([data,df0_primecontB_E],axis=0)
###############################################################
# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['typpos','abex', 'garages','couverture','typhab','classgeo','CDPISEV','tyhabri2']
for col in encode:
    dummy_primebatB = pd.get_dummies(df1_primebatB[col], prefix=col)
    #dummy_primecontB = pd.get_dummies(df1_primecontB[col], prefix=col)
    ###############################################################
    df1_primebatB = pd.concat([df1_primebatB,dummy_primebatB], axis=1)
    #df1_primecontB = pd.concat([df1_primecontB,dummy_primecontB], axis=1)
    ###############################################################
    del df1_primebatB[col]
    #del df1_primecont[col]
    ######################
df_primebatB = df1_primebatB[:1] # Selects only the first row (the user input data)
#df_primecontB = df1_primecontB[:1] # Selects only the first row (the user input data)

###########################################################################################################################################################################################
############################################################################################################################################################################################
###########################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# In[7]:

st.header('1. Contexte du projet')

st.markdown(""" Dans ce projet, nous évaluons les performances et la puissance prédictive d'un modèle de ML qui a été formé et testé sur des données collectées auprès d'une entreprise d'assurance. 
Un modèle formé sur ces données et considéré comme un bon ajustement pourrait alors être utilisé pour faire certaines prédictions sur un assuré, en particulier sa prime. Ce modèle s’avérerait ainsi estimable pour un assureur (agent commercial) qui pourrait utiliser ces informations au quotidien.

Pour l'ensemble des données de ce projet, chacune des *13794* entrées représentent des données agrégées sur *17* caractéristiques pour divers individus. Pour les besoins de ce projet, les étapes de prétraitement suivantes ont été effectuées sur l'ensemble de données :
* La commande **`pandas.read_csv()`** a été utilisée pour effectuer le chargement des fichiers *.csv* dans le notebook. 
* Les valeurs de la variable **Date_naissance** ont été remplacées par les âges correspondant et la variable a été renommée **age**. Cette variable contient *133* valeurs manquantes qui ont été imputer en utilisant la moyenne des valeurs de la variable *age*.
* Les *13394* valeurs manquantes de la variable **CDPISEV** ont été remplacée en utilisant le mode des valeurs car c'est une variable catégorielle.
* Les variables non pertinentes ont été exclues du dataset.""")

##########################################################################
##########################################################################

st.header('1.1.Aperçu du dataset.')
if st.checkbox('Show dataframe'): 
     st.write(data_B_moy)
st.header("""--------------------------------------------------------------""")

############################################################################################################################################################################################
############################################################################################################################################################################################
###########################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

st.header('2. Configuration de l\'environnement et Exploration des données')

st.markdown("""Avant de commencer notre étude de ML à l’aide du logiciel `PyCaret` dans **Python**, définir l'environnement est nécessaire et cela comporte deux étapes simples : 
* **Étape 1**: Importer un module selon le type d'expérience que vous l'on souhaite effectuer, qui prépare notre environnement **Python** pour une tâche spécifique. Dans notre cas, 
l'environnement est configuré pour effectuer uniquement des tâches de régression.
* **Étape 2**: Initialiser la configuration commune à tous les modules de `PyCaret`. C'est la première et seule étape obligatoire pour démarrer une expérience de ML ; grâce à la 
commande **`setup()`** qui exécute les tâches de prétraitement de base par défaut permettant ainsi d'obtenir une description générale et exhaustive du dataset.""")

st.markdown("""Toutes les étapes de prétraitement sont appliquées dans **`setup()`**. Avec plus de 30 fonctionnalités pour préparer les données pour l'apprentissage automatique, 
` PyCaret `crée un pipeline de transformation basé sur les paramètres définis dans la fonction de configuration**`setup()`** . Il orchestre automatiquement toutes les dépendances 
dans le pipeline afin que nous n'ayons pas à gérer manuellement l'exécution séquentielle des transformations sur un ensemble de données test ou invisible.
Les étapes de prétraitement des données qui sont obligatoires pour l'apprentissage automatique telles que le codage des variables catégorielles, le codage des étiquettes et le 
train-test-split sont automatiquement exécutées lorsque la commande est initialisée.""")

############################################################################################################################################################################################

# creating image object 
#img1_B = Image.open(r"C:\Users\Emera\Web_app1\Images_B\Setup_env1.PNG")
img1_B = Image.open(path_image_B+"Setup_env1.PNG")  
# creating image2 object having alpha 
img2_B = Image.open(path_image_B+"Setup_env2.PNG") 
img2_B = img2_B.resize(img1_B.size) 
numpy_horizontal = np.hstack((img1_B,img2_B))
numpy_horizontal_concat = np.concatenate((img1_B,img2_B), axis=1)
st.image(numpy_horizontal, caption="Résultat de la configuration ", use_column_width=False)

############################################################################################################################################################################################

st.markdown("""Une fois la configuration exécutée, la grille d'informations contenant plusieurs informations importantes est imprimée. La plupart des informations sont liées au 
pipeline de prétraitement qui est construit lorsque **`setup()`** est exécuté. Cependant, quelques points importants sont à noter à ce stade :
- **session_id**: C'est un nombre pseudo-aléatoire distribué comme une graine dans toutes les fonctions pour une reproductibilité ultérieure. Dans cette expérience, **session_id** 
est défini sur *1234* pour une reproductibilité ultérieure.
- **Données d'origine**: Affiche la forme d'origine du jeu de données. Dans cette expérience *(13794, 16) * signifie qu'on dispose de *13794* entrées et *16* caractéristiques 
comprenant la colonne cible.
- **Valeurs manquantes**: Lorsqu'il y a des valeurs manquantes dans les données d'origine, cela apparaîtra comme vrai. Pour cette expérience, il n'y a aucune valeur manquante dans 
l'ensemble de données car elles ont été traitées en dehors de la configuration.
- **Caractéristiques numériques et catégoriques**: C'est le nombre d'entités déduites respectivement comme numériques et catégoriques. Dans cet ensemble de données, *7* entité sur
*16* sont déduites comme numérique et *8* entités sur *16* comme catégoriques.
- **Ensemble d'entrainement transformé**: Affiche la forme de l'ensemble d'entraînement transformé. Notez que la forme originale de *(13794, 16) * est transformée en *(11035, 35) 
* pour la rame transformée. Le nombre d'entités est passé de *16* à *35* en raison du codage catégoriel.
- **Ensemble de test transformé**: affiche la forme de l'ensemble de test transformé. Il y a *2759* entrées dans l'ensemble de test. Cette répartition est basée sur la valeur 
*80/20* à l'aide du paramètre ` train_size` dans la configuration.""") 

st.markdown(""" Etant donné que notre objectif principal ici est de construire un modèle capable de prédire les primes de bâtiment et contenu d'une couverture Bâtiment, nous avons 
séparé l'ensemble de données en entités explicatives et en variable cible. Les variables **nbpiece_Eth**, **capbat**, **capcont** et **franchise_amount** nous donnent des informations 
quantitatives sur les données. Les variables cibles **primebat** et **primecont** seront celles que nous cherchons à prédire.
Se familiariser avec les données par un processus exploratoire est une pratique fondamentale pour nous aider à mieux comprendre et justifier les résultats. Pour cela, nous calculons 
les statistiques descriptives sur les primes de bâtiments et de contenu avec la commande **`numpy()`**.Ces statistiques seront importantes ultérieurement pour analyser divers 
résultats de prédiction à partir du modèle construit.""")

############################################################################################################################################################################################

# TODO: Minimum price of the data
minimum_primebatB = round(np.min(Y_primebatB),3)
minimum_primecontB = round(np.min(Y_primecontB),3)
############################################
# TODO: Maximum price of the data
maximum_primebatB  = round(np.max(Y_primebatB),3)
maximum_primecontB  =round( np.max(Y_primecontB),3)
############################################
# TODO: Mean price of the data
mean_primebatB  = round(np.mean(Y_primebatB),3)
mean_primecontB  = round(np.mean(Y_primecontB),3)
############################################
# TODO: Median price of the data
median_primebatB =round( np.median(Y_primebatB),3)
median_primecontB =round( np.median(Y_primecontB),3)
############################################
# TODO: Standard deviation of prices of the data
std_primebatB  = round(np.std(Y_primebatB),3)
std_primecontB  = round(np.std(Y_primecontB),3)
############################################
# There are other statistics you can calculate too like quartiles
first_quartile_primebatB = round(np.percentile(Y_primebatB, 25),3)
first_quartile_primecontB = round(np.percentile(Y_primecontB, 25),3)
############################################################
third_quartile_primebatB = round(np.percentile(Y_primebatB, 75),3)
third_quartile_primecontB = round(np.percentile(Y_primecontB, 75),3)
inter_quartile_primebatB =round( third_quartile_primebatB - first_quartile_primebatB,3)
inter_quartile_primecontB = round(third_quartile_primecontB - first_quartile_primecontB,3)
#####################################################################################################
stat_primebatB= [minimum_primebatB,maximum_primebatB,mean_primebatB,median_primebatB,std_primebatB,first_quartile_primebatB,third_quartile_primebatB,inter_quartile_primebatB]
stat_primecontB= [minimum_primecontB,maximum_primecontB,mean_primecontB,median_primecontB,std_primecontB,first_quartile_primecontB,third_quartile_primecontB,inter_quartile_primecontB]
#####################################################################################################
table_statB= pd.DataFrame([stat_primebatB,stat_primecontB],columns= ['Minimum','Maximum','Moyenne','Médiane','Ecart type','Premier quartile','Second quartile','IQR'],
index= ['prime batiment','prime contenu'])
# Show the calculated statistics
st.write("**Statistiques de base de la cible**:") 
st.table(table_statB)


############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

st.subheader('2.1.Exploration graphique des données')

import seaborn as sns
clrB=['black', 'brown']
fig_B_C, axs_B_C = plt.subplots(1,2,figsize=(30,10))
for i, var in enumerate([Y_primebatB,Y_primecontB]):
    plt.subplot(121+i)
    sns.distplot(var, color=clrB[i])
    plt.axvline(var.mean(),color=clrB[i], linestyle='solid', linewidth=2)
    plt.axvline(var.median(), color=clrB[i], linestyle='dashed', linewidth=2)
    fig_B_C.suptitle('Histogrammes des variables cible primebat et primecont',size=20)
st.pyplot(fig_B_C)

##########################################################################################
##########################################################################################

clr1B = ['blue', 'purple']
fig1B, axs1B = plt.subplots(1,2,sharex=True,figsize=(30,10))
fig1B.suptitle('Histogrammes des variables nbpiece_Eth et capbat', size=20)
for i, var in enumerate(['nbpiece_Eth','capbat']):
    plt.subplot(121 + i)
    if i==0:
        sns.distplot(X_B[var],  color = clr1B[i])
        plt.axvline(X_B[var].mean(), color=clr1B[i], linestyle='solid', linewidth=2)
        plt.axvline(X_B[var].median(), color=clr1B[i], linestyle='dashed', linewidth=2)
    else:
        sns.distplot(np.log(X_B[var]), color = clr1B[i])
        plt.axvline(np.log(X_B[var]).mean(), color=clr1B[i], linestyle='solid', linewidth=2)
        plt.axvline(np.log(X_B[var]).median(), color=clr1B[i], linestyle='dashed', linewidth=2)       
st.pyplot(fig1B)

##########################################################################################
##########################################################################################

clr2B = ['orange','darkorange']
fig2B, axs2B = plt.subplots(ncols=2,figsize=(30,10))
for i, var in enumerate(['capcont','franchise_amount']):
    plt.subplot(121 + i)
    if i==0:
        sns.distplot(X_B[var],  color = clr2B[i])
        plt.axvline(X_B[var].mean(), color=clr2B[i], linestyle='solid', linewidth=2)
        plt.axvline(X_B[var].median(), color=clr2B[i], linestyle='dashed', linewidth=2)
    else:
        sns.distplot(np.log(X_B[var]), color = clr2B[i])
        plt.axvline(np.log(X_B[var]).mean(), color=clr2B[i], linestyle='solid', linewidth=2)
        plt.axvline(np.log(X_B[var]).median(), color=clr2B[i], linestyle='dashed', linewidth=2)
        fig2B.suptitle('Histogrammes des variables capcont et franchise_amount',size=20)
st.pyplot(fig2B)

st.markdown(""" Au regard des histogrammes ci-dessus, on voit que les variables présentées ont des distributions asymétriques plus étalée à gauche de la médiane. En effet, 
étant donné que la queue de la loi de distribution pointe vers la gauche et que leur valeur d'asymétrie est négative on peut donc en conclure que qu'il existe une dissymétrie 
entre la valeur de la variable et la distribution de la loi normale.""")
st.write('---')

############################################################################################################################################################################################
############################################################################################################################################################################################

st.markdown("""Par intuition, nous pouvons deviner le comportement de ces caractéristiques. En effet, 
* **nbpiece_Eth** le nombre de pièces dans le bâtiment se réfère à la taille de celui-ci. Nous supposons que plus il y'a de pièces, plus la prime devrait être élevé. 
* **capbat** et **capcont** se réfère à l'estimation du bâtiment et du contenu de celui-ci. Nous pouvons donc en déduire que plus leur montant est élevé, plus la prime le sera.
* **franchise_amount** se réfère au montant de la franchise que l'assuré consent à prendre à sa charge en cas d'incendie. Par conséquent, plus la franchise sera élevée moins 
importante sera la prime.""")

fig3B, axs3B = plt.subplots(1,2,figsize=(30,10))
for i, var in enumerate(['nbpiece_Eth','capbat']):
     sns.regplot(X_B[var], Y_primebatB, ax = axs3B[i], color=clr1B[i])
fig3B.suptitle('Relation entre les variables nbpiece_Eth et capbat et primebat',size=20)    
st.pyplot(fig3B)

########################################################################################
########################################################################################

fig4B, axs4B = plt.subplots(1,2,figsize=(30,10))
for i, var in enumerate(['nbpiece_Eth','capbat']):
    sns.regplot(X_B[var], Y_primecontB, ax = axs4B[i], color=clr1B[i])
fig4B.suptitle('Relation entre les variables nbpiece_Eth et capbat et primecont',size=20)    
st.pyplot(fig4B)

########################################################################################
########################################################################################

fig5B, axs5B = plt.subplots(1,2,figsize=(30,10))
for i, var in enumerate(['capcont','franchise_amount']):
    sns.regplot(X_B[var], Y_primebatB, ax = axs5B[i], color=clr2B[i])
    #lm.set(ylim=(0, None))
fig5B.suptitle('Relation entre les variables capcont et franchise_amount et primebat',size=20)
st.pyplot(fig5B)

########################################################################################
########################################################################################

fig6B, axs6B = plt.subplots(1,2,figsize=(30,10))
for i, var in enumerate(['capcont','franchise_amount']):
    sns.regplot(X_B[var], Y_primecontB, ax = axs6B[i], color=clr2B[i])
    #lm.set(ylim=(0, None))
fig6B.suptitle('Relation entre les variables capcont et franchise_amount et primecont',size=20)
st.pyplot(fig6B)

st.markdown(""" Nous avons construit des scatterplots pour voir si notre intuition est correcte. En effet, les graphes ci-dessus prouvent que notre intuition est la bonne pour la 
plupart des variables analysées excepté pour la franchise dont le graphe montre qu'elle n'a pas un impact conséquent sur la prime.""")
st.write('---')

############################################################################################################################################################################################
############################################################################################################################################################################################

#Created a dataframe without the price col, since we need to see the correlation between the variables
fig7B, axs7B = plt.subplots(1,1,figsize=(16,6))
axs7B = sns.heatmap(dataB_primebat_moy.corr().round(2), vmin=-1, vmax=1, annot=True, cmap='BrBG')
axs7B.set_title('carte thermique de corrélation entre les variables et la prime batiment', fontdict={'fontsize':18}, pad=12)
st.pyplot(fig7B)

fig8B, axs8B = plt.subplots(1,1,figsize=(16,6))
axs8B = sns.heatmap(dataB_primecont_moy.corr().round(2), vmin=-1, vmax=1, annot=True, cmap='BrBG')
axs8B.set_title('carte thermique de corrélation entre les variables et la prime contenu', fontdict={'fontsize':18}, pad=12)
st.pyplot(fig8B)

st.markdown("""Ce graphe montrant la corrélation entre les différentes variables confirme d'avantage notre intuition selon laquelle la prime est fortement corrélée au nombre de 
pièces du bâtiment considéré, à la valeur de ce dernier et à la valeur du contenu.
Cependant, contrairement à ce qu'on pouvait s'imaginer, la prime n'est que très faiblement corrélé au montant de la franchise bien que positive. Ceci pourrait s'expliquer du fait 
que comme la franchise est négativement liées aux variables fortement corrélées à la prime à savoir **capbat** et **capcont** ceci impacte implicitement sa relation avec la prime.
Cette carte nous permet de voir que la  prime contenu est d'avantage corelé aux variables **capbat** et **capcont** et **nbpiece_Eth** avec un coefficients de correlation avoisinant
les 80%  ce qui est légerement supérieur au coefficient de correlation des ces mêmes variables avec la prime bâtiment. 
On constate également que la franchise est négativement corelé avec la plus part des variables avec un coefficient de correlation le plus bas de *-0.09* avec la variable **abex** 
et allant jusqu'à *0.02* pour le coefficient le plus haut correspondant à sa relation avec la variable **classgeo**.
En ce qui concerne la variable **age**, comme on pouvait s'y attendre, elle a un coefficient de correlation relativement moyen avec les variables cibles et celles considérés comme
les principales. Notons également que cette carte nous confirme la correlation quasi parfaite entre les 3 variables principales(capbat,capcont,nbpieces_Eth) avec un coefficient de 
*98%*.""")
st.header("""------------------------------------------------------------""")

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

st.header('3. Comparaison de tous les modèles')

st.markdown(""" La comparaison de tous les modèles pour évaluer les performances est le point de départ recommandé pour la modélisation supervisée une fois la configuration 
terminée à l'aide de la commande **`compare_models()`** . Cette fonction entraîne tous les modèles de la bibliothèque de modèles à l'aide d’hyperparamètres et évalue les mesures 
de performance à l'aide de la validation croisée par blocs et renvoie l'objet du modèle entraîné. 
Le nombre de blocs est défini à l'aide du paramètre `fold ` (`fold = 10` par défaut) dans la fonction **`compare_models()`**. Les métriques d'évaluation utilisées dans notre cas 
sont ceux de la **regression** : MAE, MSE, RMSE, R2, RMSLE, MAPE.
La sortie de la fonction est une grille de score qui met en évidence par ordre décroissant les modèles et les métriques les plus performants à des fins de comparaison. Dans notre 
cas, la grille de score imprimée ci-dessous (celle par défaut) est triée en utilisant le coefficient de détermination (R2) (du plus élevé au plus bas) qui peut être modifié en 
passant le paramètre `sort`. Par défaut, **`compare_models()`**  renvoie le modèle le plus performant en fonction de l'ordre de tri par défaut, mais peut être utilisé pour renvoyer 
une liste des N meilleurs modèles à l'aide du paramètre `n_select`.""")

# creating image object 
img3_primebatB = Image.open(path_image_B+"Compare_models_result_B.PNG")   
st.image(img3_primebatB, caption="Grille de scores des modèles où primebat est la cible", use_column_width=False)

##########################################################################################
##########################################################################################

img3_primecontB = Image.open(path_image_B+"Compare_models_result_C.PNG")   
st.image(img3_primecontB, caption="Grille de scores des modèles où primecont est la cible", use_column_width=False)


st.markdown(""" Au regard de ce tableau,le modèle ayant les meilleures performances selon 3 métriques (MSE, RMSE,R2) est **CatBoostRegressor** aussi bien pour la prime bâtiment que 
contenu. Il faut quand même noter que si notre métrique de performance était le RMSLE alors le meilleur modèle serait **Gradient Boosting Regressor** et **Random Sample 
Consensus** ( pour la prime bâtiment ), **SVM** (pour la prime contenu) si la métrique de performance avait été le MAE. Nous avons donc présenté à la fin de ce rapport
les prédictions selon des modèles autre que catboost qui serait intéressant pour un assureur.""")
st.header("""------------------------------------------------------------""")

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# Print the predicted values for the prime batiment using the best model
st.header('4.Modélisation de la prime de Batiment : Meilleur modèle.')

st.markdown(""" Pour toute la suite de ce projet, nous présenterons les résultats obtenues avec le meilleur modèle obtenue(**CatBoost Regressor**) et nous effectuerons une 
analyse comparative par rapport au modèle de réference classiquement utilisé dans la prédiction à savoir les **GLM** afin d'évaluer l'apport des modèles de ML ainsi que des 
interactions de variables.""")
st.write('---')

st.subheader('4.1.Création du meilleur modèle: CatBoost Regressor')

st.markdown(""" Au moyen de la fonction **`create_model()`**, nous avons entrainé et évalué le modèle **CatBoost Regressor** à l'aide de la validation croisée par blocs (`fold = 10`). 
Cette fonction renvoie une table avec des métriques de performances validées croisées. Pour éviter le sur-apprentissage des modèles, nous avons réalisé un travail d’hyperparamètrage
sur ce modèle. Pour cela, nous avons recherché les hyperparamètres optimaux et avons par conséquent agit ceux qui suivent :
-  la métrique d'évaluation de la performance RMSE a été fixée à l'aide de l'hyperparamètre `eval_metric`.
- Le critère d’arrêt à été définit avec `itérations` pour que l’algorithme s’arrête bien après avoir créé *1000* modèles.
- Le taux d'apprentissage a été définit au moyen de l'hyperparamètre `learning_rate = 0.059843000024557114` il permet de réduire le pas lors de la mise en œuvre de la descente 
de gradient.
-  `l2_leaf_reg = 3` est le coefficient du terme de régularisation L2 de la fonction de coût. 
-  `depth = 6` est la profondeur de l'arbre.
-  `max_leaves = 64` est le nombre maximum de feuilles dans l'arbre résultant.
-  `border_count = 254` Le nombre de divisions pour les entités numériques.""")

# creating image object 
img4_primebatB  = Image.open(path_image_B+"Create_Catboost_Model_B.PNG")   
st.image(img4_primebatB , caption="Grille de scores du meilleur modèle primebat: CatBoost Regressor", use_column_width=False)

###########################################################################################################
###########################################################################################################

img4_primecontB = Image.open(path_image_B+"Create_Catboost_Model_C.PNG")   
st.image(img4_primecontB, caption="Grille de scores du meilleur modèle primecont: CatBoost Regressor", use_column_width=False)

###########################################################################################################
###########################################################################################################

st.write('Le modèle de référence (**GLM**) sur l\'ensemble de données **primebat** donne les résultats suivants:')
r_sq_lm_primebatB_test = load_lm_primebatB.score(X_test_B,Y_test_primebatB)
Y_pred_lm_primebatB = load_lm_primebatB.predict(X_test_B)
MAE_lm_primebatB_test=round(metrics.mean_absolute_error(Y_test_primebatB, Y_pred_lm_primebatB),4)
MSE_lm_primebatB_test=round(metrics.mean_squared_error(Y_test_primebatB, Y_pred_lm_primebatB),4)
RMSE_lm_primebatB_test=round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_lm_primebatB)),4)
R2_lm_primebatB_test=round(r_sq_lm_primebatB_test,4)
lm_primebatB_train_result= [21.1913,2976.9232 , 54.5031 ,0.7307]
lm_primebatB_test_result= [MAE_lm_primebatB_test,MSE_lm_primebatB_test,RMSE_lm_primebatB_test,R2_lm_primebatB_test]
lm_primebatB_table= pd.DataFrame([lm_primebatB_train_result, lm_primebatB_test_result], index= ['Train Set','Test Set'] ,columns = ['MAE','MSE', 'RMSE','R2'])
st.table(lm_primebatB_table)

###########################################################################################################
###########################################################################################################

st.write('Le modèle de référence (**GLM**) sur l\'ensemble de données **primecont** donne les résultats suivants:')
r_sq_lm_primecontB_test = load_lm_primecontB.score(X_test_B,Y_test_primecontB)
Y_pred_lm_primecontB = load_lm_primecontB.predict(X_test_B)
MAE_lm_primecontB_test=round(metrics.mean_absolute_error(Y_test_primecontB, Y_pred_lm_primecontB),4) 
MSE_lm_primecontB_test=round(metrics.mean_squared_error(Y_test_primecontB, Y_pred_lm_primecontB),4)
RMSE_lm_primecontB_test=round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_lm_primecontB)),4)
R2_lm_primecontB_test=round(r_sq_lm_primecontB_test,4)
lm_primecontB_train_result= [6.6611,280.6354,16.7296,0.6791]
lm_primecontB_test_result= [MAE_lm_primecontB_test,MSE_lm_primecontB_test,RMSE_lm_primecontB_test,R2_lm_primecontB_test]
lm_primecontB_table= pd.DataFrame([lm_primecontB_train_result, lm_primecontB_test_result], index= ['Train Set','Test Set'] ,columns = ['MAE','MSE', 'RMSE','R2'])
st.table(lm_primecontB_table)


###########################################################################################################
###########################################################################################################

st.write('Le meilleur modèle (**CatBoostRegressor**) sur l\'ensemble de données **primebat** données donne les résultats suivants:')
r_sq_cbr_primebatB_train = load_cbr_primebatB.score(X_train_B,Y_train_primebatB)
r_sq_cbr_primebatB_test = load_cbr_primebatB.score(X_test_B,Y_test_primebatB)
Y_pred_load_cbr_primebatB = load_cbr_primebatB.predict(X_test_B)
MAE_cbr_primebatB_test=round(metrics.mean_absolute_error(Y_test_primebatB, Y_pred_load_cbr_primebatB),4)
MSE_cbr_primebatB_test=round(metrics.mean_squared_error(Y_test_primebatB, Y_pred_load_cbr_primebatB),4)
RMSE_cbr_primebatB_test=round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_load_cbr_primebatB)),4)
R2_cbr_primebatB_test=round(r_sq_cbr_primebatB_test,4)
cbr_train_result_primebatB= [19.1576, 2777.9120, 52.6540 ,0.7485]
cbr_test_result_primebatB= [MAE_cbr_primebatB_test,MSE_cbr_primebatB_test,RMSE_cbr_primebatB_test,R2_cbr_primebatB_test]
cbr_table_primebatB= pd.DataFrame([cbr_train_result_primebatB, cbr_test_result_primebatB], index= ['Train Set','Test Set'] ,columns = ['MAE','MSE', 'RMSE','R2'])
st.table(cbr_table_primebatB)

###########################################################################################################
###########################################################################################################

st.write('Le meilleur modèle (**CatBoostRegressor**) sur l\'ensemble de données **primecont** donne les résultats suivants:')
r_sq_cbr_primecontB_train = load_cbr_primecontB.score(X_train_B,Y_train_primecontB)
r_sq_cbr_primecontB_test = load_cbr_primecontB.score(X_test_B,Y_test_primecontB)
Y_pred_load_cbr_primecontB = load_cbr_primecontB.predict(X_test_B)
MAE_cbr_primecontB_test=round(metrics.mean_absolute_error(Y_test_primecontB, Y_pred_load_cbr_primecontB),4)
MSE_cbr_primecontB_test=round(metrics.mean_squared_error(Y_test_primecontB, Y_pred_load_cbr_primecontB),4)
RMSE_cbr_primecontB_test=round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_load_cbr_primecontB)),4)
R2_cbr_primecontB_test=round(r_sq_cbr_primecontB_test,4)
cbr_train_result_primecontB= [5.9629 ,264.3183 ,16.2351, 0.6976]
cbr_test_result_primecontB= [MAE_cbr_primecontB_test,MSE_cbr_primecontB_test,RMSE_cbr_primecontB_test,R2_cbr_primecontB_test]
cbr_table_primecontB= pd.DataFrame([cbr_train_result_primecontB, cbr_test_result_primecontB], index= ['Train Set','Test Set'] ,columns = ['MAE','MSE', 'RMSE','R2'])
st.table(cbr_table_primecontB)


st.markdown(""" Le modèle explique environ * 74 % * de la variation de la cible **primebat** et environ *69 %* de la variable de la cible **primecont**, Le coefficient de 
determination indiquant que notre modèle est bien ajusté aux données.Ce modèle nous donne des résultats légèrement meilleurs sur la base d’apprentissage mais considérable sur la base de test, par rapport 
à ceux donnés par le modèle de réference.""")

st.write(f'Les résultats du **RMSE** sur la base de test signifie que le modèle, en moyenne, a une erreur de prédiction de la prime bâtiment médiane de {round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_load_cbr_primebatB)),4)} (fois 100 €)'+
f'et une erreur de prédiction de la prime contenu médiane de {round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_load_cbr_primecontB)),4)} (fois 100 €).')
st.write(f'Ce modèle obtient pour la prime bâtiment {round(abs(((round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_load_cbr_primebatB)),2)- round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_lm_primebatB)),2 ))/ round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_lm_primebatB)),2))*100),2)} % '+
f'et pour la prime contenu {round(abs(((round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_load_cbr_primecontB)),2)- round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_lm_primecontB)),2 ))/ round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_lm_primecontB)),2))*100),2)} % de meilleurs résultats sur les données de test que le modèle de réference.')
st.write('---')

############################################################################################################################################################################################

st.subheader('4.2. Tuning du modèle')

st.markdown(""" En Tunant le modèle à l'aide de la commande **`tune_model()`** et en modifiant  la profondeur(`depth = 9`) ,le learning rate(`0.029999999329447743`), 
le max_leaves(`512`), le l2_leaf_reg (`10`) et le border_count (`100`), nous avons légerement amélioré la performance du modèle passant ainsi d'un R2 de * 74.85 % * à * 74.92 % * pour 
la prime bâtiment et de * 69.76 % * à * 69.80 % * pour la prime contenu.
les résultats du modèle tuné sont présentés dans le tableau ci dessous:""")

# creating image object 
img5_primebatB = Image.open(path_image_B+"Tuned_Catboost_Model_B.PNG")   
st.image(img5_primebatB, caption="Grille de scores du  meilleur modèle primebat: Tuned CatBoost Regressor", use_column_width=False)
st.write('---')

# creating image object 
img5_primecontB = Image.open(path_image_B+"Tuned_Catboost_Model_C.PNG")   
st.image(img5_primecontB, caption="Grille de scores du  meilleur modèle primecont: Tuned CatBoost Regressor", use_column_width=False)

Y_pred_load_cbr_tuned_primebatB = load_cbr_tuned_primebatB.predict(X_test_B)
Y_pred_load_cbr_tuned_primecontB = load_cbr_tuned_primecontB.predict(X_test_B)

st.write(f'Le modèle tuné permet ainsi une considérable amélioration en effet, on obtient pour la prime bâtiment {round(abs(((round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_load_cbr_tuned_primebatB)),2)- round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_lm_primebatB)),2 ))/ round(np.sqrt(metrics.mean_squared_error(Y_test_primebatB, Y_pred_lm_primebatB)),2))*100),2)} % '+
f'et pour la prime contenu {round(abs(((round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_load_cbr_tuned_primecontB)),2)- round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_lm_primecontB)),2 ))/ round(np.sqrt(metrics.mean_squared_error(Y_test_primecontB, Y_pred_lm_primecontB)),2))*100),2)} % de meilleurs résultats sur les données de test que le modèle de réference.')
st.write('---')

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

# Print specified input parameters
st.subheader('4.3. Les modèles d\'ensembles')

st.subheader('a. Le bagging')
st.markdown(""" Egalement connu sous le nom d'agrégation Bootstrap, il consiste à sous-échantilloner (ou ré-échantilloner au hasard avec doublons) le training set et de faire générer à l’algorithme un modèle pour 
chaque sous-échantillon. On obtient ainsi un ensemble de modèles dont il convient de moyenner (lorsqu’il s’agit d’une régression) ou de faire voter (pour une classification) les différentes prédictions.
C'est un méta-algorithme d'ensemble d'apprentissage automatique conçu pour améliorer la stabilité et la précision des algorithmes. Cela réduit également la variance et permet d'éviter le surajustement ; c’est donc 
un cas particulier de l'approche de la moyenne du modèle.""")

# creating image object 
img6_primebatB = Image.open(path_image_B+"Catboost_Bagged_B.PNG")   
st.image(img6_primebatB, caption="Grille de scores du  meilleur modèle primebat: Bagged CatBoost Regressor", use_column_width=False)

img6_primecontB= Image.open(path_image_B+"Catboost_Bagged_C.PNG")   
st.image(img6_primecontB, caption="Grille de scores du  meilleur modèle primecont: Bagged CatBoost Regressor", use_column_width=False)

st.markdown(""" Le Bagging est le seul modèle d'ensemble qui permet dans notre cas d'améliorer de façon considérable les performances du modèle. Le modèle explique dès lors * 75.39 % * de la variation de la prime 
batiment et *70.29%* pour la prime contenu.""")
st.write('---')

############################################################################################################################################################################################

st.subheader('b. Le Boosting')
st.markdown(""" Le principe du boosting est quelque peu différent du bagging. Les différents régresseurs sont pondérés de manière à ce qu’à chaque prédiction, les regresseurs ayant prédit correctement auront un poids plus 
fort que ceux dont la prédiction est incorrecte.

C'est donc un méta-algorithme d'ensemble destiné principalement à réduire les biais et la variance dans l'apprentissage supervisé. Il fait partie de la famille des algorithmes d'apprentissage automatique 
qui convertissent les apprenants faibles en apprenants forts. Un apprenant faible est défini comme un régresseur qui n'est que légèrement corrélé avec la vraie régression. En revanche, un apprenant fort est un 
régresseur qui est arbitrairement bien corrélé avec la vraie régression.

Adaboost est un algorithme de boosting qui s’appuie sur ce principe, avec un paramètre de mise à jour adaptatif permettant de donner plus d’importance aux valeurs difficiles à prédire, donc en boostant les
régresseurs qui réussissent quand d’autres ont échoué. Adaboost s’appuie sur des régresseurs existants et cherche à leur affecter les bons poids vis à vis de leurs performances.""")

# creating image object 
img7_primebatB = Image.open(path_image_B+"Catboost_Boosted_B.PNG")   
st.image(img7_primebatB, caption="Résultat meilleur modèle primebat: Boosted CatBoost Regressor", use_column_width=False)


img7_primecontB = Image.open(path_image_B+"Catboost_Boosted_C.PNG")   
st.image(img7_primecontB, caption="Résultat meilleur modèle primecont: Boosted CatBoost Regressor", use_column_width=False)
st.write('---')


############################################################################################################################################################################################

st.subheader('c. Le Blending')

st.markdown("""Le Blending est une méthode d'assemblage qui utilise le consensus entre les estimateurs pour générer des prédictions finales. L'idée derrière est de combiner différents algorithmes 
d'apprentissage automatique et d'utiliser un vote majoritaire ou les probabilités moyennes prévues en cas de classification pour prédire le résultat final.""")

# creating image object 
img8_primebatB = Image.open(path_image_B+"Catboost_Blended_B.PNG")   
st.image(img8_primebatB, caption="Résultat meilleur modèle primebat: Blended CatBoost Regressor", use_column_width=False)


img8_primecontB = Image.open(path_image_B+"Catboost_Blended_C.PNG")   
st.image(img8_primecontB, caption="Résultat meilleur modèle primecont: Blended CatBoost Regressor", use_column_width=False)
st.write('---')


############################################################################################################################################################################################

st.subheader('d. Le Stacking')

st.markdown("""Le Stacking est une méthode d'assemblage qui utilise le méta-apprentissage. L'idée derrière est de construire un méta-modèle qui génère la prédiction finale en utilisant 
prédiction de plusieurs estimateurs de base. Cette fonction prend une liste de modèles entraînés à l'aide du paramètre `estimator_list`. Tous ces modèles forment dès lors la couche 
de base du Staking et leurs prédictions sont utilisées comme entrée pour un méta-modèle qui peut être transmis à l'aide du paramètre `meta_model`. Si aucun 
méta-modèle n'est passé, un modèle linéaire est utilisé par défaut.""")

st.markdown(""" Toutes ces fonctions renvoient une table avec des scores croisés validés (10 blocs) de mesures d'évaluation communes avec un objet de modèle entraîné.""")

# creating image object 
img9_primebatB = Image.open(path_image_B+"Catboost_Stacked_B.PNG")   
st.image(img9_primebatB, caption="Résultat meilleur modèle primebat: Stacked CatBoost Regressor", use_column_width=False)


img9_primecontB = Image.open(path_image_B+"Catboost_Stacked_C.PNG")   
st.image(img9_primecontB, caption="Résultat meilleur modèle primecont: Stacked CatBoost Regressor", use_column_width=False)
st.header("""------------------------------------------------------------""")

############################################################################################################################################################################################
############################################################################################################################################################################################

# Print specified input parameters
st.subheader('5.Specification des paramètres d\'entrée') 

st.markdown("""Le tableau de bord présent à notre gauche nous permet de spécifier les caractériques voulues d'un assuré quelconque afin d'en obtenir la prédiction de selon notre meilleur modèle. La table ci dessous 
retrace dès lors l'ensemble des paramètres en entrée. pour le cas présent, notre client agé de 41 ans, est propriétaire d'un appartement une pièce sans garage dont le capital batiment est de 194.935 € ,le capital 
contenu est de 64.058€ et le montant de la franchise de 261.73. D'autres paramètres tels que le zip,daconri2 ou encore le CDPISEV peuvent aussi être spécifiés. """)
st.write(datae_B)
st.write('---')

############################################################################################################################################################################################
############################################################################################################################################################################################

import streamlit.components.v1 as components
# Print the predicted values for the prime batiment using the best model
st.subheader('5.1.Prédiction de la prime bâtiment et explication utilisant le meilleur modèle: CatBoostRegressor')

version1 = ['Model CatBoostRegressor', 'Model CatBoostRegressor tuned']
regressor1 = st.selectbox('Choix de la version d\'algorithme', version1)
           
if regressor1 == 'Model CatBoostRegressor':
            # Apply Model to Make Prediction
            prediction_cbr_primebatB = load_cbr_primebatB.predict(df_primebatB)
            prediction_cbr_primebatB = str(round(prediction_cbr_primebatB[0],2)) + ' €'
            st.success('La **prime batiment** de l\'assuré prédite par le  modèle **CatBoostRegressor** est  {}'.format(prediction_cbr_primebatB))            
            #st.write('---') 
            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)
            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_cbr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
elif regressor1 == 'Model CatBoostRegressor tuned':
            # Apply Model to Make Prediction
            prediction_cbr_tuned_primebatB = load_cbr_tuned_primebatB.predict(df_primebatB)
            prediction_cbr_tuned_primebatB = str(round(prediction_cbr_tuned_primebatB[0],2)) + ' €'
            st.success('La **prime batiment** de l\'assuré prédite par le  modèle **CatBoostRegressor tuné ** est  {}'.format(prediction_cbr_tuned_primebatB))            
            #st.write('---') 
            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)
            # explain the model's predictions using SHAP
            explainer_tuned_primebatB = shap.TreeExplainer(load_cbr_tuned_primebatB)
            shap_values_df_tuned_primebatB = explainer_tuned_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_tuned_primebatB.expected_value,shap_values_df_tuned_primebatB,df_primebatB))

st.write(f'Nous avons prédit une prime de {prediction_cbr_primebatB} alors que la valeur de base moyenne est 184.8 €. Les valeurs d\'entités entraînant une augmentation des prévisions sont en rose et '+
'leur taille visuelle indique l\'ampleur de l\'effet de l\'entité. Les valeurs des caractéristiques diminuant la prédiction sont en bleu. Le plus gros impact vient du fait'+
' que la variable **daconri2** est 1719, que l\'assuré soit proprétaire ou locataire,en fonction de la valeur du batiment ou encore que le type d\'habitation2 soit un SO; et tout ceci'+
'bien que le nombre de pièces,le code zip, le type d\'habitation M4 ou encore l\'abex aient un effet  significatif sur la diminution de cette prédiction.')

st.markdown(""" Dans le graphique ci-dessus, La distance entre la valeur de base et la sortie est obtenue en  soustrayant la longueur des barres bleues de celle des barres roses.
Les valeurs des caractéristiques sont affichées et les shap values sont représentées par la longueur de la barre spécifique.  Cependant, la valeur exacte de chacunes d'elles n'est pas 
tout à fait claire, cela peut être vu ci-dessous, si on le souhaite.""")
shap_table_primebatB=pd.DataFrame(shap_values_df_primebatB,columns=df_primebatB.columns)
if st.button ('Cliquez ici pour un aperçu des SHAP values primebat'):
        # Prepare Data
        shap_table_primebat1B = shap_table_primebatB.T
        shap_table_primebat1B.columns =['shap_value_primebatB'] 
        x_Bb = shap_table_primebat1B.loc[:, ['shap_value_primebatB']]
        shap_table_primebat1B['colors'] = ['dodgerblue' if x_Bb < 0 else 'deeppink' for x_Bb in shap_table_primebat1B['shap_value_primebatB']]
        # Draw plot
        fig9B,axs9B= plt.subplots(1,1,figsize=(10,10), dpi= 80)
        axs9B=plt.hlines(y=shap_table_primebat1B.index, xmin=0, xmax=shap_table_primebat1B.shap_value_primebatB, color=shap_table_primebat1B['colors'], alpha=1, linewidth=5)
        for x_Bb,y_Bb,tex_Bb in zip(shap_table_primebat1B.shap_value_primebatB,shap_table_primebat1B.index,shap_table_primebat1B.shap_value_primebatB):
            t_Bb = plt.text(x_Bb, y_Bb, round(tex_Bb, 2), horizontalalignment='right' if x_Bb < 0 else 'left', 
                    verticalalignment='center', fontdict={'color':'dodgerblue' if x_Bb < 0 else 'deeppink', 'size':10})
        # Decorations    
        plt.title('Barsplot Divergent des Shap values', fontdict={'size':10})
        plt.grid(linestyle='--', alpha=0.5)
        st.pyplot(fig9B)
        plt.clf()    
st.write('---')

############################################################################################################################################################################################

st.subheader('Importance des variables par rapport à la cible primebat')

st.markdown("""Les SHAP values donnent également des détails robustes, parmi lesquels l\'importance des variables. AContrairement au précedent graphe qui nous montrent quelles variables impactent la prédiction au 
niveau d'un assuré,le graphe ci dessous nous permet de visualiser les variables qui impactent de façon globale l'ensemble de notre base de données.""")
fig10B, axs10B= plt.subplots(1,1,figsize=(16,6))
axs10B = interpret_model(load_cbr_primebatB)
plt.title ('Importance des variables sur base de la Shap values (primebat)',fontdict={'size':10})
st.pyplot(fig10B)
#st.write('---')


st.write('Ce graphe peut également être représenté sous forme de barplots pour plus de visibilité.')
fig11B, axs11B = plt.subplots(1,1,figsize=(16,6))
axs11B = interpret_model(load_cbr_primebatB,plot_type="bar")
plt.title ('Importance des variables sur base des moyennes de shap values (primebat)',fontdict={'size':10})
st.pyplot(fig11B)
st.write('---')

###########################################################################################################################################
###########################################################################################################################################

st.subheader('5.2.Prédiction de la prime contenu et explication utilisant le meilleur modèle: CatBoostRegressor')

version2 = ['CatBoostRegressor', 'Model CatBoostRegressor ']
regressor2 = st.selectbox('Choix de la version d\'algorithme', version2)
            
if regressor2 == 'CatBoostRegressor':
            # Apply Model to Make Prediction
            prediction_cbr_primecontB = load_cbr_primecontB.predict(df_primebatB)
            prediction_cbr_primecontB = str(round(prediction_cbr_primecontB[0],2)) + ' €'
            st.success('La **prime contenu** de l\'assuré prédite par le  modèle **CatBoostRegressor** est  {}'.format(prediction_cbr_primecontB))            
            #st.write('---') 
            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)
            # explain the model's predictions using SHAP
            explainer_primecontB = shap.TreeExplainer(load_cbr_primecontB)
            shap_values_df_primecontB = explainer_primecontB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primecontB.expected_value,shap_values_df_primecontB,df_primebatB))
elif regressor2 == ' CatBoostRegressor tuned':
            # Apply Model to Make Prediction
            prediction_cbr_tuned_primecontB = load_cbr_tuned_primecontB.predict(df_primebatB)
            prediction_cbr_tuned_primecontB = str(round(prediction_cbr_tuned_primecontB[0],2)) + ' €'
            st.success('La **prime batiment** de l\'assuré prédite par le  modèle **CatBoostRegressor tuné ** est  {}'.format(prediction_cbr_tuned_primecontB))            
            #st.write('---') 
            def st_shap(plot, height=None):
                shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                components.html(shap_html, height=height)
            # explain the model's predictions using SHAP
            explainer_tuned_primecontB = shap.TreeExplainer(load_cbr_tuned_primecontB)
            shap_values_df_tuned_primecontB = explainer_tuned_primecontB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_tuned_primecontB.expected_value,shap_values_df_tuned_primecontB,df_primebatB))
st.write(f'Tout comme précédemment,nous avons prédit une prime de {prediction_cbr_primecontB} alors que la valeur de base est 58.3 €.Le plus gros impact  demeure du fait'+
' que la variable **daconri2** est 1719.')
            
shap_table_primecontB=pd.DataFrame(shap_values_df_primecontB,columns=df_primebatB.columns)
if st.button ('Cliquez ici pour un aperçu des SHAP values primecont'):
    # Prepare Data
    shap_table_primecont1B = shap_table_primecontB.T
    shap_table_primecont1B.columns =['shap_value_primecontB'] 
    x_Bc = shap_table_primecont1B.loc[:, ['shap_value_primecontB']]
    shap_table_primecont1B['colors'] = ['dodgerblue' if x_Bc < 0 else 'deeppink' for x_Bc in shap_table_primecont1B['shap_value_primecontB']]
    # Draw plot
    fig12B,axs12B= plt.subplots(1,1,figsize=(10,10), dpi= 80)
    axs12B=plt.hlines(y=shap_table_primecont1B.index, xmin=0, xmax=shap_table_primecont1B.shap_value_primecontB, color=shap_table_primecont1B['colors'], alpha=1, linewidth=5)
    for x_Bc,y_Bc,tex_Bc in zip(shap_table_primecont1B.shap_value_primecontB,shap_table_primecont1B.index,shap_table_primecont1B.shap_value_primecontB):
        t_Bc = plt.text(x_Bc, y_Bc, round(tex_Bc, 2), horizontalalignment='right' if x_Bc < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'dodgerblue' if x_Bc < 0 else 'deeppink', 'size':10})
    # Decorations    
    plt.title('Barsplot Divergent des Shap values', fontdict={'size':10})
    plt.grid(linestyle='--', alpha=0.5)
    st.pyplot(fig12B)
    plt.clf()
st.write('---')

############################################################################################################################################################################################

st.subheader('Importance des variables par rapport à la cible primecont')

fig13B, axs13B= plt.subplots(1,1,figsize=(16,6))
axs13B = interpret_model(load_cbr_primecontB)
plt.title ('Evaluation de l\'importance des variables sur base de la Shap values (primecont)',fontdict={'size':10})
st.pyplot(fig13B)
#st.write('---')


fig14B, axs14B = plt.subplots(1,1,figsize=(16,6))
axs14B = interpret_model(load_cbr_primecontB,plot_type="bar")
plt.title ('Evaluation de l\'importance des variables sur base des moyennes de shap values (primecont)',fontdict={'size':10})
st.pyplot(fig14B)
st.header("""------------------------------------------------------------""")

############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

st.subheader('6.Développer une compréhension plus approfondie des données primebat à l\'aide de SHAP: effets d\'interaction')
st.markdown(""" En sélectionnant des entités ci-dessous, l'algorithme trace automatiquement l'entité sélectionnée avec l'entité avec laquelle elle interagit le plus probablement. 
Cependant, le jugement final réside dans les yeux du spectateur. En règle générale, lorsqu'il y a un effet d'interaction, les points divergent fortement. """)

shape_value_primebatB_xtrain = explainer_primebatB.shap_values(X_train_B)
st.write('In the slider below, select the number of features to inspect for possible interaction effects.'
             'These are ordered based on feature importance in the model.')
rangesB = st.number_input('Please select the number of features',min_value=min(range(len(X_train_B.columns)))+1, max_value=max(range(len(X_train_B.columns)))+1,value=1)
if rangesB-1 == 0:
        st.write('you have selected the most importance feature')
elif rangesB == len(X_train_B.columns):
        st.write('you have selected all possible features')
else:
        st.write('you have selected the top:',rangesB,'important features')
for rank in range(rangesB):
            ingest_primebatB=('rank('+str(rank)+')')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig13B, axs13B = plt.subplots(1,1,figsize=(16,6))
            axs13B=shap.dependence_plot(ingest_primebatB,shape_value_primebatB_xtrain,X_train_B,show=False)
            st.pyplot(axs13B)
            plt.clf()
st.header("""------------------------------------------------------------""")
     
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################

st.subheader('7.Prédiction obtenue suivant d\'autres modèles par ordre décroissant ')

alg = ['Light Gradient Boosting Machine', 'Gradient Boosting Machine',
'Random Forest Regressor','Extrem Gradient Boosting','Support Vector Machine',
'Multi Level Perceptron Regressor/Neural Network','Decision Tree Regressor']
regressor = st.selectbox('Choix de l\'algorithme', alg)

            
if regressor == 'Light Gradient Boosting Machine':
            # Apply Model to Make Prediction
            prediction_lgbmr_primebatB = load_lgbmr_primebatB.predict(df_primebatB)
            prediction_lgbmr_primebatB = str(round(prediction_lgbmr_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_lgbmr_primebatB))            

            #st.write('---')
            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_lgbmr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()
elif regressor == 'Gradient Boosting Machine':
            # Apply Model to Make Prediction
            prediction_gbmr_primebatB = load_gbmr_primebatB.predict(df_primebatB)
            prediction_gbmr_primebatB = str(round(prediction_gbmr_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_gbmr_primebatB))            

            #st.write('---')
            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_gbmr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()
elif regressor == 'Extrem Gradient Boosting':
            # Apply Model to Make Prediction
            prediction_xgb_primebatB = load_xgb_primebatB.predict(df_primebatB)
            prediction_xgb_primebatB = str(round(prediction_xgb_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_xgb_primebatB))            

            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_xgb_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()
elif regressor == 'Random Forest Regressor':
            # Apply Model to Make Prediction
            prediction_rfr_primebatB = load_rfr_primebatB.predict(df_primebatB)
            prediction_rfr_primebatB = str(round(prediction_rfr_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_rfr_primebatB))            

            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_rfr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')           
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()            
elif regressor == 'Support Vector Machine':
            # Apply Model to Make Prediction
            prediction_svmr_primebatB = load_svmr_primebatB.predict(df_primebatB)
            prediction_svmr_primebatB = str(round(prediction_svmr_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_svmr_primebatB))            

            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_svmr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()
elif regressor == 'Multi Level Perceptron Regressor/Neural Network':
            # Apply Model to Make Prediction
            prediction_mlpr_primebatB = load_mlpr_primebatB.predict(df_primebatB)
            prediction_mlpr_primebatB = str(round(prediction_mlpr_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_mlpr_primebatB))            

            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_mlpr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')  
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()

elif regressor == 'Decision Tree Regressor':
            # Apply Model to Make Prediction
            prediction_dtr_primebatB = load_dtr_primebatB.predict(df_primebatB)
            prediction_dtr_primebatB = str(round(prediction_dtr_primebatB[0],2)) + ' €'
            st.success('La prime de l\'assuré prédite par le  modèle est  {}'.format(prediction_dtr_primebatB))            

            def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)

            # explain the model's predictions using SHAP
            explainer_primebatB = shap.TreeExplainer(load_dtr_primebatB)
            shap_values_df_primebatB = explainer_primebatB.shap_values(df_primebatB)
            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            st_shap(shap.force_plot(explainer_primebatB.expected_value,shap_values_df_primebatB,df_primebatB))
            #st.write('---')
            # une alternative au précédent code est celui ci:
            #fig8, axs= plt.subplots(1,1,figsize=(16,6))
            #axs=shap.force_plot(explainer_primebat.expected_value, shap_values_df_primebat,df_primebat,matplotlib=True,show=False)
            #st.pyplot(axs)
            #plt.clf()


############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################
############################################################################################################################################################################################


# st.subheader('7.Understanding groups: t-SNE cluster analysis')

# st.markdown("""Vous trouverez ci-dessous un diagramme de grappes T-SNE interactif.Ici, on peut obtenir des informations significatives sur différents groupes, 
# par exemple pour un ciblage et une communication dans un contexte marketing.

# Notez que la variable cible est regroupée ici pour une meilleure interprétabilité, chaque couleur distincte représente un quantile spécifique, où le rouge signale les 25 % plus élevés des valeurs médianes des 
# maisons à Boston.""")

# from sklearn.manifold import TSNE
# shap_embedded_primebatB = TSNE(n_components=2, perplexity=25,random_state=34).fit_transform(shape_value_primebatB_xtrain)
# shap_embedded = TSNE(n_components=2, perplexity=35,random_state=34).fit_transform(shap_values)
# source_primebatB =X_train_B.copy()
# source_primebatB .insert(len(source_primebatB .columns), "TARGET", Y_train_primebatB)
# source_primebatB .insert(len(source_primebatB .columns), "TSNE-1", shap_embedded_primebatB [:,0])
# source_primebatB .insert(len(source_primebatB .columns), "TSNE-2", shap_embedded_primebatB [:,1])
# source_primebatB .insert(len(source_primebatB .columns), "SHAP_C", shape_value_primebatB_xtrain.sum(1).astype(np.float64))
# bins = [0,25, 50, 75, 100]
# labels = ['lowest 25%','25 to 50%','50-75%','highest 25%']
# source_primebatB['TARGET_BINNED'] = pd.cut(source_primebatB['TARGET'], bins=4,labels=labels).astype(str)
# brush = alt.selection(type='interval',resolve='global')
#            x='TSNE-1:Q',
#            y='TSNE-2:Q',
#            color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
#     ).add_selection(
#            brush
#     ).properties(
#         width=500,
#         height=250
#     )

# points_nbpiece_Eth_primebatB  = alt.Chart(source_primebatB).mark_point().encode(
#          x='nbpiece_Eth:Q',
#          y='TARGET:Q',
#          color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
#      ).add_selection(
#          brush
#      ).properties(
#         width=500,
#         height=250
#     )
# points_capbat_primebatB  = alt.Chart(source_primebatB).mark_point().encode(
#         y='TARGET:Q',
#         color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray')),
#     ).add_selection(
#         brush
#     ).properties(
#         width=500,
#         height=250
#     )
# points_capcont_primebatB  = alt.Chart(source_primebatB ).mark_point().encode(
#         x='capcont:Q',
#         color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
#     ).add_selection(
#         brush
#     ).properties(
#        width=500,
#         height=250
#     )
# 
# st.altair_chart(points_TSNE_primebatB & points_nbpiece_Eth_primebatB & points_capbat_primebatB & points_capcont_primebatB )
# st.header("""------------------------------------------------------------""")

############################################################################################################################################################################################
#############









































