#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Loads the PRIME BATIMENT Dataset
# Faire une liste de types de valeurs manquantes
path_B = 'C:/Users/emera/Web_app1/' 
path_model_B = 'C:/Users/emera/Web_app1/model_pkl_B/'
missing_values = ["n/a", "na", "--"]
data_B = pd.read_csv(path_B+'data_B0.csv', sep=';',decimal=",",encoding = "ISO-8859-1",engine='python', na_values = missing_values)
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


# In[3]:


dataB_primebat_moy=data_B_moy.drop(['primecont'], axis=1)
dataB_primecont_moy=data_B_moy.drop(['primebat'], axis=1)


# In[4]:


# random state
SEED=1234
from pycaret.regression import * # on importe toute les fonction associée à la regression depuis pycaret step1
my_setup_primebatB = setup(data =dataB_primebat_moy, target = 'primebat', train_size=0.8, silent=True, session_id=SEED) # setup permet d'obtenir une description generale du dataset 


# In[5]:


my_setup_primecontB = setup(data =dataB_primecont_moy, target = 'primecont', train_size=0.8, silent=True, session_id=SEED) # setup permet d'obtenir une description generale du dataset 


# In[6]:


# Build machine learning model

# Build Linear Regression model
from sklearn.linear_model import LinearRegression
lm_primebatB = LinearRegression()
lm_primecontB = LinearRegression()
#lm.fit(X,Y)


# In[7]:


# Build CatBoostRegressor model
from catboost import CatBoostRegressor
cbr_primebatB = CatBoostRegressor(random_seed=1234)
cbr_primecontB = CatBoostRegressor(random_seed=1234)
cbr_tuned_primebatB=CatBoostRegressor(l2_leaf_reg=10, random_seed=1234,depth=9,border_count=100,learning_rate=0.029999999329447743,max_leaves=512)
cbr_tuned_primecontB=CatBoostRegressor(l2_leaf_reg=10, random_seed=1234,depth=9,border_count=100,learning_rate=0.029999999329447743,max_leaves=512)
#cbr.fit(X,Y)


# In[8]:


# Build light gradient boosting machine regressor model
import lightgbm 
lgbmr_primebatB =lightgbm.LGBMRegressor(random_seed=1234)
lgbmr_primecontB =lightgbm.LGBMRegressor(random_seed=1234)
#lgbmr.fit(X,Y)


# In[9]:


# Build gradient boosting machine regressor model
from sklearn.ensemble import GradientBoostingRegressor
gbmr_primebatB = GradientBoostingRegressor()
gbmr_primecontB = GradientBoostingRegressor()
#gbmr.fit(X,Y)


# In[10]:


# Build RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor 
rfr_primebatB = RandomForestRegressor()
rfr_primecontB = RandomForestRegressor()
#rfr.fit(X,Y)


# In[11]:


# Build extrem gradient boosting regressor model
import xgboost
xgb_primebatB = xgboost.XGBRegressor(random_seed=1234)
xgb_primecontB = xgboost.XGBRegressor(random_seed=1234)
#xgb.fit(X,Y)


# In[12]:


# Build support vector machine  model
from sklearn import svm
svmr_primebatB = svm.SVR()
svmr_primecontB = svm.SVR()
#svmr.fit(X,Y)


# In[13]:


# Build multi level perceptron regressor model (Neural network multi layer)
from sklearn.neural_network import MLPRegressor
mlpr_primebatB = MLPRegressor()
mlpr_primecontB = MLPRegressor()
#mlpr.fit(X,Y)


# In[14]:


# Build decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
dtr_primebatB = DecisionTreeRegressor()
dtr_primecontB = DecisionTreeRegressor()
#dtr.fit(X,Y)


# In[15]:


# Saving  models
import pickle
pickle.dump(lm_primebatB, open(path_model_B+'lm_primebatB.pkl', 'wb'))
pickle.dump(lm_primecontB, open(path_model_B+'lm_primecontB.pkl', 'wb'))

pickle.dump(cbr_primebatB, open(path_model_B+'cbr_primebatB.pkl', 'wb'))
pickle.dump(cbr_primecontB, open(path_model_B+'cbr_primecontB.pkl', 'wb'))

pickle.dump(cbr_tuned_primebatB, open(path_model_B+'cbr_tuned_primebatB.pkl', 'wb'))
pickle.dump(cbr_tuned_primecontB, open(path_model_B+'cbr_tuned_primecontB.pkl', 'wb'))

pickle.dump(lgbmr_primebatB, open(path_model_B+'lgbmr_primebatB.pkl', 'wb'))
pickle.dump(lgbmr_primecontB, open(path_model_B+'lgbmr_primecontB.pkl', 'wb'))


# In[16]:


pickle.dump(gbmr_primebatB, open(path_model_B+'gbmr_primebatB.pkl', 'wb'))
pickle.dump(gbmr_primecontB, open(path_model_B+'gbmr_primecontB.pkl', 'wb'))

pickle.dump(rfr_primebatB, open(path_model_B+'rfr_primebatB.pkl', 'wb'))
pickle.dump(rfr_primecontB, open(path_model_B+'rfr_primecontB.pkl', 'wb'))


# In[17]:


pickle.dump(xgb_primebatB, open(path_model_B+'xgb_primebatB.pkl', 'wb'))
pickle.dump(xgb_primebatB, open(path_model_B+'xgb_primecontB.pkl', 'wb'))

pickle.dump(svmr_primecontB, open(path_model_B+'svmr_primebatB.pkl', 'wb'))
pickle.dump(xgb_primebatB, open(path_model_B+'svmr_primecontB.pkl', 'wb'))


# In[18]:


pickle.dump(mlpr_primebatB, open(path_model_B+'mlpr_primebatB.pkl', 'wb'))
pickle.dump(mlpr_primebatB, open(path_model_B+'mlpr_primecontB.pkl', 'wb'))

pickle.dump(mlpr_primebatB, open(path_model_B+'dtr_primebatB.pkl', 'wb'))
pickle.dump(dtr_primecontB, open(path_model_B+'dtr_primecontB.pkl', 'wb'))


# In[ ]:





# In[ ]:




