import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import datetime as dt
import scipy.stats
import statsmodels.formula.api as sm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

cat_dataset = pd.read_csv(r'C:\Users\cmarc\OneDrive\2019\Academia\UNAB\Doctor\05. Machine Learning\Caso\Caso\csv_egresos.csv', sep =';', encoding='latin1')

cat_dataset.head()

cat_dataset.head()

cat_MINPUB = pd.read_csv(r'C:\Users\cmarc\OneDrive\2019\Academia\UNAB\Doctor\05. Machine Learning\Caso\Caso\Delitos_CSV.csv', sep =';', encoding='latin1')

cat_MINPUB.head()

df_egr.shape

df_del =pd.read_csv(r'C:\Users\cmarc\OneDrive\2019\Academia\UNAB\Doctor\05. Machine Learning\Caso\Delitos_CSV.csv', sep =';')
df_del.head()

df_del.shape

df_fin = pd.merge(df_del, df_egr, on ='COD_DELITO')
df_fin.head()

df_fin.shape
df_fin.columns
df_fin.info()
df_fin.shape
df_fin2 = df_fin
df_final = df_fin2.drop_duplicates()
df_final.shape
df_final.head()

df_final["COD_PERS"].value_counts()
df_final.groupby(['MES_EGRESO', 'COD_PERS','Codigo', 'COD_DELITO'])['SCORE'].sum()
df_fin2.isnull().sum()
df_fin2.nunique()
df_fin2['COD_DELITO'].unique()
df_fin2['MES_EGRESO'].unique()

plt.rcParams['figure.figsize'] = [18, 16]
df_fin2.plot(kind="density", subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()

df_fin2['MES_EGRESO'].value_counts()

df_fin2['MES_EGRESO'] = df_fin2['MES_EGRESO'].astype(str)

df_fin2.groupby(['MES_EGRESO'])['SCORE'].sum()

df_fin2.groupby(['MES_EGRESO'])['SCORE'].sum().plot(kind='bar', figsize=(10,4), title='SCORE EGRESOS POR MES/AÑO'

score = df_fin2.groupby(['MES_EGRESO','COD_PERS']).agg({'SCORE': lambda x: x.sum()})
score

score.reset_index(inplace=True)
score.head()

col =['COD_PERS', 'MES_EGRESO', 'SCORE', 'COD_DELITO','Codigo']
rfm = df_fin2[col]
rfm.head()
rfm.shape

rfm['MES_EGRESO'] = pd.to_datetime(rfm['MES_EGRESO'],errors ='coerce')
rfm['MES_EGRESO'].max()
f_corte = dt.datetime(2022,7,1)
rfm = rfm.drop_duplicates()

RFM1 = rfm.groupby('COD_PERS').agg({'MES_EGRESO': lambda x: (f_corte - x.max()).days})
RFM1['Frecuencia'] = (rfm.groupby(by=['COD_PERS'])['Codigo'].count()).astype(float)
RFM1['ScoreTotal'] = rfm.groupby(by=['COD_PERS']).agg({'SCORE': 'sum'})
RFM1.head()

RFM1.rename(columns={'MES_EGRESO': 'Egreso más reciente'}, inplace=True)
RFM1.head()

plt.rcParams['figure.figsize'] = [16, 14]
RFM1.plot(kind="density", subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

plt.figure(figsize=(12,10))
# Recencia
plt.subplot(3, 1, 1); sns.histplot(RFM1['Egreso más reciente'])
# Frecuencia 
plt.subplot(3, 1, 2); sns.histplot(RFM1['Frecuencia'])
# Score Total 
plt.subplot(3, 1, 3); sns.histplot(RFM1['ScoreTotal'])

RFM1.describe()
RFM1[RFM1['Egreso más reciente'] == 0]
RFM1[RFM1['Frecuencia'] == 0]
RFM1[RFM1['ScoreTotal'] == 0]
RFM1 = RFM1[RFM1['Egreso más reciente'] > 0]
RFM1.reset_index(drop=True,inplace=True)
RFM1 = RFM1[RFM1['Frecuencia'] > 0]
RFM1.reset_index(drop=True,inplace=True)
RFM1 = RFM1[RFM1['ScoreTotal'] > 0]
RFM1.reset_index(drop=True,inplace=True)

Data_RFM1 = RFM1[['Egreso más reciente','Frecuencia','ScoreTotal']]
Data_RFM1.describe()

data_log = np.log(Data_RFM1)
scaler = StandardScaler()
scaler.fit(data_log)
data_sc = scaler.transform(data_log)
df_norm = pd.DataFrame(data_sc, columns=Data_RFM1.columns)
df_norm.head()

plt.figure(figsize=(12,10))
# Distribucipon Variable Recencia 
plt.subplot(3, 1, 1); sns.histplot(df_norm['Egreso más reciente'])
# Distribución Variable Frecuencia 
plt.subplot(3, 1, 2); sns.histplot(df_norm['Frecuencia'])
# Distribución variable Score total
plt.subplot(3, 1, 3); sns.histplot(df_norm['ScoreTotal'])

grupos
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4, style="whitegrid")
sns.lineplot(data = grupos, x = 'Número de grupo', y = 'inertia').set(title = "Método del Codo")
plt.show()

def plots_model():    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for x in RFM1.grupos.unique():        
        xs = RFM1[RFM1.grupos == x]['Egreso más reciente']
        zs = RFM1[RFM1.grupos == x]['Frecuencia']
        ys = RFM1[RFM1.grupos == x]['ScoreTotal']
        ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', label = x)

    ax.set_xlabel('Egreso más reciente')
    ax.set_zlabel('Frecuencia')
    ax.set_ylabel('ScoreTotal')
    plt.title('Visualization de los grupos creados')
    plt.legend()
    plt.show()

model = KMeans(n_clusters=4, init='k-means++', max_iter=300)
grupos = model.fit_predict(df_norm)
df_norm['grupos'] = grupos
RFM1['grupos'] = grupos
plots_model()
