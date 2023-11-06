import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Definir el path en dónde se encuentran los datos
path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto/predict+students+dropout+and+academic+success'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_2/JuSaAnalyticsProject2/data/raw'
path_salida_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_2/JuSaAnalyticsProject2/data/processed/'
path_datos_actual = path_datos_juan

# Cargar los datos
data = pd.read_csv(path_datos_actual+'/data.csv', delimiter=";")
# For numerical columns, fill NaN with mean
for col in data.select_dtypes(include=['float64', 'int64']):
    data[col].fillna(data[col].mean(), inplace=True)

# For categorical columns, fill NaN with mode
for col in data.select_dtypes(include=['object']):
    data[col].fillna(data[col].mode()[0], inplace=True)

# Exploración de los datos
data = data[data['Curricular units 1st sem (enrolled)']!=0]
data['perc_approved_sem1'] = data['Curricular units 1st sem (approved)']/data['Curricular units 1st sem (enrolled)']
data['perc_approved_sem2'] = data['Curricular units 2nd sem (approved)']/data['Curricular units 2nd sem (enrolled)']

# Get NA summary
nan_summary = data.isna().sum()
print(nan_summary)

nan_summary = nan_summary[nan_summary > 0]
print(nan_summary)

data['Age at enrollment'].quantile(0.25)
data['Age at enrollment'].quantile(0.5)
data['Age at enrollment'].quantile(0.75)

# Discretize variables and 'perc_approved_sem2' into quartiles
data['Inflation rate'] = pd.qcut(data['Inflation rate'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['Unemployment rate'] = pd.qcut(data['Unemployment rate'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['perc_approved_sem1'] = pd.qcut(data['perc_approved_sem1'], q=2, labels=["Q1", "Q2"])
data['perc_approved_sem2'] = pd.qcut(data['perc_approved_sem2'], q=2, labels=["Q1", "Q2"])
data['Age at enrollment'] = pd.qcut(data['Age at enrollment'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
data['Admission grade'] = pd.qcut(data['Admission grade'], q=2, labels=["Q1", "Q2"])
data['Previous qualification (grade)'] = pd.qcut(data['Previous qualification (grade)'], q=2, labels=["Q1", "Q2"])
data['actual_target'] = np.where(data['Target']=='Dropout',1,0)


# Partir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# Save the final processed data
train_data.to_csv(path_salida_juan + 'train.csv')
test_data.to_csv(path_salida_juan + 'test.csv')