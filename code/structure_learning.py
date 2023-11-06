##Código realizado por Samuel Bernal Neira - 201630619 y Juan David Ramírez Jiménez - 201814898

import pandas as pd

df = pd.read_csv("C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Taller 8/data_asia.csv")

df = df.drop("Unnamed: 0", axis = 1 )
print(df.head())
print(df.describe())
print(df.columns)

from pgmpy.estimators import PC
est = PC(data=df)

estimated_model = est.estimate(variant="stable", max_cond_vars=4)
print(estimated_model)
print(estimated_model.nodes())
print(estimated_model.edges())

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

estimated_model = BayesianNetwork(estimated_model)
estimated_model.fit(data=df, estimator = MaximumLikelihoodEstimator) 
for i in estimated_model.nodes():
    print(estimated_model.get_cpds(i))

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

scoring_method = K2Score(data=df)
esth = HillClimbSearch(data=df)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())
print(scoring_method.score(estimated_modelh))

#Punto 2 Big Score
from pgmpy.estimators import BicScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

scoring_method = BicScore(data=df)
esth = HillClimbSearch(data=df)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())
print(scoring_method.score(estimated_modelh))

## Punto 3
#Carga de Datos
from pandas import read_csv, DataFrame
import numpy as np

def annual_growth(row, years):
    min_year = years["min"]
    max_year = years["max"]
    row["Indicator Name"] = row["Indicator Name"] + " - [annual growth %]"
    for year in range(max_year, min_year, -1):
        if not np.isnan(row[str(year)]) and not np.isnan(row[str(year - 1)]):
            row[str(year)] = 100 * (float(row[str(year)]) - float(row[str(year - 1)])) / abs(float(row[str(year - 1)]))
        else:
            row[str(year)] = np.nan     
    row[str(min_year)] = np.nan
    return row

years = {"min" : 1960, "max" : 2019}
df_raw = read_csv("C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Taller 8/italy-raw-data.csv")
df_raw_growth = DataFrame(data=[row if "growth" in row["Indicator Name"] else annual_growth(row, years) for index, row in df_raw.iterrows()])
print("There are " + str(df_raw_growth.shape[0]) + " indicators in the dataframe.")
print(df_raw_growth.head())
#Extracción de Columnas
nodes = ['Pop', 'Urb', 'GDP', 'EC', 'FFEC', 'REC', 'EI', 'CO2', 'CH4', 'N2O']
df_growth = df_raw_growth.transpose().iloc[4:]
df_growth.columns = nodes
print(df_growth.head(10))

#Transformación de Variables
TIERS_NUM = 3

def boundary_str(start, end, tier):
    return f'{tier}: {start:+0,.2f} to {end:+0,.2f}'

def relabel_value(v, boundaries):
    if v >= boundaries[0][0] and v <= boundaries[0][1]:
        return boundary_str(boundaries[0][0], boundaries[0][1], tier='A')
    elif v >= boundaries[1][0] and v <= boundaries[1][1]: 
        return boundary_str(boundaries[1][0], boundaries[1][1], tier='B')
    elif v >= boundaries[2][0] and v <= boundaries[2][1]:
        return boundary_str(boundaries[2][0], boundaries[2][1], tier='C')
    else:
        return np.nan

def relabel(values, boundaries):
    result = []
    for v in values:
        result.append(relabel_value(v, boundaries))
    return result
        
def get_boundaries(tiers):
    prev_tier = tiers[0]
    boundaries = [(prev_tier[0], prev_tier[prev_tier.shape[0] - 1])]
    for index, tier in enumerate(tiers):
        if index is not 0:
            boundaries.append((prev_tier[prev_tier.shape[0] - 1], tier[tier.shape[0] - 1]))
            prev_tier = tier
    return boundaries
    
new_columns = {}
for i, content in enumerate(df_growth.items()):  
    (label, series) = content
    values = np.sort(np.array([x for x in series.tolist() if not np.isnan(x)] , dtype=float))
    if values.shape[0] < TIERS_NUM:
        print(f'Error: there are not enough data for label {label}')
        break
    boundaries = get_boundaries(tiers=np.array_split(values, TIERS_NUM)) 
    new_columns[label] = relabel(series.tolist(), boundaries)
    
df = DataFrame(data=new_columns)
df.columns = nodes
df.index = range(years["min"], years["max"] + 1)
print(df.head(10))
##Estimación por restricciones
est2 = PC(data=df)

estimated_model2 = est2.estimate(variant="stable", max_cond_vars=10)
print(estimated_model2)
print(estimated_model2.nodes())
print(estimated_model2.edges())

## Estimación por método Score K2
scoring_method = K2Score(data=df)
esth2 = HillClimbSearch(data=df)
estimated_model2 = esth2.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_model2)
print(estimated_model2.nodes())
print(estimated_model2.edges())
print(scoring_method.score(estimated_model2))

## Modelo de Lorenzo


modelLorenzo = BayesianNetwork([
('Pop', 'EC'),   ('Urb', 'EC'),   ('GDP', 'EC'),
('EC', 'FFEC'),  ('EC', 'REC'),   ('EC', 'EI'),
('REC', 'CO2'),  ('REC', 'CH4'),  ('REC', 'N2O'),
('FFEC', 'CO2'), ('FFEC', 'CH4'), ('FFEC', 'N2O')
])

print(scoring_method.score(modelLorenzo))
