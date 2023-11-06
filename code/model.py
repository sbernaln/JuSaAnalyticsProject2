# Desarrollo de la sección 3. Ahora con otra red y datos
import numpy as np
# Define some important functions

def get_probs_dropout(probs_vector):
    probs_final=[]
    for prob in probs_vector:
        value_prob=prob.values[1]
        probs_final.append(value_prob)

    return probs_final

def evaluate_performance(predictions_prob, true_labels, threshold=0.5):
    # Convert the probabilities into predictions based on the specified threshold
    predictions = [1 if prob >= threshold else 0 for prob in predictions_prob]

    # Compute TP, FP, TN, and FN
    TP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 1])
    FP = sum([1 for i, j in zip(predictions, true_labels) if i == 1 and j == 0])
    TN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 0])
    FN = sum([1 for i, j in zip(predictions, true_labels) if i == 0 and j == 1])

    # Compute the metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "True positives": TP,
        "False positives": FP,
        "True negatives": TN,
        "False negatives": FN
    }

def plot_roc(probs, true_labels, title="ROC Curve", figsize=(10, 6)):
    """
    This function plots a ROC curve given predicted probabilities and actual labels.
    
    :param probs: Array-like, predicted probabilities.
    :param true_labels: Array-like, actual target values.
    :param title: String, desired title of the plot.
    :param figsize: tuple, size of the figure.
    """
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    
    # Calculate AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    
    # Set aesthetics
    sns.set_style("whitegrid")
    
    # Create the plot
    plt.figure(figsize=figsize)
    lw = 2  # Line width
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=13)
    
    # Display the plot
    plt.show()



# Importar los paquetes requeridos
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
import pickle

# Definir el path en dónde se encuentran los datos
path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto/predict+students+dropout+and+academic+success'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_1/project_1_ACTD/data'
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

# Save the final processed data

# Partir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# model1
# Definir la red bayesiana
model1 = BayesianNetwork([("Unemployment rate", "perc_approved_sem1"), ("Inflation rate", "perc_approved_sem1"),
                         ("Debtor", "perc_approved_sem1"), ("Scholarship holder", "perc_approved_sem1"),
                         ("perc_approved_sem1", "perc_approved_sem2"), ("perc_approved_sem2","actual_target"),
                         ("Age at enrollment","actual_target")])

model1.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in model1.nodes():
    print(model1.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model1)
# For each row in the test_data, predict the probability of "lung"
target_probabilities1 = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Age at enrollment": row["Age at enrollment"],
                                                                  "Unemployment rate": row["Unemployment rate"],
                                                                  "Inflation rate": row["Inflation rate"],
                                                                  "Debtor": row["Debtor"],
                                                                  "Scholarship holder": row["Scholarship holder"]})
    target_probabilities1.append(prob)

# Print the probabilities: OJO FALTA MIRAR ESTO

# model2
# Definir la red bayesiana
model2 = BayesianNetwork([("Previous qualification (grade)", "Debtor"), ("Previous qualification (grade)", "Scholarship holder"), ("Previous qualification (grade)", "perc_approved_sem1"),
                          ("Admission grade", "Debtor"), ("Admission grade", "Scholarship holder"), ("Admission grade", "perc_approved_sem1"),
                          ("Debtor", "perc_approved_sem2"), ("Scholarship holder", "perc_approved_sem2"), ("perc_approved_sem1", "perc_approved_sem2"),
                          ("perc_approved_sem2","actual_target")])
model2.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in model2.nodes():
    print(model2.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model2)
# For each row in the test_data, predict the probability of "lung"
target_probabilities2 = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Previous qualification (grade)": row["Previous qualification (grade)"],
                                                                  "Admission grade": row["Admission grade"]})
    target_probabilities2.append(prob)

# Model3
# Definir la red bayesiana
model3 = BayesianNetwork([("International", "perc_approved_sem1"), ("Marital status", "perc_approved_sem1"), ("Gender", "perc_approved_sem1"),
                          ("Unemployment rate", "perc_approved_sem1"), ("Inflation rate", "perc_approved_sem1"), ("Displaced", "perc_approved_sem1"),
                          ("Educational special needs", "perc_approved_sem1"), ("perc_approved_sem1", "perc_approved_sem2"), ("Scholarship holder", "actual_target"),
                          ("perc_approved_sem2","actual_target"), ("Debtor","actual_target")])

model3.fit(data=train_data, estimator=MaximumLikelihoodEstimator)

for i in model3.nodes():
    print(model3.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model3)
# For each row in the test_data, predict the probability of "lung"
target_probabilities3 = []

for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"International": row["International"],
                                                                  "Marital status": row["Marital status"],
                                                                  "Gender": row["Gender"],
                                                                  "Unemployment rate": row["Unemployment rate"],
                                                                  "Inflation rate": row["Inflation rate"],
                                                                  "Displaced": row["Displaced"],
                                                                  "Educational special needs": row["Educational special needs"],
                                                                  "Scholarship holder": row["Scholarship holder"],
                                                                  "Debtor": row["Debtor"]})
    target_probabilities3.append(prob)

# model4
# Definir la red bayesiana
model4 = BayesianNetwork([("International", "Debtor"), ("Displaced", "Debtor"),
                          ("Debtor", "perc_approved_sem2"), ("Marital status", "actual_target"),
                          ("perc_approved_sem2","actual_target")])
model4.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in model4.nodes():
    print(model4.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model4)
# For each row in the test_data, predict the probability of "lung"
target_probabilities4 = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"International": row["International"],
                                                                  "Marital status": row["Marital status"],
                                                                  "Displaced": row["Displaced"]})
    target_probabilities4.append(prob)

# model5
# Definir la red bayesiana
model5 = BayesianNetwork([("Age at enrollment", "Admission grade"), ("Educational special needs", "Admission grade"),
                          ("Admission grade", "perc_approved_sem1"), ("Debtor", "actual_target"),
                          ("perc_approved_sem1","actual_target")])
model5.fit(data=train_data, estimator=MaximumLikelihoodEstimator)
for i in model5.nodes():
    print(model5.get_cpds(i))

# Predict probabilities for testing

# Initialize VariableElimination class with the model
inference = VariableElimination(model5)
# For each row in the test_data, predict the probability of "lung"
target_probabilities5 = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Age at enrollment": row["Age at enrollment"],
                                                                  "Educational special needs": row["Educational special needs"],
                                                                  "Debtor": row["Debtor"]})
    target_probabilities5.append(prob)

# Model 6 (Bayesian estimation)


mod_fit_by = BayesianNetwork([("Debtor", "perc_approved_sem2"), ("Scholarship holder", "perc_approved_sem2"),
                         ("perc_approved_sem2", "actual_target")])

# Define some pseudo_counts for demonstration
pseudo_counts = {
    "Debtor": np.array([[0], [0]]), 
    "Scholarship holder": np.array([[0], [0]]),
    "perc_approved_sem2": np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
    "actual_target": np.array([[2500, 7500], [7500, 2500]])
}

# Fit the model using Bayesian Estimation
mod_fit_by.fit(data=train_data, estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=pseudo_counts)
# Print the CPDs for the nodes in the network
for node in mod_fit_by.nodes():
    print(mod_fit_by.get_cpds(node))
    print("\n")

# Initialize VariableElimination class with the model
inference = VariableElimination(mod_fit_by)
# For each row in the test_data, predict the probability of "lung"
target_probabilities_by = []
for index, row in test_data.iterrows():
    prob = inference.query(variables=["actual_target"], evidence={"Debtor": row["Debtor"],
                                                                  "Scholarship holder": row["Scholarship holder"]})
    target_probabilities_by.append(prob)


# UNTIL HERE WE ALREADY FOUND PREDICTED PROBAILITIES ON TEST

# Get the final probabilities of each model
probs_dropout1 = get_probs_dropout(target_probabilities1)
probs_dropout2 = get_probs_dropout(target_probabilities2)
probs_dropout3 = get_probs_dropout(target_probabilities3)
probs_dropout4 = get_probs_dropout(target_probabilities4)
probs_dropout5 = get_probs_dropout(target_probabilities5)
probs_dropout_by = get_probs_dropout(target_probabilities_by)


probs_df_1 = pd.DataFrame(probs_dropout1, columns=['Probability'])
probs_df_2 = pd.DataFrame(probs_dropout2, columns=['Probability'])
probs_df_3 = pd.DataFrame(probs_dropout3, columns=['Probability'])
probs_df_4 = pd.DataFrame(probs_dropout4, columns=['Probability'])
probs_df_5 = pd.DataFrame(probs_dropout5, columns=['Probability'])
probs_df_by = pd.DataFrame(probs_dropout_by, columns=['Probability'])


# Get descriptive statistics
quantile_cutoff = 1 - train_data['actual_target'].mean()
print(quantile_cutoff)


cutoff1 = probs_df_1['Probability'].quantile(quantile_cutoff)
cutoff2 = probs_df_2['Probability'].quantile(quantile_cutoff)
cutoff3 = probs_df_3['Probability'].quantile(quantile_cutoff)
cutoff4 = probs_df_4['Probability'].quantile(quantile_cutoff)
cutoff5 = probs_df_5['Probability'].quantile(quantile_cutoff)
cutoff_by = probs_df_by['Probability'].quantile(quantile_cutoff)


# Métricas de desempeño
performance1 = evaluate_performance(probs_dropout1, test_data['actual_target'], threshold=cutoff1)
performance2 = evaluate_performance(probs_dropout2, test_data['actual_target'], threshold=cutoff2)
performance3 = evaluate_performance(probs_dropout3, test_data['actual_target'], threshold=cutoff3)
performance4 = evaluate_performance(probs_dropout4, test_data['actual_target'], threshold=cutoff4)
performance5 = evaluate_performance(probs_dropout5, test_data['actual_target'], threshold=cutoff5)
performance_by = evaluate_performance(probs_dropout_by, test_data['actual_target'], threshold=cutoff_by)


print('------------------------')
print('Model 1 results')
for key, value in performance1.items():
    print(f"{key}: {value}")

print('------------------------')
print('Model 2 results')
for key, value in performance2.items():
    print(f"{key}: {value}")

print('------------------------')
print('Model 3 results')
for key, value in performance3.items():
    print(f"{key}: {value}")

print('------------------------')
print('Model 4 results')
for key, value in performance4.items():
    print(f"{key}: {value}")

print('------------------------')
print('Model 5 results')
for key, value in performance5.items():
    print(f"{key}: {value}")

print('------------------------')
print('Model 6 (Bayesian estimation) results')
for key, value in performance_by.items():
    print(f"{key}: {value}")


plot_roc(probs_dropout1, test_data['actual_target'], title="Model 1 ROC Curve: Testing data")
plot_roc(probs_dropout2, test_data['actual_target'], title="Model 2 ROC Curve: Testing data")
plot_roc(probs_dropout3, test_data['actual_target'], title="Model 3 ROC Curve: Testing data")
plot_roc(probs_dropout4, test_data['actual_target'], title="Model 4 ROC Curve: Testing data")
plot_roc(probs_dropout5, test_data['actual_target'], title="Model 5 ROC Curve: Testing data")
plot_roc(probs_dropout_by, test_data['actual_target'], title="Model 6 ROC Curve: Testing data")

# The best model was model #1

# Save the best model: Model 1
parts = path_datos_actual.split('/')

desired_path = '/'.join(parts[:(len(parts)-1)])

with open(desired_path+'/models/model1.pkl', 'wb') as file:
    pickle.dump(model1, file)
