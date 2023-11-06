import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
from pgmpy.inference import VariableElimination
import pickle

# Define the paths and load the data/model as you did previously...
# Define the paths
path_datos_samuel = 'C:/Users/berna/OneDrive/Escritorio/Universidad de los Andes/Semestre 2023-2/Análitica Computacional para la Toma de Decisiones/Proyecto/predict+students+dropout+and+academic+success'
path_datos_juan = '/Users/juandramirezj/Documents/Universidad - MIIND/ACTD/proyecto_1/project_1_ACTD/data'
path_datos_actual = path_datos_juan
path_aws ='/home/ec2-user/'
parts = path_datos_actual.split('/')

desired_path = '/'.join(parts[:(len(parts)-1)])


# Load the trained model
with open(path_aws+'model1.pkl', "rb") as file:
    model1 = pickle.load(file)

# Load the data for dropdown population

data = pd.read_csv(path_aws+'final_data.csv')
# Unique values for dropdowns
unique_values = {
    "Age at enrollment": data["Age at enrollment"].unique(),
    "Unemployment rate": data["Unemployment rate"].unique(),
    "Inflation rate": data["Inflation rate"].unique(),
    "Debtor": data["Debtor"].unique(),
    "Scholarship holder": data["Scholarship holder"].unique(),
}

# Dictionary to map original values to more descriptive ones
readable_values = {
    "Age at enrollment": "Age Quartile",
    "Unemployment rate": {
        "Q1": "Low Unemployment",
        "Q2": "Below Average Unemployment",
        "Q3": "Above Average Unemployment",
        "Q4": "High Unemployment"
    },
    "Inflation rate": {
        "Q1": "Low Inflation",
        "Q2": "Below Average Inflation",
        "Q3": "Above Average Inflation",
        "Q4": "High Inflation"
    },
    "Debtor": {
        0: "No Debt",
        1: "Has Debt"
    },
    "Scholarship holder": {
        0: "No Scholarship",
        1: "Has Scholarship"
    }
}

dropdown_values = {}

for column, values in unique_values.items():
    if isinstance(readable_values[column], dict):
        dropdown_values[column] = [{'label': readable_values[column].get(value, value), 'value': value} for value in values]
    else:
        dropdown_values[column] = [{'label': value, 'value': value} for value in values]

# Adjust your app layout and callback accordingly...
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Bayesian Network Prediction for Student Dropout"),
    
# Description for the dashboard
    html.Div([
        html.P("This tool predicts the likelihood of a student dropping out based on various features."),
        html.P("Choose the relevant values from the dropdown menus below and click 'Predict'."),
        html.P("Descriptions:"),
        html.Ul([
            html.Li("Age Quartile: Age range the student belongs to: Q1 (<19), Q2 (19-20), Q3 (20-25) and Q4 (25>)"),
            html.Li("Unemployment Rate: Quartile of unemployment rate during admission."),
            html.Li("Inflation Rate: Quartile of inflation rate during admission."),
            html.Li("Debtor: Indicates if the student has debt for funding the university."),
            html.Li("Scholarship Holder: Indicates if the student has a scholarship.")
        ])
    ], style={"border": "1px solid #ddd", "padding": "10px", "margin-bottom": "20px", "border-radius": "5px"}),
   
    # Dropdowns for evidence
    dbc.Row([dbc.Col([html.Label(column), dcc.Dropdown(id=column, options=dropdown_values[column], value=values[0])]) for column, values in unique_values.items()]),
    html.Br(),

    # Button to predict
    dbc.Button("Predict Dropout", id="predict-button", color="primary"),

    html.Br(), html.Br(),

    # Display results
    html.Div(id="prediction-result")
])

@app.callback(
    Output("prediction-result", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State(column, "value") for column in unique_values]
)
def predict(n_clicks, age_enrollment, unemployment_rate, inflation_rate, debtor, scholarship_holder):
    if n_clicks:
        inference = VariableElimination(model1)
        prob = inference.query(
            variables=["actual_target"],
            evidence={
                "Age at enrollment": age_enrollment,
                "Unemployment rate": unemployment_rate,
                "Inflation rate": inflation_rate,
                "Debtor": debtor,
                "Scholarship holder": scholarship_holder
            })
        
        predicted_prob = prob.values[1]
        predicted_label = "Model predicts the student will Dropout" if predicted_prob > 0.28697751471555144 else "Model predicts the student will not Dropout"
        
        return dbc.Alert([
            html.H4(f"Probability of Dropout: {predicted_prob:.4f}"),
            html.P(predicted_label, className="mb-0")
        ], color="success" if predicted_prob > 0.28697751471555144 else "warning")

    return ""

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
