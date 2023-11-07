import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
from pgmpy.inference import VariableElimination
import pickle
import plotly.express as px
import numpy as np
from sqlalchemy import create_engine
from dash_core_components import Tabs, Tab

# Replace these with your actual credentials
username = 'postgres'
password = '...' #Ask for the password
host = 'database-2.cdwvwtd3pmq1.us-east-1.rds.amazonaws.com'
port = '5432'
database_name = 'postgres'

# Create the connection string for the database engine
database_uri = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"

# Create the database engine
engine = create_engine(database_uri)

# Example query - replace this with your actual query
data = pd.read_sql_query("SELECT * FROM new_total_filtered", engine)
data_plots = pd.read_sql_query("SELECT * FROM new_data_plots", engine)

path_modelo = '/home/ubuntu/'  

with open(path_modelo+'model1.pkl', "rb") as file:
    model1 = pickle.load(file)

with open(path_modelo+'hill_climb_search_model.pkl', "rb") as file:
    model2 = pickle.load(file)

data_plots['perc_approved_sem1'] = data_plots['curricular_units_1st_sem_approved']/data_plots['curricular_units_1st_sem_enrolled']
data_plots['perc_approved_sem2'] = data_plots['curricular_units_2nd_sem_approved']/data_plots['curricular_units_2nd_sem_enrolled']
data_plots['Target'] = np.where(data_plots['target'] == 'Dropout', 1, 0)

fig1 = px.box(data_plots, x='Target', y='age_at_enrollment', title='Age by Target')
fig2 = px.box(data_plots, x='Target', y=data_plots['perc_approved_sem2']*100,
              title='Percentage of units approved in the Second Semester by Target (%)')
fig3 = px.box(data_plots, x='debtor', y='perc_approved_sem1',
              title='Percentage of units approved in the First Semester by Debtor (%)')


# Unique values for dropdowns
unique_values = {
    "Age at enrollment": data["age_at_enrollment"].unique(),
    "Unemployment rate": data["unemployment_rate"].unique(),
    "Inflation rate": data["inflation_rate"].unique(),
    "Debtor": data["debtor"].unique(),
    "Scholarship holder": data["scholarship_holder"].unique(),
}

unique_values_2 = {
        "Scholarship holder": data["scholarship_holder"].unique(),
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
    dbc.Tabs([
        dbc.Tab(label='Prediction', children=[
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
            dbc.Button("Predict Dropout Model 1", id="predict-button", color="primary"),

            html.Br(), html.Br(),

            # Display results
            html.Div(id="prediction-result"),

            # Button to predict
            dbc.Button("Predict Dropout Model 2", id="predict-button-hill", color="primary"),

            html.Br(), html.Br(),

            # Display results
            html.Div(id="prediction-result-hill")
        ]),
        dbc.Tab(label='Visualizations', children=[
            html.H1("Data Insights"),
            dcc.Graph(id='age-target-plot', figure=fig1),
            dcc.Graph(id='units-approved-target-plot', figure=fig2),
            dcc.Graph(id='units-approved-debtor-plot', figure=fig3)
        ])
    ])

    
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
        predicted_label = "Model 1 predicts the student will Dropout" if predicted_prob > 0.28697751471555144 else "Model predicts the student will not Dropout"
        
        return dbc.Alert([
            html.H4(f"Probability of Dropout: {predicted_prob:.4f}"),
            html.P(predicted_label, className="mb-0")
        ], color="success" if predicted_prob > 0.28697751471555144 else "warning")

    return ""


@app.callback(
    Output("prediction-result-hill", "children"),
    [Input("predict-button-hill", "n_clicks")],
    [dash.dependencies.State(column, "value") for column in unique_values_2]
)

def predict_hill(n_clicks, scholarship_holder):
    if n_clicks:
        inference = VariableElimination(model2)
        prob = inference.query(
            variables=["actual_target"],
            evidence={
                "Scholarship holder": scholarship_holder
            })
        
        predicted_prob = prob.values[1]
        predicted_label = "Model 2 predicts the student will Dropout" if predicted_prob >= 0.3797150041911148 else "Model predicts the student will not Dropout"
        
        return dbc.Alert([
            html.H4(f"Probability of Dropout: {predicted_prob:.4f}"),
            html.P(predicted_label, className="mb-0")
        ], color="success" if predicted_prob > 0.28697751471555144 else "warning")


    return ""

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True)
