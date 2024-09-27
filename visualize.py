import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_results(df_test, predictions, mc_predictions=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test.median_income, y=df_test.median_house_value, mode="markers", name="Actual"))
    
    # Add standard predictions
    fig.add_trace(go.Scatter(x=df_test.median_income, y=predictions, mode="markers", name="Predictions"))
    
    # Add Monte Carlo predictions if provided
    if mc_predictions is not None:
        for i in range(len(mc_predictions)):
            fig.add_trace(go.Scatter(x=df_test.median_income, y=mc_predictions[i].reshape(-1,), mode="markers", opacity=0.1, marker=dict(color='Red')))
    
    fig.show()

def plot_histogram(df_train, df_test, mc_predictions):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Histogram of income
    fig.add_trace(go.Histogram(x=df_train["median_income"].values, opacity=0.3, marker=dict(color="Gold")), secondary_y=True)

    # Actual and predicted scatter plot
    fig.add_trace(go.Scatter(x=df_test.median_income, y=df_test.median_house_value, mode="markers", marker=dict(color="Orange")))
    for i in range(len(mc_predictions)):
        fig.add_trace(go.Scatter(x=df_test.median_income, y=mc_predictions[i].reshape(-1,), mode="markers", opacity=0.1, marker=dict(color='Red')))

    fig.add_trace(go.Scatter(x=df_test.median_income, y=mc_predictions.mean(axis=0).reshape(-1,), mode="markers", opacity=0.7, marker=dict(color='Green')))

    # Set axes titles
    fig.update_yaxes(title_text="<b>primary</b> House Price", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> Number of datapoints", secondary_y=True)
    fig.update_xaxes(title_text="Household Income")
    
    fig.show()
