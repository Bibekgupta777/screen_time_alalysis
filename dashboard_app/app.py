# ==============================================
#  THESIS DASHBOARD (Improved Visuals + ML Prediction)
# ==============================================

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import joblib

# --------------------------
# ✅ LOAD DATA & MODELS
# --------------------------
df = pd.read_csv("../data/cleaned_screen_time.csv")
df = df[df["gender"].isin(["Male", "Female"])]
df["screen_time_group"] = pd.cut(
    df["daily_screen_time_hours"],
    bins=[0, 4, 8, 16],
    labels=["Low", "Moderate", "High"]
)

mental_model = joblib.load("../data/rf_mental_health_model.pkl")
stress_model = joblib.load("../data/rf_stress_level_model.pkl")
le_gender = joblib.load("../data/le_gender.pkl")
le_location = joblib.load("../data/le_location.pkl")

# --------------------------
# ✅ DASHBOARD SETUP
# --------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = "Thesis Dashboard - Final"

# --------------------------
# ✅ APP LAYOUT
# --------------------------
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("Screen Time Thesis Dashboard", className="text-center mt-3"))]),

    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Theme Mode:"),
            dcc.RadioItems(
                options=[{"label": " Light", "value": "light"}, {"label": " Dark", "value": "dark"}],
                value="light", id="theme-toggle", inline=True
            )
        ], width=2),
        dbc.Col([
            html.Label("Gender:"),
            dcc.Dropdown(options=["All"] + list(df["gender"].unique()), value="All",
                         id="gender-filter", clearable=False)
        ], width=2),
        dbc.Col([
            html.Label("Location Type:"),
            dcc.Dropdown(options=["All"] + list(df["location_type"].unique()), value="All",
                         id="location-filter", clearable=False)
        ], width=3),
        dbc.Col([
            html.Label("Age Range:"),
            dcc.RangeSlider(min=int(df["age"].min()), max=int(df["age"].max()), step=1,
                            value=[18, 40], id="age-slider",
                            marks={i: str(i) for i in range(10, 80, 10)})
        ], width=3),
        dbc.Col([
            html.Label("Daily Screen Time (Hours):"),
            dcc.RangeSlider(min=float(df["daily_screen_time_hours"].min()),
                            max=float(df["daily_screen_time_hours"].max()), step=0.5,
                            value=[2, 10], id="screen-slider",
                            marks={i: str(i) for i in range(0, 17, 4)})
        ], width=2)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Button("⬇ Download Filtered Data as CSV", id="btn-download",
                           color="primary"), width=3),
        dcc.Download(id="download-dataframe-csv"),
    ], className="mb-4"),

    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Stress Analysis", value="stress"),
        dcc.Tab(label="Sleep Analysis", value="sleep"),
        dcc.Tab(label="AI Insights", value="insights"),
        dcc.Tab(label="ML Prediction", value="ml"),
    ]),

    html.Div(id="tab-content", className="tab-content p-3")
], fluid=True)

# --------------------------
# ✅ TAB CALLBACK
# --------------------------
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"),
     Input("theme-toggle", "value"),
     Input("gender-filter", "value"),
     Input("location-filter", "value"),
     Input("age-slider", "value"),
     Input("screen-slider", "value")]
)
def update_tabs(tab, theme, gender, location, age_range, screen_range):
    app.external_stylesheets = [dbc.themes.BOOTSTRAP] if theme == "light" else [dbc.themes.DARKLY]

    filtered = df.copy()
    if gender != "All":
        filtered = filtered[filtered["gender"] == gender]
    if location != "All":
        filtered = filtered[filtered["location_type"] == location]
    filtered = filtered[
        (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1]) &
        (filtered["daily_screen_time_hours"] >= screen_range[0]) &
        (filtered["daily_screen_time_hours"] <= screen_range[1])
    ]

    # ---------------- OVERVIEW ----------------
    if tab == "overview":
        fig1 = px.histogram(filtered, x="daily_screen_time_hours", nbins=20,
                            title="Screen Time Distribution", color_discrete_sequence=["#1f77b4"])
        fig2 = px.box(filtered, x="gender", y="daily_screen_time_hours", color="gender",
                      title="Gender-wise Screen Time", color_discrete_sequence=["#e377c2", "#17becf"])
        fig3 = px.scatter(filtered, x="daily_screen_time_hours", y="mental_health_score",
                          color="gender", trendline="ols", title="Screen Time vs Mental Health Score")

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig1), width=4),
                dbc.Col(dcc.Graph(figure=fig2), width=4),
                dbc.Col(dcc.Graph(figure=fig3), width=4)
            ]),
            html.H5("Interpretation:"),
            html.P(f"Average Screen Time: {filtered['daily_screen_time_hours'].mean():.2f} hrs; "
                   f"Mental Health Score Avg: {filtered['mental_health_score'].mean():.2f}")
        ])

    # ---------------- STRESS ----------------
    elif tab == "stress":
        fig4 = px.violin(filtered, x="location_type", y="stress_level", color="location_type",
                         box=True, points="all", title="Stress Level by Location")
        fig5 = px.scatter(filtered, x="stress_level", y="daily_screen_time_hours", color="gender",
                          trendline="ols", title="Stress vs Screen Time")

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig4), width=6),
                     dbc.Col(dcc.Graph(figure=fig5), width=6)]),
            html.H5("Interpretation:"),
            html.P(f"Average Stress Level: {filtered['stress_level'].mean():.2f}")
        ])

    # ---------------- SLEEP ----------------
    elif tab == "sleep":
        fig6 = px.histogram(filtered, x="sleep_duration_hours", color="screen_time_group",
                            title="Sleep Duration by Screen Time Group")
        fig7 = px.scatter(filtered, x="sleep_duration_hours", y="mental_health_score",
                          color="screen_time_group", trendline="ols", title="Sleep vs Mental Health Score")

        return html.Div([
            dbc.Row([dbc.Col(dcc.Graph(figure=fig6), width=6),
                     dbc.Col(dcc.Graph(figure=fig7), width=6)]),
            html.H5("Interpretation:"),
            html.P(f"Average Sleep Duration: {filtered['sleep_duration_hours'].mean():.2f} hrs")
        ])

    # ---------------- AI INSIGHTS ----------------
    elif tab == "insights":
        corr = filtered.corr(numeric_only=True)["daily_screen_time_hours"].sort_values(ascending=False)
        top_corr = corr[1:6].to_frame().reset_index().rename(
            columns={"index": "Variable", "daily_screen_time_hours": "Correlation"})

        return html.Div([
            html.H4("AI Insights", className="mb-3"),
            html.Ul([
                html.Li(f"Highest correlation: {top_corr.iloc[0]['Variable']} "
                        f"(r = {top_corr.iloc[0]['Correlation']:.2f})"),
                html.Li(f"Highest Screen Time by Location: "
                        f"{filtered.groupby('location_type')['daily_screen_time_hours'].mean().idxmax()}"),
            ]),
            dbc.Table.from_dataframe(top_corr.round(3), striped=True, bordered=True, hover=True,
                                     className="table-dark" if theme == "dark" else "table-light")
        ])

    # ---------------- ML PREDICTION ----------------
    elif tab == "ml":
        return html.Div([
            html.H4("ML Prediction: Predict Mental Health & Stress Level"),
            dbc.Row([
                dbc.Col([html.Label("Age:"), dcc.Input(id="ml-age", type="number", value=25)], width=2),
                dbc.Col([html.Label("Daily Screen Time (hrs):"), dcc.Input(id="ml-screen", type="number", value=5)], width=3),
                dbc.Col([html.Label("Sleep Duration (hrs):"), dcc.Input(id="ml-sleep", type="number", value=7)], width=3),
                dbc.Col([html.Label("Gender:"), dcc.Dropdown(options=list(df["gender"].unique()), value="Male",
                                                             id="ml-gender", clearable=False)], width=2),
                dbc.Col([html.Label("Location Type:"), dcc.Dropdown(options=list(df["location_type"].unique()), value="Urban",
                                                                    id="ml-location", clearable=False)], width=2),
            ], className="mb-3"),
            dbc.Button("Predict", id="ml-predict-btn", color="success", className="mt-2"),
            html.Div(id="ml-result", className="mt-3 p-2")
        ])

# --------------------------
# ✅ ML Prediction Callback
# --------------------------
@app.callback(
    Output("ml-result", "children"),
    Input("ml-predict-btn", "n_clicks"),
    [State("ml-age", "value"), State("ml-screen", "value"), State("ml-sleep", "value"),
     State("ml-gender", "value"), State("ml-location", "value")],
    prevent_initial_call=True
)
def predict_ml(n_clicks, age, screen, sleep, gender, location):
    gender_encoded = le_gender.transform([gender])[0]
    location_encoded = le_location.transform([location])[0]
    features = [[age, screen, sleep, gender_encoded, location_encoded]]

    mental_pred = mental_model.predict(features)[0]
    stress_pred = stress_model.predict(features)[0]

    gauge1 = go.Figure(go.Indicator(
        mode="gauge+number", value=mental_pred,
        title={"text": "Mental Health Score"}, gauge={"axis": {"range": [0, 10]}}
    ))
    gauge2 = go.Figure(go.Indicator(
        mode="gauge+number", value=stress_pred,
        title={"text": "Stress Level"}, gauge={"axis": {"range": [0, 10]}, "bar": {"color": "red"}}
    ))

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=gauge1), width=6),
                 dbc.Col(dcc.Graph(figure=gauge2), width=6)]),
        html.H5("Interpretation:"),
        html.P(f"Predicted Mental Health Score: {mental_pred:.2f} "
               f"(Higher means better mental health)."),
        html.P(f"Predicted Stress Level: {stress_pred:.2f} "
               f"(Higher means more stress).")
    ])

# --------------------------
# ✅ CSV Download
# --------------------------
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("gender-filter", "value"),
    State("location-filter", "value"),
    State("age-slider", "value"),
    State("screen-slider", "value"),
    prevent_initial_call=True
)
def download_filtered_data(n_clicks, gender, location, age_range, screen_range):
    filtered = df.copy()
    if gender != "All": filtered = filtered[filtered["gender"] == gender]
    if location != "All": filtered = filtered[filtered["location_type"] == location]
    filtered = filtered[
        (filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1]) &
        (filtered["daily_screen_time_hours"] >= screen_range[0]) &
        (filtered["daily_screen_time_hours"] <= screen_range[1])
    ]
    return dcc.send_data_frame(filtered.to_csv, "filtered_screen_time_data.csv", index=False)

# --------------------------
# ✅ RUN APP
# --------------------------
if __name__ == "__main__":
    app.run(debug=False, port=3000)
