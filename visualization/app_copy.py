# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output
from plotly.graph_objs import *
import numpy as np
from scipy.special import expit, logit

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

####### APP LAYOUT #########
app.layout = html.Div(
    html.Div([
        html.H4('Lidars visualizations'),
        dcc.Graph(id='live-update-graph-lidars'),
        dcc.Graph(id='live-update-graph-probs'),
        dcc.Interval(
            id='interval-component',
            interval=0.5 * 1000,  # one 1/4 of a second
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-graph-lidars', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_lidars(n_intervals):

    # demo data
    lidar_data = np.random.random(100)

    trace = go.Scatter(
        x=np.arange(len(lidar_data)) + len(lidar_data) * n_intervals,
        y=lidar_data,
        mode='lines+markers',
        fillcolor='#000000'
    )

    layout = Layout(
        xaxis=dict(
            title='Time',
            type='log'),
        yaxis=dict(
            title='Depth (m)',
            range=[0, 2]
        )
    )

    return Figure(data=[trace], layout=layout)


@app.callback(Output('live-update-graph-probs', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_probs(n_intervals):

    # demo data
    probs_normal = np.ones(100)

    trace = go.Scatter(
        x=np.arange(len(probs_normal)) + len(probs_normal) * n_intervals,
        y=probs_normal,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='#cce6ff'
    )

    # demo data
    probs_anom = expit(np.random.randn(100))

    trace_b = go.Scatter(
        x=np.arange(len(probs_anom)) + len(probs_anom) * n_intervals,
        y=probs_anom,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='#ffcccc',
        line=dict(color='#b30000', smoothing=1.3)
    )

    layout = Layout(
        xaxis=dict(
            title='Time',
            type='log'),
        yaxis=dict(
            title='Prob',
            range=[0, 1]
        ),
        height=250
    )

    return Figure(data=[trace, trace_b], layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
