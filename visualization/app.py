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
from visualization.sampler import Sampler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

s = Sampler()

####### APP LAYOUT #########
app.layout = html.Div(
    html.Div([
        html.H4('Lidars visualizations'),
        dcc.Graph(id='live-update-graph-lidars'),
        dcc.Graph(id='live-update-graph-probs'),

        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # seconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-graph-lidars', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_lidars(n_intervals):

    # demo data
    # lidar_data = np.random.random(100)
    # data = lidar_data[np.random.choice(np.arange(10000), 150), 3]
    lidar_data = s.get_sample_lidar(n=1000)

    trace = go.Scatter(
        x=np.arange(len(lidar_data)) + len(lidar_data) * n_intervals,
        y=lidar_data,
        mode='lines',
        name='lidar',
        line=dict(color='#000000')
    )

    layout = Layout(
        yaxis=dict(
            title='Depth (m)',
            range=[0, 2]
        ),
        height=400,
        width=1200,
        showlegend=True,
        legend=dict(xanchor='right', yanchor='top')
    )

    return Figure(data=[trace], layout=layout)


@app.callback(Output('live-update-graph-probs', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_probs(n_intervals):

    # demo data
    probs_normal = np.ones(1000)

    trace = go.Scatter(
        x=np.arange(len(probs_normal)) + len(probs_normal) * n_intervals,
        y=probs_normal,
        mode='lines',
        fill='tozeroy',
        fillcolor='#cce6ff',
        name='normal',
        line=dict(smoothing=1., shape='spline')
    )

    # demo data
    probs_anom = expit(np.random.randn(1000))

    trace_b = go.Scatter(
        x=np.arange(len(probs_anom)) + len(probs_anom) * n_intervals,
        y=probs_anom,
        mode='lines',
        fill='tozeroy',
        fillcolor='#ffcccc',
        line=dict(color='#b30000', smoothing=1., shape='spline'),
        name='anomalous'
    )

    layout = Layout(
        xaxis=dict(
            title='Time'
        ),
        yaxis=dict(
            title='Prob',
            range=[0, 1]
        ),
        height=150,
        width=1200,
        showlegend=True,
        legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, 0.75)'),
        margin=dict(t=5)
    )

    return Figure(data=[trace, trace_b], layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
