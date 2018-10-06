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
from sampler import Sampler_anomaly_clustering

app = dash.Dash(__name__)

N_SAMPLES = 100
N_LIDAR = 3
N_UPDATE_EVERY = 1
N_STD = 3

p = Sampler_anomaly_clustering(n=N_SAMPLES, l=N_LIDAR)

####### APP LAYOUT #########
app.layout = html.Div(
    html.Div([
        html.Div([
            html.H3('Lidars visualizations'),
            dcc.Graph(id='live-update-graph-lidars'),
            dcc.Graph(id='live-update-graph-probs'),
            dcc.Interval(
                id='interval-component',
                interval=N_UPDATE_EVERY * 1000,  # seconds
                n_intervals=0
            )
        ], className="nine columns"),

        html.Div([
            html.H3('Clustering'),
            dcc.Graph(id='live-update-bar-chart-clustering')
        ], className="three columns"),

    ], className="row")
)


@app.callback(Output('live-update-graph-lidars', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_lidars(n_intervals):

    data = p.get_data(n_intervals)
    lidar_inp = data['input']
    lidar_mean = data['mu']
    lidar_std = data['std']

    trace_inp = go.Scatter(
        x=np.arange(len(lidar_inp)) + len(lidar_inp) * n_intervals,
        y=lidar_inp,
        mode='lines',
        name='lidar',
        line=dict(color='#000000')
    )

    trace_mean = go.Scatter(
        x=np.arange(len(lidar_mean)) + len(lidar_mean) * n_intervals,
        y=lidar_mean,
        mode='lines',
        name='mean',
        line=dict(color='#E74C3C')
    )

    trace_upper_std = go.Scatter(
        x=np.arange(len(lidar_std)) + len(lidar_std) * n_intervals,
        y=lidar_mean + N_STD * lidar_std,
        mode='lines',
        name='conf interval',
        line=dict(color='#D6EAF8')
    )

    trace_lower_std = go.Scatter(
        x=np.arange(len(lidar_std)) + len(lidar_std) * n_intervals,
        y=lidar_mean - N_STD * lidar_std,
        mode='lines',
        line=dict(color='#D6EAF8')
    )

    layout = Layout(
        yaxis=dict(
            title='Depth (m)',
            range=[0, 2]
        ),
        height=400,
        showlegend=True,
        legend=dict(xanchor='right', yanchor='top'),
        margin=dict(r=0)
    )

    return Figure(data=[trace_inp, trace_mean, trace_upper_std, trace_lower_std], layout=layout)


@app.callback(Output('live-update-bar-chart-clustering', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_barchart_clustering(n_intervals):

    probs_classes = np.random.randn(3)
    probs_classes = np.exp(probs_classes) / np.sum(np.exp(probs_classes), axis=0)

    trace = go.Bar(
        y=['thyself', 'background', 'other agents'],
        x=probs_classes,
        orientation='h',
        marker=dict(
            color=['#D2B4DE', '#D1F2EB',
                   '#F7DC6F']),
        width=1
    )

    layout = Layout(
        xaxis=dict(
            title='Class probability',
            range=[0, 1]
        ),
        height=400,
        showlegend=False
    )

    return Figure(data=[trace], layout=layout)


@app.callback(Output('live-update-graph-probs', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_probs(n_intervals):
    probs_anom = p.get_data(n_intervals)['prob']
    lidar = p.get_data(n_intervals)['input']

    probs_normal = np.ones(len(probs_anom))

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
    # probs_anom = expit(np.random.randn(N_SAMPLES))

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
        showlegend=True,
        legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, 0.75)'),
        margin=dict(t=3, r=0)
    )

    return Figure(data=[trace, trace_b], layout=layout)



################################
#### START APP
################################

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
