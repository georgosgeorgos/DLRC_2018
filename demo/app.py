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
from demo.sampler_app import Sampler_anomaly_clustering
import json

app = dash.Dash(__name__)

N_SAMPLES = 1
N_LIDAR = 3
N_INTERVAL_UPDATE = 1  # in seconds
N_STD = 3
list_lidar_depth = []
list_lidar_depth_mean = []
list_lidar_depth_std = []
list_prob_anomaly = []
list_prob_normal = []
N_MAX_INTERVALS = 100

p = Sampler_anomaly_clustering(n=N_SAMPLES, l=N_LIDAR)

####### APP LAYOUT #########
app.layout = html.Div(
    html.Div([
        html.Div([
            dcc.Graph(id='live-update-graph-lidars'),
            dcc.Graph(id='live-update-graph-probs'),
            dcc.Interval(
                id='interval-component',
                interval=N_INTERVAL_UPDATE * 1000,
                n_intervals=N_MAX_INTERVALS
            )
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='live-update-bar-chart-clustering-two-classes')
        ], className="three columns"),
        html.Div([
            dcc.Graph(id='live-update-bar-chart-clustering-three-classes')
        ], className="three columns"),
        html.Div(id='store-data', style={'display': 'none'})
    ], className="row")
)


@app.callback(Output('store-data', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_data(n):
    data = p.get_data(n)
    return json.dumps(data)


@app.callback(Output('live-update-graph-lidars', 'figure'),
              [
                  Input('interval-component', 'n_intervals'),
                  Input('store-data', 'children')
              ])
def update_graph_lidars(n, json_data):
    data = json.loads(json_data)

    list_lidar_depth.append(data['input'][0])
    list_lidar_depth_mean.append(data['mu'][0])
    list_lidar_depth_std.append(data['std'][0])

    timesteps = np.arange(len(list_lidar_depth))

    trace_inp = go.Scatter(
        x=timesteps,
        y=list_lidar_depth,
        mode='lines',
        name='lidar',
        line=dict(color='#000000', smoothing=0.5, shape='spline')
    )

    trace_mean = go.Scatter(
        x=timesteps,
        y=list_lidar_depth_mean,
        mode='lines',
        name='mean',
        line=dict(color='#E74C3C', smoothing=0.5, shape='spline')
    )

    trace_upper_std = go.Scatter(
        x=timesteps,
        y=np.array(list_lidar_depth_mean) + N_STD * np.array(list_lidar_depth_std),
        mode='lines',
        name='conf interval',
        fill='tonexty',
        fillcolor='rgba(174, 214, 241, 0.5)',
        line=dict(color='rgba(174, 214, 241, 0.5)')
    )

    trace_lower_std = go.Scatter(
        x=timesteps,
        y=np.array(list_lidar_depth_mean) - N_STD * np.array(list_lidar_depth_std),
        mode='lines',
        showlegend=False,
        fillcolor='rgba(174, 214, 241, 0.5)',
        line=dict(color='rgba(174, 214, 241, 0.5)')
    )

    layout = Layout(
        yaxis=dict(
            title='Depth (m)'
        ),
        height=400,
        showlegend=True,
        title='Lidar measurements',
        legend=dict(xanchor='right', yanchor='top'),
    )

    return Figure(data=[trace_lower_std, trace_upper_std, trace_inp, trace_mean], layout=layout)


@app.callback(Output('live-update-graph-probs', 'figure'),
              [
                  Input('interval-component', 'n_intervals'),
                  Input('store-data', 'children')
              ])
def update_graph_probs(n, json_data):
    data = json.loads(json_data)
    probs_anom = data['prob'][0]
    list_prob_anomaly.append(probs_anom)
    list_prob_normal.append(1)
    timesteps = np.arange(len(list_prob_anomaly))

    trace = go.Scatter(
        x=timesteps,
        y=list_prob_normal,
        mode='lines',
        fill='tozeroy',
        fillcolor='#ABEBC6',
        name='normal',
        line=dict(color='#239B56', smoothing=0.5, shape='spline')
    )

    trace_b = go.Scatter(
        x=timesteps,
        y=list_prob_anomaly,
        mode='lines',
        fill='tozeroy',
        fillcolor='#ffcccc',
        line=dict(color='#b30000', smoothing=0.5, shape='spline'),
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
        height=250,
        showlegend=True,
        title='Stage one: Anomaly Detection',
        legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, 0.5)'),
    )

    return Figure(data=[trace, trace_b], layout=layout)


@app.callback(Output('live-update-bar-chart-clustering-two-classes', 'figure'),
              [
                  Input('interval-component', 'n_intervals'),
                  Input('store-data', 'children')
              ])
def update_barchart_clustering_two(n, json_data):
    data = json.loads(json_data)
    probs_classes = data['cluster'][0]

    trace = go.Bar(
        y=['background', 'thyself'],
        x=probs_classes,
        orientation='h',
        marker=dict(
            color=['#D6DBDF', '#273746']),
        width=1
    )

    layout = Layout(
        xaxis=dict(
            title='Class probability',
            range=[0, 1]
        ),
        height=400,
        title='Stage two: Clustering into two classes',
        showlegend=False
    )

    return Figure(data=[trace], layout=layout)


@app.callback(Output('live-update-bar-chart-clustering-three-classes', 'figure'),
              [
                  Input('interval-component', 'n_intervals'),
                  Input('store-data', 'children')
              ])
def update_barchart_clustering_three(n, json_data):
    data = json.loads(json_data)
    probs_classes = data['cluster_n'][0]

    trace = go.Bar(
        y=['background', 'thyself', 'other agents'],
        x=probs_classes,
        orientation='h',
        marker=dict(
            color=['#D6DBDF', '#273746',
                   '#b30000']),
        width=1
    )

    layout = Layout(
        xaxis=dict(
            title='Class probability',
            range=[0, 1]
        ),
        height=400,
        title='Clustering of exteroceptive signals into three classes',
        showlegend=False
    )

    return Figure(data=[trace], layout=layout)



################################
#### START APP
################################

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
