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
from visualization.probs import Probs

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

p = Probs(n=1000)

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
    # lidar_data = np.random.random(1000)
    # data = lidar_data[np.random.choice(np.arange(10000), 150), 3]
    lidar_data = p.get_data()['input'][:, 3]

    probs_classes = np.random.randn(3)
    probs_classes = np.exp(expit(probs_classes)) / np.sum(expit(probs_classes))

    trace = go.Scatter(
        x=np.arange(len(lidar_data)) + len(lidar_data) * n_intervals,
        y=lidar_data,
        mode='lines',
        name='lidar',
        line=dict(color='#000000')
    )

    trace2 = go.Bar(
        y=['thyself', 'background', 'other agents'],
        x=probs_classes,
        orientation='h',
        marker=dict(
            color=['#D2B4DE', '#D1F2EB',
                   '#F7DC6F'])
    )

    # layout = Layout(
    #     yaxis=dict(
    #         title='Depth (m)',
    #         range=[0, 2]
    #     ),
    #     height=400,
    #     width=1200,
    #     showlegend=True,
    #     legend=dict(xanchor='right', yanchor='top')
    # )

    fig = tools.make_subplots(rows=1, cols=2, column_width=[4,1])

    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout'].update(height=400, width=1300)

    return fig

    # return Figure(data=[trace], layout=layout)


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
        width=955,
        showlegend=True,
        legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, 0.75)'),
        margin=dict(t=3)
    )

    return Figure(data=[trace, trace_b], layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
