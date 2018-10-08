# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly import tools
from dash.dependencies import Input, Output
from plotly.graph_objs import *
import numpy as np
from scipy.special import expit, logit

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


trace1 = go.Scatter(
    x=[1, 2, 3],
    y=[4, 5, 6],
    mode='markers+text',
    text=['Text A', 'Text B', 'Text C'],
    textposition='bottom center'
)
trace2 = go.Scatter(
    x=[20, 30, 40],
    y=[50, 60, 70],
    mode='markers+text',
    text=['Text D', 'Text E', 'Text F'],
    textposition='bottom center'
)

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1/2)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(height=600, width=1200, title='i <3 annotations and subplots')




####### APP LAYOUT #########
app.layout = html.Div(
    html.Div([
        html.H4('Lidars visualizations'),
        dcc.Graph(id='graph',
                  figure=fig),

        dcc.Interval(
            id='interval-component',
            interval=0.5 * 1000,  # one 1/4 of a second
            n_intervals=0
        )
    ])
)


if __name__ == '__main__':
    app.run_server(debug=True)
