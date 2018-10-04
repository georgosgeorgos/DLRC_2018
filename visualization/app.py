# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output
from plotly.graph_objs import *
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

####### APP LAYOUT #########
app.layout = html.Div(
    html.Div([
        html.H4('Lidars visualizations'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=0.5 * 1000,  # one 1/4 of a second
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n_intervals):

    # demo data
    lidar_data = np.random.randn(100)

    trace = go.Scatter(
        x=np.arange(len(lidar_data)) + len(lidar_data) * n_intervals,
        y=lidar_data,
        mode='lines+markers'
    )

    layout = Layout(
        xaxis=dict(
            title='Time',
            type='log'),
        yaxis=dict(
            title='Depth (m)',
            range=[-3, 3]
        )
    )

    return Figure(data=[trace], layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
