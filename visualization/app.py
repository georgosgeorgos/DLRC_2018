# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from plotly import tools
from dash.dependencies import Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

N_LIDARS_VIZ = 1
N_TIMESTEPS = 1000
n_spacing_rows = 0.1

lidar_data = np.random.randn(N_TIMESTEPS)

####### CALLBACKS #########

@app.callback(
    Output('example-graph', 'children'),
    [Input()]
)
def update_graph_lidar():
    pass


####### APP LAYOUT #########
measurements_lidar = go.Scatter(
    x=np.arange(len(lidar_data)),
    y=lidar_data,
    mode='lines'
)

layout = go.Layout(
    title="Lidar Visualizations",
    xaxis=dict(title='Time'),
    yaxis=dict(title='Depth (m)')
)

figure = go.Figure(
    data=[measurements_lidar],
    layout=layout
)

app.layout = html.Div([

    # Hidden Div Storing JSON-serialized dataframe of run log
    html.Div(id='run-log-storage', style={'display': 'none'}),

    dcc.Graph(
        id='example-graph',
        figure=figure
    )
]
)

if __name__ == '__main__':
    app.run_server(debug=True)
