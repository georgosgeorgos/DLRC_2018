from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import os.path as osp
import json
from collections import defaultdict

rootdir = '../../../'
path_results = osp.join(rootdir, 'experiments', 'anomVAE')

d_train_input = defaultdict(list)
d_train_input_recon = defaultdict(list)
N_UNIT_TIMESTEP = 0.5
N_WINDOW_SIZE = 2
INPUT_IDX = 3

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv(osp.join(path_results, 'train.csv'))
df_recon = pd.read_csv(osp.join(path_results, 'train_recon.csv'))

app.layout = html.Div(children=[
    dcc.Graph(
        id='graph'
    ),
    html.Div(id='store-data-json', style={'display': 'none'}),
    dcc.Interval(
        id='interval',
        interval=N_UNIT_TIMESTEP * 1000,
        n_intervals=0,
        max_intervals=len(df) - 1
    )
], className="row")


@app.callback(
    Output(component_id='store-data-json', component_property='children'),
    [Input(component_id='interval', component_property='n_intervals')]
)
def read_data(n):
    row = df.iloc[n, :].values.tolist()
    row_recon = df_recon.iloc[n, :].values.tolist()

    json_data = {}
    json_data['input'] = row
    json_data['recon'] = row_recon

    return json.dumps(json_data)


@app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='interval', component_property='n_intervals'),
     Input(component_id='store-data-json', component_property='children')]
)
def update_subplots(n, row_json):
    data = json.loads(row_json)

    row = data['input']
    row_recon = data['recon']

    d_train_input[INPUT_IDX].append(row[INPUT_IDX])
    d_train_input_recon[INPUT_IDX].append(row_recon[INPUT_IDX])

    fig = tools.make_subplots(rows=1, cols=1,
                              print_grid=False)

    if n < N_WINDOW_SIZE:
        timesteps = np.arange(len(d_train_input[INPUT_IDX]))
        # trace2 = go.Scatter(y=np.array(d_train_input_recon[INPUT_IDX]), name='train recon input lidar 3')
    else:
        timesteps = np.arange(start=n - N_WINDOW_SIZE + 1, stop=n + 1, step=1)
    trace1 = go.Scatter(x=timesteps, y=np.array(d_train_input[INPUT_IDX])[timesteps],
                        name='train input lidar 3', mode='markers')
    fig.append_trace(trace1, 1, 1)
    # fig.append_trace(trace2, 1, 1)

    fig['layout'].update(title='anomVAE stuff', xaxis={'domain': [0, 1.]})

    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
