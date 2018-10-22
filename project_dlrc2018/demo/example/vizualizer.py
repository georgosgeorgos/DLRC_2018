import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from franka_config import FrankaConfig
import py_at_broker as pab
import numpy as np

lidar_num = FrankaConfig.data.lidar_num
broker_id = FrankaConfig.connections.broker_server_id
# Number of graphs for joint visualizations
N_STATES_VIZ = 4
# Number of all the joints
N_STATES = FrankaConfig.data.state_size
# Template for ids
JOINT_ID_TEMP = 'joint-{}'  
LIDAR_ID_TEMP = 'lidar-{}'
# Prediction interval
PRED_INTERVAL = 3    # in seconds
# Interval for real robot data
STATE_INTERVAL = 0.1
# Length os sequences
SEQ_LEN = 30
# Real joint message
REAL_JOINTS_MSG = FrankaConfig.connections.state_channel_id
# Predicted joints message
PRED_JOINTS_MSG  = FrankaConfig.connections.state_pred_channel_id

REAL_LIDAR_MSG = FrankaConfig.connections.lidar_channel_id
PRED_LIDAR_MSG = FrankaConfig.connections.lidar_pred_channel_id

JOINT_OPTIONS = [
        {'label': FrankaConfig.robot.state_names[i],
         'value': '{}'.format(i)} for i in range(N_STATES)
]

# Joint predicted data which should be update
# Warning: storing data in a global scope will work only with one client!
# In our case we should be fine with with
# Multiplied by two because we also get the predicted speeds
J_PRED_DATA = np.zeros((SEQ_LEN, N_STATES))
J_PRED_VAR  = np.zeros((SEQ_LEN, N_STATES))

# Real state data should be updated sequentially so I think we should just
# use a list of list and append new values as they come
def get_empty_j_real_data():
    return [[] for i in range(N_STATES)]

J_REAL_DATA = get_empty_j_real_data() 

# Here I assume that there is only one lidar
# Should be changed if that is not the case
L_PRED_DATA = np.zeros((SEQ_LEN, lidar_num))
L_REAL_DATA = np.zeros((SEQ_LEN)) 

TIME = np.arange(SEQ_LEN)

# TODO: need to add data structure for variances of predictions and finish vizualizations

def refresh_joint_graph(viz_joint):
    std = np.sqrt(J_PRED_VAR)

    upper_bound_pred = go.Scatter(
        name='mean + 3std',
        x=TIME,
        y=J_PRED_DATA[:, viz_joint] + 3*std,
        mode='lines',
        marker=dict(color="444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    pred_mean = go.Scatter(
        name='mean prediction',
        x=TIME,
        y=J_PRED_DATA[:, viz_joint],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound_pred = go.Scatter(
        name='mean - 3std',
        x=TIME,
        y=J_PRED_DATA[:, viz_joint] - 3*std,
        marker=dict(color="444"),
        line=dict(width=0),
        mode='lines')

    real_state = go.Scatter(
                    x=TIME,
                    y=J_REAL_DATA[viz_joint],
                    mode='lines',
                    name='Real')

    return go.Figure(
            data=[
                upper_bound_pred,
                pred_mean,
                lower_bound_pred,
                real_state
            ],
            layout=go.Layout(
                legend=go.Legend(
                    x=0,
                    y=1.0
                ),
                margin=go.Margin(l=30, r=30, t=50, b=30),
                xaxis=dict(range=[0, 30]),
                yaxis=dict(range=[-200, 200])
                )
            )

def refresh_lidar_graph():
    return go.Figure(
            data=[
                go.Bar(x=TIME, y=L_PRED_DATA[:, 0], opacity=0.75, name='Predicted'),
                go.Bar(x=TIME, y=L_REAL_DATA, opacity=0.75, name='Real')
                ],
            layout=go.Layout(barmode='group')
            )


def joint_viz(id):
    return html.Div([
                dcc.Dropdown(
                     options=JOINT_OPTIONS,
                     value='{}'.format(id),
                     id='dropdown-joint-{}'.format(id)
                ),

                dcc.Graph(
                    id=JOINT_ID_TEMP.format(id)
                )],

                style={'width': 550, 'display': 'inline-block',
                       'margin-left': 20, 'margin-right': 20})

def lidar_viz(id):
    return html.Div([
                dcc.Graph(
                    id=LIDAR_ID_TEMP.format(id)
                )],

                style={'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'})

app = dash.Dash()
broker = pab.broker(broker_id)
broker.request_signal(REAL_JOINTS_MSG, pab.MsgType.NumpyMsg)
broker.request_signal(PRED_JOINTS_MSG, pab.MsgType.NumpyMsg)
broker.request_signal(PRED_LIDAR_MSG, pab.MsgType.NumpyMsg)
broker.request_signal(REAL_LIDAR_MSG, pab.MsgType.NumpyMsg)

# Layout of the whole gui
app.layout = html.Div([
    html.H1('Joint Visualizations'),
    html.Div([joint_viz(i) for i in range(N_STATES_VIZ)]),
    html.H1('Lidar Visualizations'),
    html.Div(lidar_viz(0)),
    dcc.Interval(
        id='state-interval',
        interval=STATE_INTERVAL*1000, # in milliseconds
        n_intervals=0
    ),

    html.Div('', id='real-div', style={'display':'none'}),
    html.Div('', id='real-div-0', style={'display':'none'})
    ], 
    style={'font-family': 'Helvetica'}
)


def update_joint_pred():
    global J_PRED_DATA
    msg = broker.recv_msg(PRED_JOINTS_MSG, 0)
    if msg is not None:
        data = msg.get()
        mean, var = data[:N_STATES], data[:N_STATES]
        J_PRED_DATA = mean
        #TODO: fill std
    return ''

@app.callback(Output('real-div', 'children'),
              [Input('state-interval', 'n_intervals')])
def update_joint_real(timestep):
    timestep %= SEQ_LEN
    global J_REAL_DATA
    # Check if the sequence is already full, if so clear the states and run the prediction
    if timestep == 0:
        J_REAL_DATA = get_empty_j_real_data() 
        update_joint_pred()

    msg = broker.recv_msg(REAL_JOINTS_MSG, 0)
    if msg is not None:
        joints = msg.get().reshape((-1))
        for i, joint in enumerate(joints):
            J_REAL_DATA[i].append(joint)
    return ''

@app.callback(Output(JOINT_ID_TEMP.format(0), 'figure'),
              [Input('real-div', 'children')],
              [State('dropdown-joint-{}'.format(0), 'value')])
def update_0(time, index):
    return refresh_joint_graph(int(index))

@app.callback(Output(JOINT_ID_TEMP.format(1), 'figure'),
              [Input('real-div', 'children')],
              [State('dropdown-joint-{}'.format(1), 'value')])
def update_1(time, index):
    return refresh_joint_graph(int(index))

@app.callback(Output(JOINT_ID_TEMP.format(2), 'figure'),
              [Input('real-div', 'children')],
              [State('dropdown-joint-{}'.format(2), 'value')])
def update_2(time, index):
    return refresh_joint_graph(int(index))

@app.callback(Output(JOINT_ID_TEMP.format(3), 'figure'),
              [Input('real-div', 'children')],
              [State('dropdown-joint-{}'.format(3), 'value')])
def update_2(time, index):
    return refresh_joint_graph(int(index))

def update_lidar_pred():
    global L_PRED_DATA
    msg = broker.recv_msg(PRED_LIDAR_MSG, 0)
    if msg is not None:
        data = msg.get()
        mean, var = data[:lidar_num], data[lidar_num:]
        L_PRED_DATA = mean
        #TODO: var
    return ''

@app.callback(Output('real-div-0', 'children'),
              [Input('state-interval', 'n_intervals')])
def update_lidar_real(timestep):
    timestep %= SEQ_LEN
    global L_REAL_DATA
    # Check if the sequence is already full, if so clear the states and run the prediction
    if timestep == 0:
        L_REAL_DATA = np.zeros(SEQ_LEN)
        L_PRED_DATA = np.zeros((SEQ_LEN, 1, 1))
        update_lidar_pred()

    msg = broker.recv_msg(REAL_LIDAR_MSG, 0)
    if msg is not None:
        lidar = msg.get().ravel()
        L_REAL_DATA[timestep] = lidar
    return ''

@app.callback(Output(LIDAR_ID_TEMP.format(0), 'figure'),
              [Input('real-div-0', 'children')])
def update_lidar(time):
    return refresh_lidar_graph()

if __name__ == '__main__':
    app.run_server(debug=False, port=8052, threaded=True)
