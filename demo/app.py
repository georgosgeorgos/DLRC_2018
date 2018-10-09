import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from plotly.graph_objs import *
import numpy as np
from demo.sampler_app import SamplerAnomalyClustering
import json
import time
from collections import defaultdict
import py_at_broker as pab
import src.utils.configs as cfg

app = dash.Dash(__name__)

broker = pab.broker()
print(broker.request_signal("franka_lidar", pab.MsgType.franka_lidar))
time.sleep(0.5)
print(broker.request_signal("franka_state", pab.MsgType.franka_state))
time.sleep(0.5)

N_SAMPLES = 1
N_INTERVAL_UPDATE = 1.
N_STD = 3
N_MAX_INTERVALS = 100
N_LIDAR_IDX = [3]

# Lists that store data coming in over time
list_lidar_depth = defaultdict(list)
list_lidar_depth_mean = defaultdict(list)
list_lidar_depth_std = defaultdict(list)
list_prob_anomaly = defaultdict(list)
list_prob_normal = defaultdict(list)

p = SamplerAnomalyClustering(n=N_SAMPLES)


################################
#### APP LAYOUT
################################

def lidar_viz(lidar_id):
    return html.Div([
        html.H3("Status LiDAR Sensor: {}".format(lidar_id), style={
            "font-weight": "bold",
            "text-align": "center",
            "margin": 5,
            "border": 2,
            "border-radius": 5,
            "border-color": "#808B96",
            "color": "#FFFFFF",
            "background-color": "#808B96"
        }
                ),
        html.Div([
            dcc.Graph(id='live-update-graph-lidar{}'.format(lidar_id)),
            dcc.Graph(id='live-update-graph-anom-lidar{}'.format(lidar_id)),
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='live-update-bar-chart-clustering-two-classes-lidar{}'.format(lidar_id))
        ], className="three columns"),
    ], className="row")


app.layout = html.Div(
    [
        dcc.Interval(
            id='interval-component',
            interval=N_INTERVAL_UPDATE * 1000,
            n_intervals=N_MAX_INTERVALS
        )] +
    [lidar_viz(id) for id in N_LIDAR_IDX] +
    [html.Div(id='store-data-lidars', style={'display': 'none'})]
)


################################
#### CALLBACKS
################################

def create_callback_probs_clustering(id):
    def callback(n, json_data):
        data = json.loads(json_data)
        probs_classes = data['cluster'][-1][id]

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
            title='Stage two: Clustering',
            showlegend=False
        )

        return Figure(data=[trace], layout=layout)

    return callback


def create_callback_anomaly_clustering(id):
    def callback(n, json_data):
        data = json.loads(json_data)

        list_prob_anomaly[id].append(data['prob'][-1][id])
        list_prob_normal[id].append(1)
        timesteps = np.arange(len(list_prob_anomaly[id]))

        trace = go.Scatter(
            x=timesteps,
            y=list_prob_normal[id],
            mode='lines',
            fill='tozeroy',
            fillcolor='#ABEBC6',
            name='normal',
            line=dict(color='#239B56', smoothing=0.5, shape='spline')
        )

        trace_b = go.Scatter(
            x=timesteps,
            y=list_prob_anomaly[id],
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

    return callback


def create_callback_lidar_graph(id):
    def callback(n, json_data):
        data = json.loads(json_data)

        list_lidar_depth[id].append(data['input'][-1][id])
        list_lidar_depth_mean[id].append(data['mu'][-1][id])
        list_lidar_depth_std[id].append(data['std'][-1][id])

        timesteps = np.arange(len(list_lidar_depth[id]))

        trace_inp = go.Scatter(
            x=timesteps,
            y=list_lidar_depth[id],
            mode='lines',
            name='lidar',
            line=dict(color='#000000', smoothing=0.5, shape='spline')
        )

        trace_mean = go.Scatter(
            x=timesteps,
            y=list_lidar_depth_mean[id],
            mode='lines',
            name='mean',
            line=dict(color='#E74C3C', smoothing=0.5, shape='spline')
        )

        trace_upper_std = go.Scatter(
            x=timesteps,
            y=np.array(list_lidar_depth_mean[id]) + N_STD * np.array(list_lidar_depth_std[id]),
            mode='lines',
            name='conf interval',
            fill='tonexty',
            fillcolor='rgba(174, 214, 241, 0.5)',
            line=dict(color='rgba(174, 214, 241, 0.5)')
        )

        trace_lower_std = go.Scatter(
            x=timesteps,
            y=np.array(list_lidar_depth_mean[id]) - N_STD * np.array(list_lidar_depth_std[id]),
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

    return callback


@app.callback(Output('store-data-lidars', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_data(n):

    # msg_lidar = broker.recv_msg("franka_lidar", -1)

    # msg_state = broker.recv_msg("franka_state", -1)

    data = p.get_data(None, None, n, is_robot=False)

    return json.dumps(data)


# @app.callback(Output('live-update-bar-chart-clustering-three-classes', 'figure'),
#               [
#                   Input('interval-component', 'n_intervals'),
#                   Input('store-data-lidar3', 'children')
#               ])
# def update_barchart_clustering_three(n, json_data):
#     data = json.loads(json_data)
#     probs_classes = data['cluster_n'][0]
#
#     trace = go.Bar(
#         y=['background', 'thyself', 'other agents'],
#         x=probs_classes,
#         orientation='h',
#         marker=dict(
#             color=['#D6DBDF', '#273746',
#                    '#b30000']),
#         width=1
#     )
#
#     layout = Layout(
#         xaxis=dict(
#             title='Class probability',
#             range=[0, 1]
#         ),
#         height=400,
#         title='Clustering of exteroceptive signals into three classes',
#         showlegend=False
#     )
#
#     return Figure(data=[trace], layout=layout)


# Create separate callbacks for each lidar
for lidar_idx in N_LIDAR_IDX:
    app.callback(Output('live-update-bar-chart-clustering-two-classes-lidar{}'.format(lidar_idx), 'figure'),
                 [
                     Input('interval-component', 'n_intervals'),
                     Input('store-data-lidars', 'children')
                 ])(create_callback_probs_clustering(lidar_idx))
    app.callback(Output('live-update-graph-anom-lidar{}'.format(lidar_idx), 'figure'),
                 [
                     Input('interval-component', 'n_intervals'),
                     Input('store-data-lidars', 'children')
                 ])(create_callback_anomaly_clustering(lidar_idx))
    app.callback(Output('live-update-graph-lidar{}'.format(lidar_idx), 'figure'),
                 [
                     Input('interval-component', 'n_intervals'),
                     Input('store-data-lidars', 'children')
                 ])(create_callback_lidar_graph(lidar_idx))

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
