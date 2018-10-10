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
N_INTERVAL_UPDATE = 0.75
N_STD = 3
N_LIDAR_IDX = [3]
USE_MOCKUP_DATA = True
ANOMALY_THRESHOLD = .95

# Lists that store data coming in over time
list_lidar_depth = defaultdict(list)
list_lidar_depth_mean = defaultdict(list)
list_lidar_depth_std = defaultdict(list)
list_prob_anomaly = defaultdict(list)
list_prob_normal = defaultdict(list)
list_prob_background = defaultdict(list)
list_prob_self = defaultdict(list)

p = SamplerAnomalyClustering(n=N_SAMPLES)


################################
#### APP LAYOUT
################################

def lidar_viz(lidar_id):
    return html.Div([
        # visualize sensors belonging to other agent state and either self or background
        html.Div([
            html.H3("Status LiDAR Sensor: {}".format(lidar_id), style={
                "font-weight": "bold",
                # "text-align": "center",
                "margin": 10,
                "border": 2,
                "border-radius": 3,
                "border-color": "#808B96",
                "color": "#FFFFFF",
                "background-color": "#808B96"
            }
                    ),
            dcc.Graph(id='live-update-graph-lidar{}'.format(lidar_id)),
            dcc.Graph(id='live-update-graph-anom-lidar{}'.format(lidar_id)),
            dcc.Graph(id='live-update-graph-normal-lidar{}'.format(lidar_id)),
        ], className="six columns"),
        # html.Div([
        #     dcc.Graph(id='live-update-bar-chart-clustering-two-classes-lidar{}'.format(lidar_id))
        # ], className="three columns"),
    ], className="row")


app.layout = html.Div(
    [
        dcc.Interval(
            id='interval-component',
            interval=N_INTERVAL_UPDATE * 1000,
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


def create_callback_normal_graph(id):
    def callback(n, json_data):
        data = json.loads(json_data)
        list_prob_background[id].append(data['cluster'][-1][id][0])
        list_prob_self[id].append(1)
        timesteps = np.arange(len(list_prob_background[id]))

        trace_background = go.Scatter(
            x=timesteps,
            y=np.array(list_prob_background[id]),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(165, 105, 189, .8)',
            name='background',
            line=dict(color='rgba(165, 105, 189, .8)', smoothing=0.5, shape='spline')
        )

        trace_self = go.Scatter(
            x=timesteps,
            y=np.array(list_prob_self[id]),
            mode='lines',
            fill='tozeroy',
            name='thyself',
            fillcolor='rgba(247, 220, 111, .8)',
            line=dict(color='rgba(247, 220, 111, .8)', smoothing=0.5, shape='spline'),
            showlegend=True
        )

        layout = Layout(
            xaxis=dict(
                title='Timesteps'
            ),
            yaxis=dict(
                title='Prob',
                range=[0, 1]
            ),
            height=275,
            showlegend=True,
            legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, .8)'),
        )

        return Figure(data=[trace_self, trace_background], layout=layout)

    return callback


def create_callback_anomaly_graph(id):
    def callback(n, json_data):
        data = json.loads(json_data)

        list_prob_anomaly[id].append(data['prob'][-1][id])
        list_prob_normal[id].append(1)
        timesteps = np.arange(len(list_prob_anomaly[id]))

        mask_anom_thresh = np.array(list_prob_anomaly[id]) > ANOMALY_THRESHOLD

        trace_normal = go.Scatter(
            x=timesteps,
            y=np.array(list_prob_normal[id]),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(133, 193, 233, .8)',
            name='normal',
            line=dict(color='rgba(133, 193, 233, .8)', smoothing=0.5, shape='spline')
        )

        trace_anom = go.Scatter(
            x=timesteps,
            y=np.array(list_prob_anomaly[id]),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(240, 128, 128, .1)',
            line=dict(color='rgba(240, 128, 128, .1)', smoothing=0.5, shape='spline'),
            showlegend=False
        )

        trace_anom_decision = go.Scatter(
            x=timesteps[mask_anom_thresh],
            y=np.array(list_prob_anomaly[id])[mask_anom_thresh],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(240, 128, 128, .8)',
            line=dict(color='rgba(240, 128, 128, .8)', smoothing=0.5, shape='spline'),
            name='other agent'
        )

        layout = Layout(
            yaxis=dict(
                title='Prob',
                range=[0, 1]
            ),
            height=275,
            showlegend=True,
            title='Clustering of sensor measurements',
            legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, .8)'),
        )

        return Figure(data=[trace_normal, trace_anom, trace_anom_decision], layout=layout)

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
            name='input depth',
            line=dict(color='#2ECC71', smoothing=0.5, shape='spline')
        )

        trace_mean = go.Scatter(
            x=timesteps,
            y=list_lidar_depth_mean[id],
            mode='lines',
            name='prediction mean',
            line=dict(color='rgb(231, 76, 60)', smoothing=0.5, shape='spline')
        )

        trace_upper_std = go.Scatter(
            x=timesteps,
            y=np.array(list_lidar_depth_mean[id]) + N_STD * np.array(list_lidar_depth_std[id]),
            mode='lines',
            name='prediction ± {} standard deviation'.format(N_STD),
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.15)',
            line=dict(color='rgba(231, 76, 60, 0.15)')
        )

        trace_lower_std = go.Scatter(
            x=timesteps,
            y=np.array(list_lidar_depth_mean[id]) - N_STD * np.array(list_lidar_depth_std[id]),
            mode='lines',
            showlegend=False,
            fillcolor='rgba(231, 76, 60, 0.15)',
            line=dict(color='rgba(231, 76, 60, 0.15)')
        )

        layout = Layout(
            yaxis=dict(
                title='Depth (m)'
            ),
            height=400,
            showlegend=True,
            title='Sensor measurements and predictions',
            legend=dict(xanchor='right', yanchor='top', bgcolor='rgba(255, 255, 255, 0.25)'),
        )

        return Figure(data=[trace_lower_std, trace_upper_std, trace_inp, trace_mean], layout=layout)

    return callback


@app.callback(Output('store-data-lidars', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_data(n):
    if USE_MOCKUP_DATA:
        data = p.get_data(None, None, n, is_robot=False)
    else:
        msg_lidar = broker.recv_msg("franka_lidar", -1)
        msg_state = broker.recv_msg("franka_state", -1)

        data = p.get_data(msg_lidar, msg_state, n, is_robot=True)

    return json.dumps(data)


# Create separate callbacks for each lidar
for lidar_idx in N_LIDAR_IDX:
    # app.callback(Output('live-update-bar-chart-clustering-two-classes-lidar{}'.format(lidar_idx), 'figure'),
    #              [
    #                  Input('interval-component', 'n_intervals'),
    #                  Input('store-data-lidars', 'children')
    #              ])(create_callback_probs_clustering(lidar_idx))
    app.callback(Output('live-update-graph-normal-lidar{}'.format(lidar_idx), 'figure'),
                 [
                     Input('interval-component', 'n_intervals'),
                     Input('store-data-lidars', 'children')
                 ])(create_callback_normal_graph(lidar_idx)),
    app.callback(Output('live-update-graph-anom-lidar{}'.format(lidar_idx), 'figure'),
                 [
                     Input('interval-component', 'n_intervals'),
                     Input('store-data-lidars', 'children')
                 ])(create_callback_anomaly_graph(lidar_idx))
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
