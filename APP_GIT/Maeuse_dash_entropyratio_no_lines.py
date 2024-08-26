
from dash import Dash, html, dcc, callback, Output, Input

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.io import wavfile
from detection import detect_entropyratio
#import dash_daq as daq

samplerate, data = wavfile.read('MAGEL2BeispielZweiteKohorte.WAV')

# spectrogram parameters
WINDOWLENGTH = 256    # 250 = 1ms
OVERLAP = 0

#initial parameter setting
ENTROPYTHRESHOLD = 3.5
RATIOTHRESHOLD = 2.0
NDETECT = 5
NGAP = 15
DENOISE_MODE = None

#computing spectrogram
f, t, Sxx = signal.spectrogram(data,samplerate,nperseg=WINDOWLENGTH,noverlap=OVERLAP)

#initial detection
detection,ratio,entropy = detect_entropyratio(Sxx,f,Nd=NDETECT,Ng=NGAP,entropythreshold=ENTROPYTHRESHOLD,ratiothreshold=RATIOTHRESHOLD)

#doing N 30 second snippets starting from the beginning; (full 5 minutes would be N=10)
N = 5

#build figure
fig = make_subplots(rows=N, cols=1)
fig.update_layout(width=3000,height=1000)
#fig.update_layout(yaxis_range=[0,125000])
#fig.update_layout(yaxis3 = dict(range=[0, 125000]))
fig.update_yaxes(range=[0, 125000])

for i in range(1,N+1):
  idx_t = ((t>(i-1)*30) & (t<i*30))
  t_selected = t[idx_t]
  Sxx_selected = Sxx[:,idx_t]
  detection_selected = detection[idx_t]
  fig.add_trace(
    go.Heatmap(
        z=np.log(Sxx_selected),
        x=t_selected,
        y=f,
        zmin=-2.,
        zmax=2.,
        colorscale='Greys'),
    row=i, col=1
    )
  fig.add_trace(
    go.Scatter(
        x=t_selected,
        y=50000*detection_selected,
        mode="lines",
        line=go.scatter.Line(color="orange")),
    row=i,col=1
    )
  
'''
#add ratio and entropy functions (separate loop to not mess up the trace numbers)
for i in range(1,N+1):
    idx_t = ((t>(i-1)*30) & (t<i*30))
    t_selected = t[idx_t]
    entropy_selected = entropy[idx_t]
    ratio_selected = ratio[idx_t]
    fig.add_trace(
        go.Scatter(
            x=t_selected,
            y=5000*ratio_selected,
            mode="lines",
            line=go.scatter.Line(color="red"),
            visible=False),
        row=i,col=1
    )
    fig.add_trace(go.Scatter(
            x=[t_selected[0],t_selected[-1]],
            y=2*[5000*RATIOTHRESHOLD],
            mode="lines",
            line=go.scatter.Line(color="pink"),
            visible=False),
        row=i,col=1
    )
'''

#fig.show()

# Initialize the app
app = Dash(__name__)

# App layout
# setting slider range and steps here
entropymin, entropymax, entropystep = 3,4,0.01
ratiomin, ratiomax, ratiostep = 1,2,0.01
app.layout = html.Div([
    html.Div(children='Entropy Threshold'),
    dcc.Slider(entropymin, entropymax, entropystep,
                marks= {round(each,2) : {"label": str(round(each,2)), "style": {"transform": "rotate(45deg)"}} for each in np.arange(entropymin, entropymax, entropystep)},
                value=ENTROPYTHRESHOLD,
                id='entropythreshold',
    ),
    html.Div(children='Ratio Threshold'),
    dcc.Slider(ratiomin, ratiomax, ratiostep,
                marks= {round(each,2) : {"label": str(round(each,2)), "style": {"transform": "rotate(45deg)"}} for each in np.arange(ratiomin, ratiomax, ratiostep)},
                value=RATIOTHRESHOLD,
                id='ratiothreshold',
    ),
    html.Div(children='NDETECT'),
    dcc.Slider(1, 10, 1,
                value=NDETECT,
                id='ndetect',
    ),
    html.Div(children='NGAP'),
    dcc.Slider(1, 30, 1,
                value=NGAP,
                id='ngap',
    ),
    dcc.Graph(figure=fig, id='spectrogram')
])

# Add controls to build the interaction
@callback(
      Output(component_id='spectrogram', component_property='extendData'),
      Input(component_id='entropythreshold', component_property='value'),
      Input(component_id='ratiothreshold', component_property='value'),
      Input(component_id='ndetect', component_property='value'),
      Input(component_id='ngap', component_property='value')
)
def update_data(entropythresh,ratiothresh,ndetect,ngap):
    detection,_,_ = detect_entropyratio(Sxx,f,Nd=ndetect,Ng=ngap,entropythreshold=entropythresh,ratiothreshold=ratiothresh)
    y=[]
    for i in range(1,N+1):
      idx_t = ((t>(i-1)*30) & (t<i*30))
      y.append(50000*detection[idx_t])
    return dict(y=y), list(range(1,2*N,2)), dict(y=[len(yi) for yi in y])
    #return a tuple of [updateData, traceIndices, maxPoints]
    # updateData is a dict with the data to update
    # e.g. y=y, and y is a list of the updated data
    # traceIndices is the corresponding indices of the traces to update (e.g. of the same length as y)
    # maxPoints is either an integer with the maximum number of points in each trace,
    #or a dict with the same keys as udpateData, and y is a list of the maximum number of points for each trace separately
    # since the recordings are not exactly 5 minutes, do this separately to account for the last trace with a different length
    #len(t[idx_t]) documented here https://dash.plotly.com/dash-core-components/graph
    



# Run the app
if __name__ == '__main__':
    #debug mode: when turned on, automatically updates the app when this script is changed
    app.run_server(debug=True)

