# Neonatal_USV_Detection_Classifiacation
This repository contains algorithms designed for detecting and classifying neonatal ultrasonic vocalizations.

%%%%%%%
Welcome
%%%%%%%

We encourage you to clone the entire repository to your local machine.

#Environment Setup
Create a New Environment (Conda): Set up a new environment using Conda and install all necessary dependencies, including Jupyter Lab, for later execution.

Install Required Programs:
absl-py                  2.1.0
aiofiles                 22.1.0
aiohttp                  3.9.3
aiosignal                1.3.1
aiosqlite                0.19.0
altgraph                 0.17.2
anyio                    3.6.2
appnope                  0.1.3
argon2-cffi              21.3.0
argon2-cffi-bindings     21.2.0
arrow                    1.2.3
asttokens                2.2.1
async-lru                2.0.4
async-timeout            4.0.3
attrs                    23.1.0
Babel                    2.12.1
backcall                 0.2.0
beautifulsoup4           4.12.2
bleach                   6.0.0
blinker                  1.6.2
certifi                  2023.5.7
cffi                     1.15.1
charset-normalizer       3.1.0
click                    8.1.3
comm                     0.1.3
contourpy                1.1.0
cycler                   0.11.0
dash                     2.9.3
dash-core-components     2.0.0
dash-html-components     2.0.0
dash-table               5.0.0
debugpy                  1.6.7
decorator                5.1.1
defusedxml               0.7.1
executing                1.2.0
fastjsonschema           2.16.3
filelock                 3.13.1
Flask                    2.3.2
fonttools                4.40.0
fqdn                     1.5.1
frozenlist               1.4.1
fsspec                   2024.2.0
future                   0.18.2
grpcio                   1.62.1
icalendar                5.0.11
idna                     3.4
imageio                  2.34.0
imapy                    1.2.1
importlib-metadata       6.6.0
importlib-resources      5.12.0
ipykernel                6.23.0
ipython                  8.13.2
ipython-genutils         0.2.0
ipywidgets               8.1.0
isoduration              20.11.0
itsdangerous             2.1.2
jedi                     0.18.2
Jinja2                   3.1.2
json5                    0.9.11
jsonpointer              2.3
jsonschema               4.17.3
jupyter                  1.0.0
jupyter_client           8.2.0
jupyter-console          6.6.3
jupyter_core             5.3.0
jupyter-events           0.6.3
jupyter-lsp              2.2.0
jupyter_server           2.5.0
jupyter_server_fileid    0.9.0
jupyter_server_terminals 0.4.4
jupyter_server_ydoc      0.8.0
jupyter-ydoc             0.2.4
jupyterlab               4.0.5
jupyterlab-pygments      0.2.2
jupyterlab_server        2.22.1
jupyterlab-widgets       3.0.8
kiwisolver               1.4.4
lazy_loader              0.3
lightning-utilities      0.10.1
macholib                 1.15.2
Markdown                 3.5.2
MarkupSafe               2.1.2
matplotlib               3.7.1
matplotlib-inline        0.1.6
mistune                  2.0.5
mpmath                   1.3.0
multidict                6.0.5
nbclassic                1.0.0
nbclient                 0.7.4
nbconvert                7.4.0
nbformat                 5.8.0
nest-asyncio             1.5.6
networkx                 3.2.1
notebook                 6.5.4
notebook_shim            0.2.3
numpy                    1.24.3
packaging                23.1
pandas                   2.2.1
pandocfilters            1.5.0
parso                    0.8.3
pexpect                  4.8.0
pickleshare              0.7.5
Pillow                   9.5.0
pip                      24.0
platformdirs             3.5.0
plotly                   5.14.1
prometheus-client        0.16.0
prompt-toolkit           3.0.38
protobuf                 4.25.3
psutil                   5.9.5
ptyprocess               0.7.0
pure-eval                0.2.2
pycparser                2.21
pyDeprecate              0.3.2
Pygments                 2.15.1
pyparsing                3.0.9
pyrsistent               0.19.3
python-dateutil          2.8.2
python-json-logger       2.0.7
pytorch-lightning        1.6.0
pytz                     2024.1
PyYAML                   6.0
pyzmq                    25.0.2
qtconsole                5.4.4
QtPy                     2.4.0
requests                 2.30.0
rfc3339-validator        0.1.4
rfc3986-validator        0.1.1
scikit-image             0.22.0
scipy                    1.10.1
Send2Trash               1.8.2
setuptools               58.0.4
six                      1.15.0
sniffio                  1.3.0
soupsieve                2.4.1
stack-data               0.6.2
sympy                    1.12
tenacity                 8.2.2
tensorboard              2.16.2
tensorboard-data-server  0.7.2
terminado                0.17.1
tifffile                 2024.2.12
tinycss2                 1.2.1
tomli                    2.0.1
torch                    1.13.1
torchaudio               0.13.1
torchmetrics             0.11.4
torchvision              0.14.1
tornado                  6.3.1
tqdm                     4.66.2
traitlets                5.9.0
typing_extensions        4.10.0
tzdata                   2024.1
uri-template             1.2.0
urllib3                  1.26.8
wcwidth                  0.2.6
webcolors                1.13
webencodings             0.5.1
websocket-client         1.5.1
Werkzeug                 2.3.4
wheel                    0.37.0
widgetsnbextension       4.0.8
y-py                     0.5.9
yarl                     1.9.4
ypy-websocket            0.8.2
zipp                     3.15.0

Note: Ensure to install the specified versions for each program listed above.

#Calibration
The first step involves calibrating the detection algorithm to match the settings of your recordings. Follow these steps:

Run Maeuse_dash_entropyratio.py: Utilize the Maeuse_dash_entropyratio.py program from the repository.
Define Path to .WAV Files: Set a new path to your personal .WAV files inside Maeuse_dash_entropyratio.py and save the file.
Activate the Program: Open your shell (Windows) or terminal (Mac), navigate to your local repository, and run the command python Maeuse_dash_entropyratio.py.
Access Link: The shell or terminal will provide a link (usually starting with https://...). Copy this link and open it in your preferred browser (e.g., Safari).
Adjust Parameters: Modify the parameters ENTROPYTHRESHOLD, RATIOTHRESHOLD, NDETECT, and NGAP to fit your requirements.
Update Values: Update these values in the detection.py file.
Start Jupyter Lab: Launch Jupyter Lab, open the Rudolf_Net.ipynb file, and execute the code.

#Result Analysis
Upon execution, a "results" folder will be automatically created within your local repository, containing a .csv file with the results. Additionally, an "images" folder will include visualizations of all considered spectrograms for manual proofreading.