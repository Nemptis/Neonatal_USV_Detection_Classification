# Neonatal_USV_Detection_Classifiacation
This repository contains algorithms designed for detecting and classifying neonatal ultrasonic vocalizations.

%%%%%%%
Welcome
%%%%%%%

We encourage you to clone the entire repository to your local machine.

#Environment Setup
Create a New Environment (Conda): Set up a new environment using Conda and install all necessary dependencies, including Jupyter Lab, for later execution.

You can install the required packages by running: pip install -r requirements.txt
This will install the dependencies as well as the jupyter kernel
By default the latest pytorch version as of 17.05.2024 will be installed, but the code should also work with older versions.

By default the code will run on the gpu, if one is available, otherwise it will run on the cpu (see config.py).

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