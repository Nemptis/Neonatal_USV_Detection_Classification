# Neonatal USV detection and classifiacation

**Welcome!** We encourage you to clone the entire repository to your local machine.
This repository contains algorithms designed for detecting and classifying neonatal ultrasonic vocalizations.

Paper: 
[Enhancing the analysis of murine neonatal ultrasonic vocalizations: Development, evaluation, and application of different mathematical models](https://doi.org/10.48550/arXiv.2405.12957).

Code for the trained models can be found here
<a href="https://doi.org/10.5281/zenodo.15880588"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15880588.svg" alt="DOI"></a>

Code for USV files can be found here
<a href="https://doi.org/10.5281/zenodo.13376980"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13376980.svg" alt="DOI"></a>


# Environment Setup

## Install required packages with Conda

In `environment.yml` is an environment with all required packages defined, which you can install by running the command
```
conda env create -f environment.yml
```
in your terminal.


## Install required packages with pip

You can install the required packages by running `pip install -r requirements.txt`
This will install the dependencies as well as the jupyter kernel.
By default, the latest PyTorch version as of *2024-05-17* will be installed, but the code should also work with older versions.

By default, the code will run on the GPU, if one is available, otherwise it will run on the CPU (see `config.py`).



# Calibration

The first step involves calibrating the detection algorithm to match the settings of your recordings. Follow these steps:

1. Run `Maeuse_dash_entropyratio.py`.
  Utilize the `Maeuse_dash_entropyratio.py` program from the repository.
2. Define the path to the `.WAV` files.
  Set a new path to your personal `.WAV` files inside of `Maeuse_dash_entropyratio.py` and save the file.
3. Activate the Program.
   Open your shell (Windows) or terminal (Mac), navigate to your local repository, and run the command `python Maeuse_dash_entropyratio.py`.
4. Access the Link.
   The shell or terminal will provide a link (usually starting with https://...).
   Copy this link and open it in your preferred browser (e.g. Safari).
5. Adjust Parameters.
   Modify the parameters *ENTROPYTHRESHOLD*, *RATIOTHRESHOLD*, *NDETECT*, and *NGAP* to fit your requirements.
6. Update Values.
   Update these values in the `detection.py` file.
7. Start Jupyter Lab.
   Launch Jupyter Lab, open `Rudolf_Net.ipynb` and execute the code.



# Result Analysis

Upon execution, a "results" folder will be automatically created within your local repository, containing a .csv file with the results. 
Additionally, an "images" folder will include visualizations of all considered spectrograms for manual proofreading.
