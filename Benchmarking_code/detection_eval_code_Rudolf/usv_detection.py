#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:50:35 2023

@author: lgranzow
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from detection import detect_entropyratio
from utils.utils import save_csv

# spectrogram parameters
WINDOWLENGTH = 256  # 250 = 1ms
OVERLAP = 0

# parameter setting
ENTROPYTHRESHOLD = 3.50
RATIOTHRESHOLD = 2
NDETECT = 5
NGAP = 15


def construct_csv_from_wav_file(wav_file, savefile_csv=None):
    if savefile_csv is None:
        savefile_csv = wav_file[:-4] + "_detections.csv"

    samplerate, data = wavfile.read(wav_file)

    # computing spectrogram
    f, t, Sxx = signal.spectrogram(
        data, samplerate, nperseg=WINDOWLENGTH, noverlap=OVERLAP
    )

    # detection
    detection, _, _ = detect_entropyratio(
        Sxx,
        f,
        Nd=NDETECT,
        Ng=NGAP,
        entropythreshold=ENTROPYTHRESHOLD,
        ratiothreshold=RATIOTHRESHOLD,
    )

    # if there is part of a call detected at the very beginning or end of the file, delete that call.
    if detection[0] == 1:
        i = 0
        while detection[i] == 1:
            detection[i] = 0
            i += 1
    if detection[-1] == 1:
        i = -1
        while detection[i] == 1:
            detection[i] = 0
            i -= 1

    starttimes = t[
        np.where(np.diff(detection) == 1)[0] + 1
    ]  # np.where finds the index before the step, so I am adding 1 here?
    endtimes = t[np.where(np.diff(detection) == -1)[0]]

    # save detections as csv
    start_end = np.stack((starttimes, endtimes), axis=1).astype(np.float32)
    # replace the . by ,  cause later the mouse_dataset code uses , as decimal and not .
    start_end = np.char.replace(start_end.astype(str), ".", ",")
    save_csv(savefile_csv, start_end)
    return savefile_csv
