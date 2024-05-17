from torch.utils.data import DataLoader
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import os

from usv_detection import construct_csv_from_wav_file
from mouse_dataset import mouse_dataset
from mel_dataset import MouseAudioDataset_RegularSpectrogram
from classification_net_cnn import classification_net_cnn_image_lightning, classification_net_cnn_image_lightning_EfficentNetB5


from utils import get_file_list
from config import DEVICE
from pathlib import Path

import matplotlib.pyplot as plt

# those are the mean and standard deviation values of the normal spectorgram and DB scaled spectrogram
# from the labeled dataset (manual detection and manual classification)
MEAN_SPECTROGRAM = 217957840.0
STD_SPECTROGRAM = 29768316928.0
MEAN_DB_SPECTROGRAM = 58.01118087768555
STD_DB_SPECTROGRAM = 6.819430828094482


DATA_DIR = "data" #"/Users/johannmaass/Desktop/Doktor/ZeTeM/Rudolf_net_2/Data"
MODEL_PATH_CUSTOM_CNN = "models/custom_cnn/epoch=139-step=12880.ckpt"#"/Users/johannmaass/Desktop/Doktor/ZeTeM/Rudolf_net_2/Checkpoints/CustomCNN/version_0/checkpoints/epoch=139-step=12880.ckpt"
MODEL_PATH_EFFICENTNETB5 = "models/efficientnetb5/epoch=19-step=1840.ckpt" #"/Users/johannmaass/Desktop/Doktor/ZeTeM/Rudolf_net_2/Checkpoints/efficentnetb5/version_0/checkpoints/epoch=19-step=1840.ckpt"


def create_dataset(folder_dir, normalize_smooth_spec_individually=False):
    """creates the dataset from a folder that contains the .WAV and detections.csv files

    normalize_smooth_spec_individually: set to False for the custom cnn,
                set to True for the EfficentNetB5
    """

    # use mouse_dataset to extract the whole signal, the start end times and duration of
    # the individual calls
    auto_manu_ds = mouse_dataset.from_folder(
        folder_dir,
        name="auto-manu-set",
        categories=[1, 2, 3, 4, 5],
        category_map={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
        pad_start_ms=60,
        pad_end_ms=60,
        verbose=True,
    )

    # build a new dataset from the data of the mouse_dataset
    dataset = MouseAudioDataset_RegularSpectrogram(
        auto_manu_ds.data,
        mean_spectogram=MEAN_SPECTROGRAM,
        std_spectogram=STD_SPECTROGRAM,
        mean_scaled_spectogram=MEAN_DB_SPECTROGRAM,
        std_scaled_spectogram=STD_DB_SPECTROGRAM,
        final_crop_size_no_aug=170,
        normalize_smooth_spec_individually=normalize_smooth_spec_individually,
        resize_size=None,
    )

    return dataset


def dataset_from_wav_file(wav_file, normalize_smooth_spec_individually=False):
    csv_file = construct_csv_from_wav_file(wav_file)

    auto_mouse_ds = mouse_dataset.from_wav_csv_files(
        wav_files=[wav_file],
        csv_files=[csv_file],
        name="auto-manu-set",
        categories=[1, 2, 3, 4, 5],
        category_map={"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
        pad_start_ms=60,
        pad_end_ms=60,
        verbose=False,
    )

    dataset = MouseAudioDataset_RegularSpectrogram(
        auto_mouse_ds.data,
        mean_spectogram=MEAN_SPECTROGRAM,
        std_spectogram=STD_SPECTROGRAM,
        mean_scaled_spectogram=MEAN_DB_SPECTROGRAM,
        std_scaled_spectogram=STD_DB_SPECTROGRAM,
        final_crop_size_no_aug=170,
        normalize_smooth_spec_individually=normalize_smooth_spec_individually,
        resize_size=None,
    )

    return dataset



def load_model(model_path, model_class):
    model = model_class.load_from_checkpoint(model_path).eval().to(DEVICE)

    return model


def example_run_model(
    data_folder_dir, model_path, model_class, normalize_smooth_spec_individually=False
):
    model = load_model(model_path, model_class)
    dataset = create_dataset(
        data_folder_dir,
        normalize_smooth_spec_individually=normalize_smooth_spec_individually,
    )

    # set the batch_size so that it still fits in VRAM / RAM (depending on what DEVICE is used)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    predictions = []

    # just a dummy loop running the model over the data
    for spectrogram, target in tqdm(dataloader):
        # no training, so no need to track gradients here
        with torch.no_grad():
            pred = model(spectrogram.to(DEVICE))
            predicted_categories = torch.argmax(pred, dim=1).cpu()
            predictions.append(predicted_categories)

    predictions = torch.cat(predictions, dim=0)
    for category_class in range(5):
        print(
            "category: {}, num calls: {}".format(
                category_class + 1, torch.sum(predictions == category_class)
            )
        )


def run_evaluation(data_folder_dir, model_path, model_class, normalize_smooth_spec_individually=False, confidence_threshold=0.0,
                   plot_images=False
):
    model = load_model(model_path, model_class)
    wav_files = get_file_list(data_folder_dir, ext=".WAV")

    num_calls_per_category_csv = ["Number of Calls per Category"]
    categories_csv = ["Call Category"]
    wav_files_csv = ["File Name"]

    for wav_file in wav_files:
        spectrograms_db_scale_per_category = [[] for i in range(6)]
        predictions = []
        dataset = dataset_from_wav_file(
            wav_file,
            normalize_smooth_spec_individually=normalize_smooth_spec_individually,
        )

        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        for spectrogram, _ in tqdm(dataloader):
            with torch.no_grad():
                pred = model(spectrogram.to(DEVICE))
                confidences, _ = torch.max(pred, dim=1)
                predicted_categories = torch.argmax(pred, dim=1)
                for idx, confidence in enumerate(confidences):
                    if confidence > confidence_threshold:
                        predictions.append(predicted_categories[idx].unsqueeze(dim=0).cpu())
                        spectrograms_db_scale_per_category[predicted_categories[idx]+1].append(spectrogram[idx, 1])
                    else:
                        # set to -1 for usv calls skipped due to low confidence
                        predictions.append(torch.tensor(-1,).unsqueeze(dim=0))
                        spectrograms_db_scale_per_category[0].append(spectrogram[idx, 1])

        if len(predictions) > 0:
            predictions = torch.cat(predictions, dim=0)

        # -1 is for usv calls skipped due to low confidence
        for category_class in [i-1 for i in range(6)]:
            if len(predictions) > 0:
                num_calls = int(torch.sum(predictions == category_class).numpy())
            else:
                num_calls = 0
            num_calls_per_category_csv.append(num_calls)
            categories_csv.append(category_class + 1)
            wav_files_csv.append(wav_file.split("/")[-1])

        # add an empty line between wav files, for easier readability
        num_calls_per_category_csv.append("")
        categories_csv.append("")
        wav_files_csv.append("")

        if plot_images:
            Path("results/images/").mkdir(parents=True, exist_ok=True)
            for category, spectrograms in enumerate(spectrograms_db_scale_per_category):
                if len(spectrograms) > 0:
                    # need to be of shape b,c,h,w -> add c=1
                    spectrograms = torch.stack(spectrograms, dim=0).unsqueeze(dim=1)
                    image = torchvision.utils.make_grid(spectrograms, normalize=True, scale_each=True)[0]
                    plt.figure(figsize=(image.shape[0]/100, image.shape[1]/100), dpi=1000)
                    plt.imshow(image)
                    plt.axis('off')
                    wav_file_name = os.path.normpath(wav_file).split(os.path.sep)[-1]
                    plt.savefig("results/images/" + wav_file_name + "_" + str(category) + ".jpg", bbox_inches='tight')
                    plt.close()
                    #torchvision.utils.save_image(image, "results/images/" + wav_file.split("/")[-1] + "_" + str(category) + ".jpg")

        # in case a crash occurs during evaluation
        Path("results/").mkdir(parents=True, exist_ok=True)
        np.savetxt("results/results.csv", [p for p in zip(wav_files_csv, categories_csv, num_calls_per_category_csv)], delimiter=";", fmt='%s')
    Path("results/").mkdir(parents=True, exist_ok=True)
    np.savetxt("results/results.csv", [p for p in zip(wav_files_csv, categories_csv, num_calls_per_category_csv)], delimiter=";", fmt='%s')


# custom cnn

"""
run_evaluation(
    data_folder_dir=DATA_DIR,
    model_path=MODEL_PATH_CUSTOM_CNN,
    model_class=classification_net_cnn_image_lightning,
    confidence_threshold=0.4,
    plot_images=True
)
"""

# efficentnet b5
run_evaluation(
    data_folder_dir=DATA_DIR,
    model_path=MODEL_PATH_EFFICENTNETB5,
    model_class=classification_net_cnn_image_lightning_EfficentNetB5,
    confidence_threshold=0.4,
    normalize_smooth_spec_individually=True,
    plot_images=False,
)







