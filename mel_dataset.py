
import torch
import torch.nn.functional as F
spectrogram_specs=None
import torchaudio
from tqdm import tqdm
import random
import numpy as np
import os
import copy

from config import DEVICE

def train_val_test_split(dataset, force_new=False, prefix=""):
    num_val_samples = int(len(dataset)*0.1)
    num_test_samples = int(len(dataset)*0.1)

    if not os.path.exists(prefix + "train_indices.npy") or force_new:
        val_indices = []
        test_indices = []
        train_indices = []

        while len(val_indices) < num_val_samples:
            i = random.randint(0, len(dataset))
            if not i in val_indices:
                val_indices.append(i)

        while len(test_indices) < num_test_samples:
            i = random.randint(0, len(dataset))
            if not i in val_indices:
                if not i in test_indices:
                    test_indices.append(i)

        for i in range(len(dataset)):
            if not i in val_indices:
                if not i in test_indices:
                    train_indices.append(i)

        if not force_new:
            np.save(prefix + "train_indices.npy", train_indices)
            np.save(prefix + "val_indices.npy", val_indices)
            np.save(prefix + "test_indices.npy", test_indices)

    else:
        train_indices = np.load(prefix + "train_indices.npy")
        val_indices = np.load(prefix + "val_indices.npy")
        test_indices = np.load(prefix + "test_indices.npy")

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    return train_ds, val_ds, test_ds



def combine_data(data_list):
    combined_data = []

    for data in data_list:
        for d in data:
            combined_data.append(d)

    return combined_data



class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.build_map()


    def build_map(self):
        index_dataset_map = []
        i = 0
        for dataset in self.datasets:
            for j in range(len(dataset)):
                index_dataset_map.append([j, dataset])
        self.index_dataset_map = index_dataset_map


    def __len__(self):
        final_len = 0
        for dataset in self.datasets:
            final_len += len(dataset)

        return final_len


    def __getitem__(self, index):
        #print("index: {}".format(index))
        idx, dataset = self.index_dataset_map[index]
        return dataset.__getitem__(idx)



class MouseAudioDataset(torch.utils.data.Dataset):
    """
    Uses the data of the mouse_dataset to construct a new dataset.
    It uses the detection times of the mouse_dataset to extract the signal of the data.
    Then it constructs its own spectrogram.


    data: the data dict from the mouse_dataset

    mean_spectrogram, std_spectrogram: The smooth spectrograms will be normalized by those values,
                if do_normalization=True and normalize_smooth_spec_individually=False

    mean_scaled_spectogram, std_scaled_spectogram: The DB scale spectrograms will be normalized by those values,
                if do_normalization=True

    roll_amount: Amount of rolling in time axis as data augmentation, only used if use_augmentations=True (in training)

    use_augmentations: Whether to use data augmentations or not. Set to False for validation

    do_normalization: Whether to normalize the smooth and DB scale spectrogram.
                Set to False when computing the mean and std of the dataset

    pad_to_same_size: Whether to pad the spectrograms to a fixed size.

    final_crop_size_no_aug: The crop size in the time axis if no augmentation is used (170 seems to work best here)

    resize_size: Will resize the final spectrograms to this size. Do nothing if it is None.

    use_mixup: Whether to use mixup augmentation

    normalize_smooth_spec_individually: Normalize the smooth spectrogram individually if this is set to True.
                Should be True for the EfficentNetB5, ResNet50 and ResNet34 models.

    use_amplitude_jitter: Jitter the amplitude of the spectrograms as augmentation if this is True.
    """

    def __init__(self, data, mean_spectogram=315564192.0, std_spectogram=42108116992.0,
                 mean_scaled_spectogram=53.20830154418945, std_scaled_spectogram=13.420377731323242,
                 roll_amount=10, use_augmentations=False, do_normalization=True, pad_to_same_size=True,
                 final_crop_size_no_aug=170, resize_size=None, use_mixup=False,
                 normalize_smooth_spec_individually=False, use_amplitude_jitter=False):
        self.data = data
        self.roll_amount = roll_amount

        self.mean_spectogram = mean_spectogram
        self.std_spectogram = std_spectogram

        self.mean_scaled_spectogram = mean_scaled_spectogram
        self.std_scaled_spectogram = std_scaled_spectogram

        self.do_normalization = do_normalization

        self.use_augmentations = use_augmentations
        self.pad_to_same_size = pad_to_same_size

        #self.frequency_masking_transforms = [torchaudio.transforms.FrequencyMasking(freq_mask_param=5) for i in range(5)]
        #self.time_masking_transforms = [torchaudio.transforms.TimeMasking(time_mask_param=5) for i in range(5)]

        self.final_crop_size_no_aug = final_crop_size_no_aug

        self.resize_size = resize_size
        self.use_mixup = use_mixup
        self.normalize_smooth_spec_individually = normalize_smooth_spec_individually
        self.use_amplitude_jitter = use_amplitude_jitter



    def signal_to_spectrogram(self, signal, sample_rate=None):
        return torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(signal)


    def normalize_spectrograms(self, spectrogram, db_scaled_spectrogram):
        if not self.normalize_smooth_spec_individually:
            spectrogram = (spectrogram - self.mean_spectogram) / self.std_spectogram
        else:
            spectrogram = (spectrogram - torch.mean(spectrogram)) / torch.std(spectrogram)
        db_scaled_spectrogram = (db_scaled_spectrogram - self.mean_scaled_spectogram) / self.std_scaled_spectogram

        return spectrogram, db_scaled_spectrogram


    def create_dataset_from_data_list(self, data_list):
        return MouseAudioDataset_RegularSpectrogram(data_list, self.mean_spectogram, self.std_spectogram,
                                                    self.mean_scaled_spectogram, self.std_scaled_spectogram, self.roll_amount,
                                                    self.use_augmentations, self.do_normalization, self.pad_to_same_size,
                                                    self.final_crop_size_no_aug)


    def select_wav_files_by_name(self, contain_any_of):
        filtered_data = []
        for sample in self.data:
            wav_file_name = sample['wav_file']
            for substring in contain_any_of:
                if substring in wav_file_name:
                    filtered_data.append(sample)
                    break

        return self.create_dataset_from_data_list(filtered_data)


    def select_wav_files_by_genotype(self, genotype):
        filtered_data = []
        for sample in self.data:
            wav_file_name = sample['wav_file']
            split_name = copy.deepcopy(wav_file_name).split("_")
            if len(split_name) == 8:
                if wav_file_name.split(".")[0][-1] == genotype:
                    filtered_data.append(sample)
            else:
                if wav_file_name.split("_")[-2] == genotype:
                    filtered_data.append(sample)

        return self.create_dataset_from_data_list(filtered_data)


    def select_wav_files_by_mouse_id(self, mouse_id):
        filtered_data = []
        for sample in self.data:
            wav_file_name = sample['wav_file']
            #print(wav_file_name)
            #print(wav_file_name.split(".")[0])
            #print(wav_file_name.split("_"))
            split_name = copy.deepcopy(wav_file_name).split("_")

            if len(split_name) == 8:
                #print("mouse_id: {}, in data: {}".format(mouse_id, split_name[-2]))
                if str(split_name[-2]) == str(mouse_id):
                    filtered_data.append(sample)
            else:
                #print("mouse_id: {}, in data: {}".format(mouse_id, split_name[-3]))
                if str(split_name[-3]) == str(mouse_id):
                    filtered_data.append(sample)


        return self.create_dataset_from_data_list(filtered_data)



    def select_wav_files_by_mother_id(self, mouse_id):
        filtered_data = []
        for sample in self.data:
            wav_file_name = sample['wav_file']
            #print(wav_file_name)
            #print(wav_file_name.split(".")[0])
            #print(wav_file_name.split("_"))
            split_name = copy.deepcopy(wav_file_name).split("_")

            if len(split_name) == 8:
                #print("mouse_id: {}, in data: {}".format(mouse_id, split_name[-2]))
                if str(split_name[-4]) == str(mouse_id):
                    filtered_data.append(sample)
            else:
                #print("mouse_id: {}, in data: {}".format(mouse_id, split_name[-3]))
                if str(split_name[-5]) == str(mouse_id):
                    filtered_data.append(sample)


        return self.create_dataset_from_data_list(filtered_data)


    def select_data_by_class(self, category):
        filtered_data = []
        for sample in self.data:
            if sample['category'] == category:
                filtered_data.append(sample)

        return self.create_dataset_from_data_list(filtered_data)


    def __getitem__(self, index):
        if not self.use_mixup:
            return self.getitem_internal(index)
        else:
            spectrogram, target = self.getitem_internal(index)
            other_index = random.randint(0, len(self)-1)
            spectrogram_other, target_other = self.getitem_internal(other_index)

            alpha = random.uniform(0, 1)

            spectrogram_mixed = spectrogram*alpha + spectrogram_other*(1-alpha)
            target_mixed = target*alpha + target_other*(1-alpha)

            return spectrogram_mixed, target_mixed




    def getitem_internal(self, index):
        if index >= len(self.data):
            print("getitem index: {}".format(index))
            print("self len: {}".format(len(self.data)))
        #print("use augmentations: {}".format(self.use_augmentations))
        sample = self.data[index]
        signal = sample['padded_signal']
        sampling_rate = sample['sampling_rate']
        rel_duration = (sample['end_time'] - sample['start_time']) / 0.15

        #print("signal shape before: {}".format(signal.shape))
        signal = torch.tensor(signal).float()
        #print("signal shape: {}".format(signal.shape))
        #print("sampling_rate: {}".format(sampling_rate))

        spectrogram = self.signal_to_spectrogram(signal, sampling_rate) #torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate)(signal)
        db_scaled_spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=None)(spectrogram)

        #print("spectrogram shape: {}".format(spectrogram.shape))
        #db_scaled_spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=None)(spectrogram)


        if not self.pad_to_same_size:
            if self.do_normalization:
                spectrogram, db_scaled_spectrogram = self.normalize_spectrograms(spectrogram, db_scaled_spectrogram)
            spectrogram = torch.stack([spectrogram, db_scaled_spectrogram], dim=0)
            time_feature = torch.ones_like(spectrogram[0].unsqueeze(dim=0)) * torch.tensor(rel_duration).float().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
            spectrogram = torch.cat([spectrogram.float(), time_feature], dim=0)
            return spectrogram, sample['one_hot_category']


        #print(torch.mean(db_scaled_spectrogram))

        length = spectrogram.shape[1]
        #wanted_length = min(length - 40, 170)
        wanted_length = min(length - 100, 170)

        if self.use_augmentations:
            roll_amount = random.randint(-self.roll_amount, self.roll_amount)
            spectrogram = torch.roll(spectrogram, roll_amount, dims=1)
            db_scaled_spectrogram = torch.roll(db_scaled_spectrogram, roll_amount, dims=1)

        start_crop = int((length - wanted_length) / 2)
        end_crop = start_crop + wanted_length

        spectrogram = spectrogram[:, start_crop:end_crop]
        db_scaled_spectrogram = db_scaled_spectrogram[:, start_crop:end_crop]

        #print(mel_spectrogram.shape)

        if self.do_normalization:
            spectrogram, db_scaled_spectrogram = self.normalize_spectrograms(spectrogram, db_scaled_spectrogram)

        spectrogram = torch.stack([spectrogram, db_scaled_spectrogram], dim=0).unsqueeze(dim=0)
        if self.use_amplitude_jitter:
            spectrogram *= random.uniform(0.7, 1.3)

        #print("spectogram.shape: {}".format(spectogram.shape))

        length = spectrogram.shape[-1]
        wanted_length = 190
        pad_left = int((wanted_length - length) / 2)
        pad_right = wanted_length - pad_left - length
        #print("spectrogram shape before pad: {}".format(spectrogram.shape))
        # hotfix for spectrograms with 0 length
        if spectrogram.shape[-1] == 0:
            spectrogram = torch.zeros((1, 2, 201, 190))
        else:
            spectrogram = torch.nn.functional.pad(spectrogram, (pad_left, pad_right, 0, 0), mode='replicate')

        #print("spectogram.shape after pad: {}".format(spectogram.shape))

        if self.use_augmentations:
            new_length = random.randint(170, 210)
            spectrogram = torch.nn.functional.interpolate(spectrogram, size=(spectrogram.shape[2], new_length))[0]
        else:
            spectrogram = spectrogram[0]

        #print("spectogram.shape after interp: {}".format(spectogram.shape))

        if self.use_augmentations:
            #print("use augmentations...")
            start_crop = random.randint(0, new_length - 150)
            end_crop = start_crop + 150
            spectrogram = spectrogram[:, :, start_crop:end_crop]
            #spectrogram += torch.randn_like(spectrogram)*0.01
            freq_shift = random.randint(-10, 10)
            spectrogram = torch.roll(spectrogram, shifts=freq_shift, dims=1)
            #for transform in self.frequency_masking_transforms:
            #    spectrogram = transform(spectrogram)
            #for transform in self.time_masking_transforms:
            #    spectrogram = transform(spectrogram)
        else:
            length = spectrogram.shape[-1]
            wanted_length = self.final_crop_size_no_aug
            start_crop = int((length - wanted_length) / 2)
            end_crop = start_crop + wanted_length

            spectrogram = spectrogram[:, :, start_crop:end_crop]


        #print("spectogram.shape final: {}".format(spectrogram.shape))

        time_feature = torch.ones_like(spectrogram[0].unsqueeze(dim=0)) * torch.tensor(rel_duration).float().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        spectrogram = torch.cat([spectrogram.float(), time_feature], dim=0)

        if self.resize_size is not None:
            spectrogram = F.interpolate(spectrogram.unsqueeze(dim=0), size=self.resize_size)[0]

        if self.use_augmentations:
            spectrogram += torch.randn_like(spectrogram)*0.01
        #print("spectorgram shape: {}".format(spectogram.shape))

        #print("spectrogram shape: {}".format(spectrogram.shape))

        return spectrogram, sample['one_hot_category']


        #return spectogram.float(), torch.tensor(rel_duration).float().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0), sample['one_hot_category']


    def __len__(self):
        return len(self.data)



class MouseAudioDataset_RegularSpectrogram(MouseAudioDataset):
    """ See MouseAudioDataset """

    def signal_to_spectrogram(self, signal, sample_rate=None):
        #return torchaudio.transforms.Spectrogram()(torch.tensor(signal).float())
        return torchaudio.transforms.Spectrogram()(signal.float())



class FeaturesTargetsFromArrayDS(torch.utils.data.Dataset):
    def __init__(self, features, targets, use_augmentations=False):
        self.features = features
        self.targets = targets
        self.use_augmentations = use_augmentations

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        target = self.targets[index]

        x[0:2] = torch.roll(x[0:2], shifts=[random.randint(-10, 10), random.randint(-10, 10)], dims=[1, 2])

        return x, target



def get_mean_std(mouse_audio_ds):
    mouse_audio_dl = torch.utils.data.DataLoader(mouse_audio_ds, batch_size=32, num_workers=8, shuffle=False)
    return get_mean_std_dl(mouse_audio_dl)

    mel_spectograms = []
    scaled_mel_spectograms = []

    #rel_durations = []

    for i in tqdm(range(len(mouse_audio_ds))):
        spectrogram, target = mouse_audio_ds.__getitem__(i)
        #print(i)
        mel_spectogram = spectrogram[0]
        scaled_mel_spectogram = spectrogram[1]
        #rel_duration = torch.mean(spectrogram[2])

        mel_spectograms.append(mel_spectogram)
        scaled_mel_spectograms.append(scaled_mel_spectogram)
        #rel_durations.append(torch.tensor(rel_duration))
        i += 1

    mel_spectograms = torch.stack(mel_spectograms, dim=0)
    scaled_mel_spectograms = torch.stack(scaled_mel_spectograms, dim=0)
    #rel_durations = torch.stack(rel_durations, dim=0)

    print("mean mel spectogram: {}".format(torch.mean(mel_spectograms)))
    print("std mel spectogram: {}".format(torch.std(mel_spectograms)))

    print("mean scaled_mel_spectograms: {}".format(torch.mean(scaled_mel_spectograms)))
    print("std scaled_mel_spectograms: {}".format(torch.std(scaled_mel_spectograms)))

    #print("mean rel_durations: {}".format(torch.mean(rel_durations)))
    #print("std rel_durations: {}".format(torch.std(rel_durations)))
    #raise RuntimeError

    return torch.mean(mel_spectograms), torch.std(mel_spectograms), torch.mean(scaled_mel_spectograms), torch.std(scaled_mel_spectograms)





def get_mean_std_dl(mouse_audio_dl, verbose=False):
    mel_spectograms = []
    scaled_mel_spectograms = []

    #rel_durations = []

    for spectrograms, targets in tqdm(mouse_audio_dl, disable=(not verbose)):
        for i, spectrogram in enumerate(spectrograms):
            target = targets[i]
            mel_spectogram = spectrogram[0]
            scaled_mel_spectogram = spectrogram[1]
            #rel_duration = torch.mean(spectrogram[2])

            mel_spectograms.append(mel_spectogram)
            scaled_mel_spectograms.append(scaled_mel_spectogram)


    mel_spectograms = torch.stack(mel_spectograms, dim=0)
    scaled_mel_spectograms = torch.stack(scaled_mel_spectograms, dim=0)
    #rel_durations = torch.stack(rel_durations, dim=0)

    if verbose:
        print("mean mel spectogram: {}".format(torch.mean(mel_spectograms)))
        print("std mel spectogram: {}".format(torch.std(mel_spectograms)))

        print("mean scaled_mel_spectograms: {}".format(torch.mean(scaled_mel_spectograms)))
        print("std scaled_mel_spectograms: {}".format(torch.std(scaled_mel_spectograms)))

    return torch.mean(mel_spectograms.to(DEVICE)).cpu(), torch.std(mel_spectograms.to(DEVICE)).cpu(), torch.mean(scaled_mel_spectograms.to(DEVICE)).cpu(), torch.std(scaled_mel_spectograms.to(DEVICE)).cpu()
    #return torch.mean(mel_spectograms.cuda()).cpu(), torch.std(mel_spectograms.cuda()).cpu(), torch.mean(scaled_mel_spectograms.cuda()).cpu(), torch.std(scaled_mel_spectograms.cuda()).cpu()


def get_mean_std_no_padding(mouse_audio_ds):
    mel_spectograms = []
    scaled_mel_spectograms = []

    rel_durations = []

    for i in range(len(mouse_audio_ds)):
        spectrogram, target = mouse_audio_ds.__getitem__(i)
        #print(i)
        mel_spectogram = spectrogram[0]
        scaled_mel_spectogram = spectrogram[1]

        mel_spectograms.append(mel_spectogram.flatten())
        scaled_mel_spectograms.append(scaled_mel_spectogram.flatten())
        #rel_durations.append(torch.tensor(rel_duration))
        i += 1

    mel_spectograms = torch.cat(mel_spectograms, dim=0)
    scaled_mel_spectograms = torch.cat(scaled_mel_spectograms, dim=0)
    #rel_durations = torch.stack(rel_durations, dim=0)

    print("mean mel spectogram: {}".format(torch.mean(mel_spectograms)))
    print("std mel spectogram: {}".format(torch.std(mel_spectograms)))

    print("mean scaled_mel_spectograms: {}".format(torch.mean(scaled_mel_spectograms)))
    print("std scaled_mel_spectograms: {}".format(torch.std(scaled_mel_spectograms)))

    #print("mean rel_durations: {}".format(torch.mean(rel_durations)))
    #print("std rel_durations: {}".format(torch.std(rel_durations)))

    return torch.mean(mel_spectograms), torch.std(mel_spectograms), torch.mean(scaled_mel_spectograms), torch.std(scaled_mel_spectograms)



"""
data = mouse_data_module(
    train_val_test_split=[0.8, 0.1, 0.1],
    batch_size=32,
    num_workers=8,
    #augmentations=None,
    #pct_left=1,
    #pad_same_length=False,
    #pad_factor=None,
    #max_pad=None,
    spectrogram_specs=spectrogram_specs,
    pad_start_ms=60, #60
    pad_end_ms=60,
)
data.setup("test")

print("len dataset: {}".format(len(data.dataset.data)))
dataset = MouseAudioDataset_RegularSpectrogram(data.dataset.data, use_augmentations=False, do_normalization=False, pad_to_same_size=True,
                                               final_crop_size_no_aug=170) #MouseAudioDataset_RegularSpectrogram
train_ds, _, _ = train_val_test_split(dataset)
print("len dataset: {}".format(len(train_ds)))
get_mean_std(train_ds)
"""


