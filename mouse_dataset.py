
import hashlib
from typing import List
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


from utils import parse_metadata, get_file_list


def hash_256(obj):
    return hashlib.sha256(str(obj).encode()).hexdigest()

default_category_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

default_flat_feature_shape = (32, 64) #(freq, time)

USE_CATEGORIES = False

# To combine categories 1 and 5, set:
# categories = [1, 2, 3, 4]
# category_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 1}
# The original category is still available in item['real_category']

# Sxx.shape = (freq, time)
class mouse_dataset(Dataset):
    def __init__(self, wav_files:List[str]=None, csv_files:List[str]=None, name='mouse_dataset', skip_loading=False, skip_pickle=False, pad_start_ms=10, pad_end_ms=10, categories:List[int]=[1, 2, 3, 4, 5], ignore_categories:List[int]=[], category_map:dict=default_category_map, flat_feature_shape=default_flat_feature_shape,
                 verbose=False):
        super().__init__()

        self.name = name

        self.pad_start_ms = pad_start_ms
        self.pad_end_ms = pad_end_ms
        self.categories = categories
        self.ignore_categories = ignore_categories
        self.category_map = category_map
        self.flat_feature_shape = flat_feature_shape
        self.num_classes = len(categories)

        # self._wav_data = [None]*len(wav_files)
        self.data = []
        self.categories_sorted = {str(c): [] for c in self.categories}
        self.indexes_sorted = []

        self.verbose = verbose

        if skip_loading:
            if verbose:
                print(f'({self.name}) Skipped loading.')
        else:
            self.load_data(wav_files, csv_files, skip_pickle)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


    def __str__(self):
        return f"mouse_dataset '{self.name}', length: {len(self)}."


    def get_config(self):
        return {'pad_start_ms': self.pad_start_ms,
                'pad_end_ms': self.pad_end_ms,
                'categories': self.categories,
                'ignore_categories': self.ignore_categories,
                'category_map': self.category_map,
                'flat_feature_shape': self.flat_feature_shape
                }


    def config_hash(self):
        return hash_256(self.get_config())


    def data_hash(self):
        return hash_256(self.data)


    def getsorted(self, idx):
        return self[self.indexes_sorted[idx]]


    def categories_counts(self):
        return [len(l) for l in self.categories_sorted.values()]


    def generate_category_sort(self):
        self.categories_sorted = {str(c): [] for c in self.categories}
        self.indexes_sorted = []

        for n, item in enumerate(self.data):
            self.categories_sorted[str(item['category'])].append(n)

        for c_indxs in self.categories_sorted.values():
            self.indexes_sorted.extend(c_indxs)


    def equalize(self):
        max_cat_size = max(self.categories_counts())

        for c in self.categories:
            indx_list = self.categories_sorted[str(c)]
            if len(indx_list) < max_cat_size:
                q, r = divmod(max_cat_size, len(indx_list))
                append_indx = max((q-1), 0) * indx_list + indx_list[:r]

                for i in append_indx:
                    self.data.append(self.data[i])
        self.generate_category_sort()


    def eq_factors(self):
        counts = self.categories_counts()
        factors = (max(counts) / np.array(counts))
        return factors / sum(factors)


    def load_data(self, wav_files:list, csv_files:list, skip_pickle=True):
        """
        if wav_files is None or csv_files is None or len(wav_files) == 0 or len(csv_files) == 0:
            print(f'({self.name}) Nothing to load.')
            return

        if len(wav_files) != len(csv_files):
            raise ValueError(f'({self.name}) Lengths of wav_files and csv_files must be the same. Got {len(wav_files)} and {len(csv_files)}.')

        filename = ''
        if not skip_pickle:
            #Generate hash for files to load and config. Only if this exists, load data from pickle.
            files_hash = hash_256({'wav_files': wav_files, 'csv_files': csv_files, 'conf': self.get_config()})
            filename = os.path.join("./", f'data_{files_hash}.pickle')
            if os.path.exists(filename):
                print(f"Loading data from {filename}")
                with open(filename, 'rb') as f:
                    self.data, self.categories_sorted, self.indexes_sorted = pickle.load(f)
                return

        """
        self.data = []
        if self.verbose:
            print(f'Loading {len(wav_files)} files...')

        for i, (wav_file, csv_file) in enumerate(zip(wav_files, csv_files)):
            sampling_rate, signal = wavfile.read(wav_file)
            if self.verbose:
                print(f'[{i+1:02}/{len(wav_files):02}] WAV: {wav_file} | {len(signal) / sampling_rate}s, sampling rate: {sampling_rate}')
                print(f"   |    CSV: {csv_file}")

            if not wav_file.startswith(csv_file[:csv_file.upper().find('.CSV')]) and not wav_file.startswith(csv_file[:csv_file.upper().find('_DETECTIONS.CSV')]):
                if self.verbose:
                    print(f"   └ ❌ WAV and CSV files doesn't match.")
                continue

            # Load csv file
            df = pd.read_csv(csv_file, delimiter=';', decimal=",", thousands='.')

            # Initialize lists to store the start times, end times, and categories
            if 'starttime' in df.columns:
                start_times = df['starttime'].tolist()
                #print(df['starttime'])
                #raise RuntimeError
            else:
                if self.verbose:
                    print(f"   └ ❌ File has no \"starttime\" column.")
                continue

            if 'endtime' in df.columns:
                end_times = df['endtime'].tolist()
            else:
                if self.verbose:
                    print(f"   └ ❌ File has no \"endtime\" column.")
                continue

            if 'category' in df.columns:
                categories = df['category'].tolist()
            else:
                categories = [1 for i in range(len(start_times))]
            #if self.verbose:
            #    print(f"   └ ❌ File has no \"category\" column.")
            #continue

            if '#' in df.columns:
                numbers = df['#'].tolist()
            else:
                if self.verbose:
                    print(f"   | ⚠️ File has no number column \"#\". Fine, I'll do it myself!")
                numbers = list(range(1, len(categories)+1))


            # If any of the lists are empty, skip this file
            if len(categories) == 0 or len(start_times) == 0 or len(end_times) == 0:
                if self.verbose:
                    print(f'   └ ❌ File is empty.')
                continue

            # Convert units of start time and end time
            # This calculates a factor f = 10^n with n in Z as big as possible under the constraint, that f * endtime <= length of audio data.
            div = (len(signal) / sampling_rate) / max(end_times)
            div_order = int(np.log10(div))
            if div < 1:
                div_order -= 1
            factor = 1#10 ** div_order

            if self.verbose:
                print(f"   | Assuming a time factor of {factor}. Max end time: {max(end_times)} -> {max(end_times) * factor}.")

            start_times = [s * factor for s in start_times]
            end_times = [e * factor for e in end_times]

            if not (len(categories) == len(start_times) == len(end_times)):
                if self.verbose:
                    print(f"   └ ❌ Got {len(categories)} categories, {len(start_times)} start times, {len(end_times)} end times. Not the same number.")
                continue

            count = 0
            cat_value_errors = 0
            not_cat_errors = 0
            ignored_cat_errors = 0
            time_value_errors = 0
            end_time_errors = 0
            too_short_errors = 0

            for k, (nr, start_time, end_time, real_category) in enumerate(zip(numbers, start_times, end_times, categories)):
                # start_time and end_time are now in seconds
                #print("start_time: {}, end_time: {}".format(start_time, end_time))
                try:
                    real_category = int(float(real_category))
                    category = self.category_map[str(real_category)]
                except ValueError:
                    if self.verbose:
                        print(f"   | ❗ Error (line {k+2}): Could not convert category \"{real_category}\".")
                    cat_value_errors += 1
                    continue
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                except ValueError:
                    time_value_errors += 1
                    continue
                if end_time > len(signal) / sampling_rate:
                    if self.verbose:
                        print(f"   | ❗ Error (line {k+2}): end time {end_time} is greater than length of signal {len(signal) / sampling_rate}.")
                    end_time_errors += 1
                    continue

                if end_time - start_time < 0.001:
                    if self.verbose:
                        print(f"   | ❗ Error (line {k+2}): Signal duration {(end_time - start_time)*1000}ms is less than 1ms.")
                    too_short_errors += 1
                    continue

                if category in self.categories:
                    count += 1
                    signal_slice = signal[round(start_time*sampling_rate):round(end_time*sampling_rate)]

                    start_pad = round((start_time - self.pad_start_ms/1000)*sampling_rate)
                    end_pad = round((end_time + self.pad_end_ms/1000)*sampling_rate)
                    padded_signal = signal[start_pad:end_pad]
                    #raise RuntimeError
                    if len(padded_signal) > 0:
                        f, t, Sxx = self.calc_spectrogram(padded_signal, sampling_rate)
                        Sxx_clean = None#temporal_cleanup(Sxx, sigma=1, iterations=1)
                        Sxx_clean2 = None#temporal_cleanup(Sxx_clean, sigma=1, iterations=1)

                        self.data.append({
                            'wav_file_index': i,
                            'number': nr,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration_ms': (end_time - start_time)*1000,
                            'sampling_rate': sampling_rate,
                            'signal': signal_slice,
                            'padded_signal': padded_signal,
                            'f': f,
                            't': t,
                            'Sxx': Sxx,
                            'Sxx_clean': Sxx_clean,
                            'Sxx_clean2': Sxx_clean2,
                            'features': None,
                            'category': category,
                            'one_hot_category': self.cat_to_one_hot(category),
                            'real_category': real_category,
                            'pad_start_ms': self.pad_start_ms,
                            'pad_end_ms': self.pad_end_ms,
                            'metadata': parse_metadata(csv_file),
                            'csv_file': csv_file,
                            'wav_file': wav_file
                        })
                elif category in self.ignore_categories:
                    ignored_cat_errors += 1
                else:
                    if self.verbose:
                        print(f"   | ❗ Error (line {k+2}): Category {category} is invalid.")
                    not_cat_errors += 1

            if sum((cat_value_errors, ignored_cat_errors, time_value_errors, not_cat_errors, too_short_errors)) == 0:
                if self.verbose:
                    print(f'   └ Added {count} segments, no errors 🎉')
            else:
                if self.verbose:
                    msg = f'   └ Added {count} segments of {len(categories)}'
                    msg += f', {ignored_cat_errors} ignored categories' if ignored_cat_errors > 0 else ''
                    msg += f', {not_cat_errors} invalid categories' if not_cat_errors > 0 else ''
                    msg += f', {cat_value_errors} category value errors' if cat_value_errors > 0 else ''
                    msg += f', {time_value_errors} time value errors' if time_value_errors > 0 else ''
                    msg += f', {end_time_errors} end time errors' if end_time_errors > 0 else ''
                    msg += f', {too_short_errors} too short' if too_short_errors > 0 else ''
                    msg += '.'
                    print(msg)
        self.generate_feature_vectors()
        self.generate_category_sort()

        """
        if not skip_pickle:
            print(f'Saving data to {filename}.')
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.categories_sorted, self.indexes_sorted), f)
        ### End of loading data.
        """

    def set_data(self, new_data, force_feature_generation=False):
        self.data = new_data
        self.generate_feature_vectors(force=force_feature_generation)
        self.generate_category_sort()


    def extend_data(self, new_data, force_feature_generation=False):
        self.data.extend(new_data)
        self.generate_feature_vectors(force=force_feature_generation)
        self.generate_category_sort()


    def add_results(self, model, collate_fn, force=False):
        if 'confidence' not in self[0].keys() or force:
            if self.verbose:
                print('Adding results...')
            features, labels = collate_fn(self.data)
            results = model(features).detach()

            for item, R in zip(self, results):
                item['prediction'] = self.one_hot_to_cat(R)
                item['confidence'] = max(R)
        else:
            if self.verbose:
                print('Results already added.')


    def add_img_results(self, model, collate_fn, key='result', force=False):
        if key not in self[0].keys() or force:
            if self.verbose:
                print('Adding results...')
            batch = collate_fn(self.data)
            results = model(batch).detach()

            for item, R in zip(self, results):
                item[key] = R.view(R.shape[1], R.shape[2])
        else:
            if self.verbose:
                print('Results already added.')


    def add_feature(self, key, f, from_key='Sxx'):
        if key in self[0].keys():
            if self.verbose:
                print(f"Warning! This will overwrite {key}.")
        for d in self.data:
            x = d[from_key]
            d[key] = f(x)



    def cat_to_one_hot(self, category):
        index = self.categories.index(category)
        return F.one_hot(torch.tensor(index), len(self.categories)).float()


    def one_hot_to_cat(self, one_hot):
        index = torch.argmax(one_hot).item()
        return self.categories[index]


    def generate_feature_vectors(self, from_key='Sxx', force=False):
        for d in tqdm(self.data, 'Generating feature vectors', disable=(not self.verbose)):
            if d['features'] is None or force:
                d['features'] = 0.0
                """
                #Sxx = 0.0
                Sxx = resize(d[from_key], self.flat_feature_shape)

                rel_duration = (d['end_time'] - d['start_time']) / 0.15 #maximum length = 150 ms

                d['features'] = torch.cat((torch.tensor(Sxx, dtype=torch.float32).contiguous().view(-1), torch.tensor([rel_duration], dtype=torch.float32)))
                """

    def random_split(self, lengths:List[float]=[0.1]*10, seed=42):
        gen = torch.Generator().manual_seed(seed)
        splits = random_split(range(len(self)), lengths, generator=gen)
        datasets:List[mouse_dataset] = []

        if self.verbose:
            print(f'Splitting {self.name} into {len(splits)} new datasets...')
        for n, split in enumerate(splits):
            name = f'{self.name}_s{n:02}'
            if self.verbose:
                print(f"  New dataset '{name}' with length: {len(split)}.")
            datasets.append(mouse_dataset.part_of_dataset(self, [i for i in split], name))
        return datasets


    def include(self, datasets):
        if not isinstance(datasets, list):
            datasets:List[mouse_dataset] = [datasets]

        own_hash = self.config_hash()

        for d in datasets:
            if own_hash != d.config_hash():
                if self.verbose:
                    print(f"({self.name}) Can't include '{d.name}', because it's config is different. Mine: {self.get_config()}, theirs: {d.get_config()}.")
            else:
                if self.verbose:
                    print(f"({self.name}) Including data from '{d.name}'.")
                self.extend_data(d.data)


    def split_90_10(self, mutation=0):
        if 10 < mutation or mutation < 0:
            raise ValueError(f'Variant needs to be in range of [0, 9], got {mutation}.')
        splits = self.random_split(seed=42)
        splits[mutation].name = f'{self.name}_p10_m{mutation}'
        return (mouse_dataset.combine(splits[:mutation] + splits[mutation+1:], f'{self.name}_p90_v{mutation}'), splits[mutation])


    @classmethod
    def part_of_dataset(cls, dataset, index, name=None):
        #Load part of dataset into new dataset.
        ds = cls.__new__(cls)
        name = name or dataset.name + '_part'
        ds.__init__(name=name, skip_loading=True, skip_pickle=True, **dataset.get_config())

        if not isinstance(index, list):
            index = [index]

        ds.set_data([dataset.data[i] for i in index])
        return ds


    @classmethod
    def combine(cls, datasets:list, name=None, verbose=False):
        if len(set([d.config_hash() for d in datasets])) > 1:
            raise ValueError('Cannot combine datasets with mixed configs!')
        else:
            name = name or 'combined_dataset'
            ds = cls.__new__(cls)
            ds.__init__(name=name, skip_loading=True, skip_pickle=True, **datasets[0].get_config())
            for d in datasets:
                ds.extend_data(d.data)
            if verbose:
                print(f'Combined {len(datasets)} datasets into one.')

        return ds


    @classmethod
    def from_folder(cls, folder_path:str, wav_ext='.WAV', csv_ext='.csv', **kwargs):
        wav_files = get_file_list(folder_path, wav_ext)
        csv_files = get_file_list(folder_path, csv_ext)
        return cls(wav_files, csv_files, **kwargs)


    @classmethod
    def from_wav_csv_files(cls, wav_files, csv_files, **kwargs):
        return cls(wav_files, csv_files, **kwargs)



    @staticmethod
    def calc_spectrogram(signal, sampling_rate):
        f, t, Sxx = spectrogram(signal, sampling_rate, nperseg=256, noverlap=0, nfft=256, mode='magnitude', scaling='spectrum')
        # Cuts off lower 1% of signal (considered noise), returns Sxx in logarithmic (dB), scaled to [0, 1]
        s = 100
        Sxx = None #np.log10(np.clip(Sxx * s / np.max(Sxx), 1, None)) / np.log10(s)
        return f, t, Sxx













class mouse_data_module(pl.LightningDataModule):
    def __init__(self, train_set, val_set, test_set, train_collate, val_collate=None, test_collate=None, batch_size:int=128, num_workers:int=8, pin_memory=True, spectrogram_specs=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.train_collate = train_collate
        self.val_collate = val_collate or train_collate
        self.test_collate = test_collate or self.val_collate

        self.stage = 'fit'


    def setup(self, stage='fit'):
        self.stage = stage


    def train_dataloader(self):
        if self.stage == 'fit':
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=(self.num_workers > 0), collate_fn=self.train_collate)
        else:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=(self.num_workers > 0), collate_fn=self.val_collate)


    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=(self.num_workers > 0), collate_fn=self.val_collate)


    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=False, collate_fn=self.val_collate)
