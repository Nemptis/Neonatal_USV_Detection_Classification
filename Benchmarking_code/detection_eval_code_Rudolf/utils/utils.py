import numpy as np
import csv
import os
import torch
from scipy.ndimage import gaussian_filter
import random

from torch.utils.data import random_split

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.style as mplstyle
from matplotlib.axes._axes import Axes
import matplotlib
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

from config import DEVICE

import seaborn as sns
sns.set_theme()

#plt.rcParams["font.family"] = "Times New Roman"


mplstyle.use('fast')

def get_file_list(folder:str, ext:str='.csv'):
    return sorted([os.path.join(folder, f) for f in os.listdir(os.path.join(folder)) if f.endswith(ext)])


def make_filepath(path: str, extension='.csv'):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    filename = os.path.basename(path)
    if not filename.endswith(extension):
        filename += extension
    return os.path.join(dirname, filename)


def save_csv(file_path, data, header=['starttime', 'endtime']):
    with open(file_path, 'w') as file:
        wr = csv.writer(file, delimiter=';', lineterminator='\n', quoting=0)
        wr.writerow(header)
        for call in data:
            wr.writerow(call)


def plot_big_spectogram(fig: Figure, t, f, Sxx, detection):
    ax = fig.subplots(10, 1)
    for i in range(10):
        idx_t = ((t > i*30) & (t < (i+1)*30))
        t_sel = t[idx_t]
        Sxx_selected = Sxx[:, idx_t]
        detection_selected = detection[idx_t]
        ax[i].imshow(np.log(Sxx_selected), aspect='auto', origin='lower', extent=[
                     t_sel[0], t_sel[-1], 0, f[-1]], interpolation='nearest', vmin=-2., vmax=2., cmap='Greys')
        ax[i].set_ylabel('Frequency [Hz]')
        ax[i].set_xlabel('Time [sec]')
        ax[i].set_xlim(ax[i].get_xlim())  # fixing x-axis scaling
        ax[i].plot(t_sel, 50000*detection_selected, color='orange')



def plot_item_to(ax: Axes, item:dict, pre_title='', post_title='', cmap='RdPu', colorcode='category', add_meta=False, add_source_file=False, print_title=True, key='Sxx'):
    Sxx = item[key]
    t = item['t']
    f = item['f']
    nr = item['number']
    category = item['category']
    duration_s = item['end_time'] - item['start_time']

    c = '#000000'
    if colorcode == 'category':
        c = cat_color(category)

    extent = [t[0]*1000, t[-1]*1000, f[0], f[-1]/1000]

    ax.imshow(Sxx, aspect='auto', origin='lower', interpolation='nearest', extent=extent, cmap=cmap)

    title = ''
    if add_source_file:
        title += os.path.basename(item['csv_file']) + '\n'

    title += f"#{nr} | {(duration_s)*1000:.0f} ms | {cat_name(category)}"

    if 'prediction' in item.keys():
        title += f"\nPr: {cat_name(item['prediction'], short=False)} | Cf: {item['confidence']:.2%}"
        if item['prediction'] == item['category']:
            border_color = 'green'
        else:
            border_color = 'red'
        plt.setp(ax.spines.values(), color=border_color, linewidth=3)

    title = pre_title + title + post_title

    if add_meta:
        title += "\n" + ' - '.join(item['metadata'].values())

    if print_title:
        ax.set_title(title, fontdict = {'color':c})
    ax.set_xlabel('ms', loc='right')
    ax.set_ylabel('kHz', loc='top')

    if item['pad_start_ms'] != 0:
        ax.axvline(item['pad_start_ms'])

    if item['pad_end_ms'] != 0:
        ax.axvline(item['pad_start_ms'] + duration_s * 1000)


def plot_pie_to(ax: Axes, counts, labels, colors, title='Counts'):
    ax.set_title(title, fontdict={'fontweight': 'bold'})
    ax.pie(counts, labels=labels, wedgeprops=dict(width=0.6),
           colors=colors, autopct=make_autopct(counts), pctdistance=0.7)
    middleText = f"Total:\n{sum(counts)}"
    ax.text(0., 0., middleText, horizontalalignment='center',
            verticalalignment='center', fontdict={'fontweight': 'bold'})


def to_feature_vecor(Sxx, relative_duration):
    """Convertes the downsampled spectrogram and relative (to 150 ms) duration to the combined feature vector, the model gets.

    Args:
        Sxx (numpy.ndarray): The downsampled spectrogram.
        relative_duration (float): The relative duration of the signal.

    Returns:
        torch.tensor: The combined feature vector.
    """
    return torch.cat((torch.tensor(Sxx, dtype=torch.float32).view(-1), torch.tensor([relative_duration], dtype=torch.float32)))


def to_spect_and_duration(feature_vector, spect_res=(25, 8), use_torch=False):
    """Convertes the combined feature vector back to the downsampled spectrogram and relative duration of the signal.

    Args:
        feature_vector (torch.tensor): The combines feature vector.
        spect_res (tuple, optional): The shape of the downsampled spectrogram. Defaults to (25, 8).
        use_torch (bool, optional): If this should return torch tensors. Defaults to False.

    Returns:
        tuple(torch.tensor|numpy.ndarray, ): The downsampled spectrogram and relative duration of the signal.
    """
    if use_torch:
        return feature_vector[:-1].view(*spect_res), feature_vector[-1:]
    else:
        return feature_vector[:-1].view(*spect_res).numpy(), feature_vector[-1:].numpy()


def temporal_cleanup(x, filter=None, sigma=1, delta=1, axes=0, direction='fb', iterations=1):
    if iterations > 1:
        x = temporal_cleanup(x, filter=filter, sigma=sigma, axes=axes, direction=direction, iterations=iterations-1)

    if filter is None:
        filter = gaussian_filter(x, sigma=sigma)#, axes=axes)

    if direction == 'f':
        return x[:,:-delta] * filter[:,delta:]
    elif direction == 'b':
        return x[:,delta:] * filter[:,:-delta]
    elif direction == 'fb':
        return ((x[:,:-delta] * filter[:,delta:]) + (x[:,delta:] * filter[:,:-delta])) / 2
    else:
        raise ValueError(f"Direction must be 'f' (forward) or 'b' (backward). Got {direction}.")



def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.0f}%\n({v:d})'.format(p=pct, v=val)
    return my_autopct


def cat_name(category, short=False):
    if hasattr(category, '__iter__'):
        return [cat_name(c) for c in category]
    if int(category) == 1:
        return '1 Constant' if not short else '1 Cns'
    elif int(category) == 2:
        return '2 Modulated' if not short else '2 Mdl'
    elif int(category) == 3:
        return '3 Stepped' if not short else '3 Stp'
    elif int(category) == 4:
        return '4 Composite' if not short else '4 Cmp'
    elif int(category) == 5:
        return '5 Short' if not short else '5 Shr'
    else:
        return f'{int(category)} undefined' if not short else f'{int(category)} ndf'


def cat_color(category):
    if hasattr(category, '__iter__'):
        return [cat_color(c) for c in category]
    if int(category) == 1:
        return '#1E90FF' #blue
    elif int(category) == 2:
        return '#0EAD69' #green
    elif int(category) == 3:
        return '#FF7518' #orange
    elif int(category) == 4:
        return '#8806CE' #violet
    elif int(category) == 5:
        return '#F2003C' #red
    else:
        return '#000000'


def parse_metadata(filename: str):
    parts = os.path.basename(filename).split('.')[0].split('_')

    return {'line': parts[0],
            'mama': parts[1],
            'age': parts[2],
            'animal': parts[3],
            'genotype': parts[4]}


def cross_validation_split_mouse_visu(dataset):
    train_datasets = []
    val_datasets = []
    for i in range(10):
        train_set, test_set = dataset.split_90_10(mutation=i)

        train_datasets.append(train_set)
        val_datasets.append(test_set)

    return train_datasets, val_datasets



def cross_validation_split(dataset, num_splits=10, seed=42):
    gen = torch.Generator().manual_seed(seed)
    splits = random_split(dataset, [1/num_splits for i in range(num_splits)], generator=gen)

    train_datasets = []
    val_datasets = []

    for i, val_ds in enumerate(splits):
        train_splits = []
        for j in range(len(splits)):
            if not j == i:
                train_splits.append(splits[j])

        train_ds = torch.utils.data.ConcatDataset(train_splits)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    return train_datasets, val_datasets



def assert_non_overlapping_val_indices(val_splits):
    val_split_sets = []
    for val_split in val_splits:
        val_split_sets.append(set(val_split))

    for i, val_split in enumerate(val_split_sets):
        for j in range(len(val_split_sets)):
            if j != i:
                if len(val_split.intersection(val_split_sets[j])) > 0:
                    raise RuntimeError("validation splits are overlapping!")



def get_positives_negatives(model, dataloader, class_idx, softmax_cutoff_confidence=None):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    num_class_samples = 0

    for b_data, b_target in dataloader:
        with torch.no_grad():
            pred = model(b_data.to(DEVICE))
            for i, target in enumerate(b_target):
                if softmax_cutoff_confidence is not None:
                    if pred[i] < softmax_cutoff_confidence:
                        continue
                if torch.argmax(target) == class_idx:
                    num_class_samples += 1
                    if torch.argmax(pred[i]) == class_idx:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if torch.argmax(pred[i]) == class_idx:
                        false_positives += 1
                    else:
                        true_negatives += 1

    return true_positives, true_negatives, false_positives, false_negatives, num_class_samples


def compute_scores(model, dataloader, class_idx, softmax_cutoff_confidence=None):
    true_positives, true_negatives, false_positives, false_negatives, num_class_samples = get_positives_negatives(model,
                                                                                                                  dataloader,
                                                                                                                  class_idx,
                                                                                                                  softmax_cutoff_confidence=softmax_cutoff_confidence)

    #print("true_positives: {}".format(true_positives))
    #print("true_negatives: {}".format(true_negatives))
    #print("false_positives: {}".format(false_positives))
    #print("false_negatives: {}".format(false_negatives))

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    if (true_positives + false_positives) == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    if (true_positives + false_negatives) == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    if (precision + recall) == 0:
        F1_score = 0
    else:
        F1_score = 2 * precision * recall / (precision + recall)

    return F1_score, precision, recall, specificity, accuracy, num_class_samples




def print_accuracy(model, dataloader, maskout_spectrogram=False):
    model = model.eval()
    num_samples = 0
    num_correct_samples = 0
    with torch.no_grad():
        for x, targets in dataloader:
            #time_feature = x[:, 2, 0, 0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            #x = x[:, :2]
            """
            x = torch.zeros_like(x)[0].unsqueeze(dim=0)
            print("zeros baseline: {}".format(model(x.to(DEVICE))))
            x[:, 2] = 0.18816110491752625 / 4
            print("1/4 time baseline: {}".format(model(x.to(DEVICE))))
            x[:, 2] = 0.18816110491752625 / 2
            print("1/2 time baseline: {}".format(model(x.to(DEVICE))))
            x[:, 2] = 0.18816110491752625 / 1.5
            print("1/(1.5) time baseline: {}".format(model(x.to(DEVICE))))
            x[:, 2] = 0.18816110491752625
            print("1/(1.5) time baseline: {}".format(model(x.to(DEVICE))))
            raise RuntimeError
            """
            x = x.to(DEVICE)

            """
            if maskout_spectrogram:
                with torch.no_grad():
                    mask = x[:, 0].to(DEVICE)
                    mask_max = torch.max(mask.view(mask.size(0), -1), dim=1)[0].unsqueeze(dim=-1).unsqueeze(dim=-1) * torch.ones_like(mask)
                    mask = torch.where(mask > 0.05*mask_max, 1.0, 0.0)
                    mask = mask.unsqueeze(dim=1)
                    mask = torch.nn.functional.max_pool2d(mask, kernel_size=7, padding=3, stride=1)[:, 0]#.cpu()

                x[:, 1] = torch.where(mask > 0.3, 0.0, x[:, 1])
                #x[:, 1] = torch.where(x[:, 0] > 0.3, 0.0, x[:, 1])

            #x[:, 2] = 0.18816110491752625   # mean time from labeled
            #x[:, 1] = 0
            #time_feature = time_feature.to(DEVICE)
            """
            targets = targets.to(DEVICE)

            preds = model(x)

            preds = torch.argmax(preds, dim=1)
            targets = torch.argmax(targets, dim=1)

            num_correct_samples += torch.sum(preds == targets)
            num_samples +=  len(preds)
        accuracy = num_correct_samples / num_samples

        #print("accuracy: {}, num_correct_samples: {}, num_total_samples: {}".format(accuracy, num_correct_samples, num_samples))
        return accuracy


def print_models_accuracy(models, train_dls, val_dls):
    accuracies_train = []
    accuracies_val = []
    accuracies_test = []
    for i, model in enumerate(models):
        train_dl = train_dls[i]
        val_dl = val_dls[i]

        accuracy = print_accuracy(model, train_dl, maskout_spectrogram=False).cpu().numpy()
        accuracies_train.append(accuracy)

        accuracy = print_accuracy(model, val_dl, maskout_spectrogram=False).cpu().numpy()
        accuracies_val.append(accuracy)

        #accuracy = print_accuracy(model, test_dl).cpu().numpy()
        #accuracies_test.append(accuracy)


    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print("mean accuracies train: {}".format(np.mean(accuracies_train)))
    #print("std accuracies train: {}".format(np.std(accuracies_train)))

    print("mean accuracies val: {}".format(np.mean(accuracies_val)))
    print("std accuracies val: {}".format(np.std(accuracies_val)))

    #print("{}, accuracies_val".format(accuracies_val))
    return np.mean(accuracies_val), np.std(accuracies_val)

    #print("mean accuracies test: {}".format(np.mean(accuracies_test)))
    #print("std accuracies test: {}".format(np.std(accuracies_test)))

    #print("mean num val samples: {}".format(np.mean(nums_val_samples)))
    #print("std num val samples: {}".format(np.std(nums_val_samples)))

    #print("mean num test samples: {}".format(np.mean(nums_test_samples)))
    #print("std num test samples: {}".format(np.std(nums_test_samples)))



def print_models_stats(models, val_loaders):

    print_models_accuracy(models, val_loaders, val_loaders)
    #raise RuntimeError

    # one vs all like in pessoa
    F1_scores = []
    precisions = []
    recalls = []
    specificities = []
    accuracies = []

    F1_scores_classes = []
    precisions_classes = []
    recalls_classes = []
    specificities_classes = []
    accuracies_classes = []

    num_class_samples = []
    for j, model in enumerate(models):
        F1_scores_model = []
        precisions_model = []
        recalls_model = []
        specificities_model = []
        num_class_samples_model = []
        accuracies_model = []

        for class_idx in range(5):
            F1_score, precision, recall, specificity, accuracy, n_class_samples = compute_scores(model, val_loaders[j], class_idx)

            F1_scores_model.append(torch.tensor(F1_score))
            precisions_model.append(torch.tensor(precision))
            recalls_model.append(torch.tensor(recall))
            specificities_model.append(torch.tensor(specificity))
            num_class_samples_model.append(torch.tensor(n_class_samples))
            accuracies_model.append(torch.tensor(accuracy))

        F1_scores_model = torch.stack(F1_scores_model, dim=0)
        precisions_model = torch.stack(precisions_model, dim=0)
        recalls_model = torch.stack(recalls_model, dim=0)
        specificities_model = torch.stack(specificities_model, dim=0)
        num_class_samples_model = torch.stack(num_class_samples_model, dim=0)
        accuracies_model = torch.stack(accuracies_model, dim=0)

        F1_scores_classes.append(F1_scores_model)
        precisions_classes.append(precisions_model)
        recalls_classes.append(recalls_model)
        specificities_classes.append(specificities_model)
        num_class_samples.append(num_class_samples_model)
        accuracies_classes.append(accuracies_model)

        F1_scores.append(torch.sum(F1_scores_model*num_class_samples_model)/torch.sum(num_class_samples_model))
        precisions.append(torch.sum(precisions_model*num_class_samples_model)/torch.sum(num_class_samples_model))
        recalls.append(torch.sum(recalls_model*num_class_samples_model)/torch.sum(num_class_samples_model))
        specificities.append(torch.sum(specificities_model*num_class_samples_model)/torch.sum(num_class_samples_model))
        accuracies.append(torch.sum(accuracies_model*num_class_samples_model)/torch.sum(num_class_samples_model))
        #break

    F1_scores = torch.stack(F1_scores, dim=0)
    precisions = torch.stack(precisions, dim=0)
    recalls = torch.stack(recalls, dim=0)
    specificities = torch.stack(specificities, dim=0)
    #num_class_samples = torch.stack(num_class_samples, dim=0)

    #weighted_accuracy = torch.sum((accuracies*num_class_samples)/(201*len(models)))
    #weighted_precision = torch.sum((precisions*num_class_samples)/(201*len(models)))
    #weighted_recalls = torch.sum((recalls*num_class_samples)/(201*len(models)))
    #weighted_specificity = torch.sum((specificities*num_class_samples)/(201*len(models)))

    F1_scores_classes = torch.stack(F1_scores_classes, dim=0)
    precisions_classes = torch.stack(precisions_classes, dim=0)
    recalls_classes = torch.stack(recalls_classes, dim=0)
    specificities_classes = torch.stack(specificities_classes, dim=0)
    num_class_samples = torch.stack(num_class_samples, dim=0)
    accuracies_classes = torch.stack(accuracies_classes, dim=0)

    #print("accuracies classes shape: {}".format(accuracies_classes.shape))
    #print("accuracies classes: {}".format(accuracies_classes))


    for class_idx in range(5):
        #average = np.average(precisions_classes[:, class_idx].numpy(), weights=num_class_samples[:, class_idx].numpy())
        #std_dev = np.sqrt(np.average((precisions_classes[:, class_idx].numpy()-average)**2, weights=num_class_samples[:, class_idx].numpy()))
        print("{} {}, class idx  F1_score".format(class_idx, torch.mean(F1_scores_classes, dim=0)[class_idx]))
        print("{} {}, class idx  precision".format(class_idx, torch.mean(precisions_classes, dim=0)[class_idx]))
        #print("{} {}, class idx  precision".format(class_idx, average))
        print("{} {}, class idx  recall".format(class_idx, torch.mean(recalls_classes, dim=0)[class_idx]))
        print("{} {}, class idx  specificity".format(class_idx, torch.mean(specificities_classes, dim=0)[class_idx]))
        print("{} {}, class idx  accuracy".format(class_idx, torch.mean(accuracies_classes, dim=0)[class_idx]))

        print("{} {}, class idx  F1_score".format(class_idx, torch.std(F1_scores_classes, dim=0)[class_idx]))
        print("{} {}, class idx  std precision".format(class_idx, torch.std(precisions_classes, dim=0)[class_idx]))
        #print("{} {}, class idx  std precision".format(class_idx, std_dev))
        print("{} {}, class idx  std recall".format(class_idx, torch.std(recalls_classes, dim=0)[class_idx]))
        print("{} {}, class idx  std specificity".format(class_idx, torch.std(specificities_classes, dim=0)[class_idx]))
        print("{} {}, class idx  std accuracy".format(class_idx, torch.std(accuracies_classes, dim=0)[class_idx]))


    print("{}, F1_score".format(torch.mean(F1_scores)))
    print("{}, precision".format(torch.mean(precisions)))
    print("{}, recall".format(torch.mean(recalls)))
    #print("{}, recalls".format(recalls))
    print("{}, specificity".format(torch.mean(specificities)))

    print("{}, F1_score".format(torch.std(F1_scores, unbiased=False)))
    print("{}, std precision".format(torch.std(precisions, unbiased=False)))
    print("{}, std recall".format(torch.std(recalls, unbiased=False)))
    print("{}, std specificity".format(torch.std(specificities, unbiased=False)))


            #print("{}, {}  class, accuracy".format(class_idx+1, accuracy))
            #print("{}, {}  class, precision".format(class_idx+1, precision))
            #print("{}, {}  class, recall".format(class_idx+1, recall))
            #print("{}, {}  class, specificity".format(class_idx+1, specificity))



def get_model_confidence_sorted_predictions(model, dataloader, do_adversarial_step_for_confidence=False):
    predictions = []
    predictions_confidence = []
    targets = []

    model.eval().to(DEVICE)
    with torch.no_grad():
        for x, target in dataloader:
            pred = model(x.to(DEVICE)).cpu()
            #print(torch.max(pred, dim=1))
            if do_adversarial_step_for_confidence:
                with torch.enable_grad():
                    x = x.to(DEVICE)
                    x.requires_grad = True
                    pred_ = model(x)
                    loss = torch.mean(pred_[:, torch.argmax(pred_, dim=1)])
                    loss.backward()
                    with torch.no_grad():
                        x = x + torch.abs(x.grad) * 0.2
                        pred_confidence = model(x).cpu()
            else:
                pred_confidence = pred

            predictions_confidence.append(torch.max(pred_confidence, dim=1).values)
            predictions.append(torch.argmax(pred, dim=1))
            targets.append(torch.argmax(target, dim=1))

    predictions_confidence = torch.cat(predictions_confidence, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    sorting = predictions_confidence.argsort()
    return predictions[sorting], targets[sorting], predictions_confidence[sorting]


def get_model_accuracy_confidence_threshold(models, dataloaders, model_name="EfficientNet-B5"):
    n_bins = 20
    accuracies = [[] for i in range(n_bins)]
    n_data_samples = [[] for i in range(n_bins)]

    confidence_correct_predictions = []
    confidence_wrong_predictions = []

    for i, model in enumerate(models):
        dataloader = dataloaders[i]

        predictions, targets, predictions_confidence = get_model_confidence_sorted_predictions(model, dataloader)
        for i, pred in enumerate(predictions):
            if pred == targets[i]:
                confidence_correct_predictions.append(predictions_confidence[i])
            else:
                confidence_wrong_predictions.append(predictions_confidence[i])

        step_size = len(predictions) / float(n_bins)

        for i in range(n_bins):
            preds_adjusted = []
            targets_adjusted = []

            confidence_threshold = i/n_bins
            for cutoff_index, confidence in enumerate(predictions_confidence):
                if confidence >= confidence_threshold:
                    break

            preds_adjusted = predictions[cutoff_index:]
            targets_adjusted = targets[cutoff_index:]
            accuracy = np.mean(preds_adjusted==targets_adjusted)
            #print("remove {} lowest confidence samples, accuracy: {}".format(i*20, accuracy))
            accuracies[i].append(torch.tensor(accuracy))
            n_data_samples[i].append(torch.tensor(cutoff_index))

    accuracies_per_model = []
    n_samples_per_model = []
    for k, acc in enumerate(accuracies):
        accuracies_per_model.append(torch.stack(acc, dim=0))
        n_samples_per_model.append(torch.stack(n_data_samples[k], dim=0))
    accuracies_per_model = torch.stack(accuracies_per_model, dim=0)
    n_samples_per_model = torch.stack(n_samples_per_model, dim=0)

    print(accuracies_per_model.shape)

    mean_accuracies = []
    std_accuracies = []
    mean_data_samples = []
    std_data_samples = []

    for i in range(accuracies_per_model.shape[0]):
        mean_acc = torch.mean(accuracies_per_model[i, :])
        mean_n_samples = torch.mean(n_samples_per_model[i, :].float())
        if len(accuracies_per_model[i]) > 0:
            std_acc = torch.std(accuracies_per_model[i, :])
            std_n_samples = torch.std(n_samples_per_model[i, :].float())
        else:
            std_acc = 0
            std_n_samples = 0

        #print("remove {} lowest confidence samples, mean accuracy: {}".format(int(i*step_size), mean_acc))
        #print("remove {} lowest confidence samples, std accuracy: {}".format(int(i*step_size), std_acc))

        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)
        mean_data_samples.append(mean_n_samples)
        std_data_samples.append(std_n_samples)

    #mean_accuracies *= 100
    #std_accuracies *= 100

    #data_removal = [int(i*step_size) for i in range(n_bins)]
    #data_removal = [d/len(predictions) for d in data_removal]
    data_removal = 100-100*np.array(mean_data_samples)/len(predictions)
    std_data_removal = 100*np.array(std_data_samples)/len(predictions)
    mean_accuracies = 100*np.array(mean_accuracies)
    std_accuracies = 100*np.array(std_accuracies)

    mean_accuracies_print = [float(mean_accuracies[i]) for i in range(len(mean_accuracies))]
    print("mean accuracies: {}".format(mean_accuracies_print))
    mean_data_removal_print = [float(data_removal[i]) for i in range(len(data_removal))]
    print("mean data removal: {}".format(mean_data_removal_print))

    confidence_plot_new(mean_accuracies, std_accuracies, data_removal, std_data_removal, confidence_correct_predictions,
                    confidence_wrong_predictions, model_name, n_bins)



def confidence_plot_new(mean_accuracies, std_accuracies, data_removal, std_data_removal, confidence_correct_predictions,
                    confidence_wrong_predictions, model_name="EfficientNet-B5", n_bins=20):
    blue = '#517CE8'
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()

    bins = np.linspace(min(min(confidence_correct_predictions), min(confidence_wrong_predictions)), max(max(confidence_correct_predictions), max(confidence_wrong_predictions)), n_bins)


    ax.hist(confidence_correct_predictions, bins=bins, color='#4BB04B99', label="Correct Prediction", align='left')
    ax.hist(confidence_wrong_predictions, bins=bins, color='#F72B2B80', label="Wrong Prediction", align='left')
    ax_acc = ax.twinx()

    # plt.title(model_name)
    ax.set_xlabel('Confidence Threshold $p$')

    ax_acc.tick_params(axis='y', colors=blue)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1000, symbol='%'))
    ax_acc.yaxis.set_major_formatter(mtick.PercentFormatter(100, symbol='%'))
    ax_acc.set_ylim(0, 105)

    ax_acc.plot(np.linspace(0, 1, n_bins), mean_accuracies, linewidth=3, color=blue, label="Accuracy (conf $\geq p$)")
    ax_acc.plot(np.linspace(0, 1, n_bins), data_removal, linewidth=3, linestyle='dashed', color=blue, label="Data Preserved (conf $\geq p$)")
    ax_acc.fill_between(np.linspace(0, 1, n_bins), mean_accuracies-std_accuracies, np.clip(mean_accuracies+std_accuracies, a_min=0.0, a_max=100.0), linewidth=0, color=blue, alpha=0.25)

    ax_acc.fill_between(np.linspace(0, 1, n_bins), data_removal-std_data_removal, data_removal+std_data_removal, linewidth=0, color=blue, alpha=0.25)

    ax.legend(loc='lower left')
    ax_acc.legend(loc='center left')

    plt.savefig("save_figs/accuracy_confidence/accuracy_confidence_" + model_name + ".pdf", bbox_inches='tight')



def confidence_plot(mean_accuracies, std_accuracies, data_removal, std_data_removal, confidence_correct_predictions,
                    confidence_wrong_predictions, model_name="EfficientNet-B5", n_bins=20):
    plt.figure(dpi=300)
    ax = plt.gca()
    ax.set_facecolor((0.9, 0.95, 1.0))

    #confidence_correct_predictions = confidence_correct_predictions / len(models)
    #confidence_wrong_predictions = confidence_wrong_predictions / len(models)

    bins = np.linspace(min(min(confidence_correct_predictions), min(confidence_wrong_predictions)), max(max(confidence_correct_predictions), max(confidence_wrong_predictions)), n_bins)

    vline_x_values = [0.2*(i+1) for i in range(4)]
    for vline_x_value in vline_x_values:
        ax.axvline(vline_x_value, 0.0, 1.0, color="white", alpha=1.0, linewidth=0.75)

    ax.hist(confidence_correct_predictions, bins=bins, alpha=0.5, color=(0, 1, 0), histtype='bar', ec='white', label="Correct Prediction")
    ax.hist(confidence_wrong_predictions, bins=bins, alpha=0.5, color=(1, 0, 0), histtype='bar', ec='white', label="Wrong Prediction")
    #ax.set_ylim([0.84, 1.0])
    ax_acc = ax.twinx()
    ax_acc.set_ylim([0.0, 100.0])

    plt.title(model_name)
    ax.set_xlabel('Confidence Threshold $p$', fontsize=10)
    ax_acc.set_ylabel('Recall, Data Preserved', fontsize=10)
    ax_acc.yaxis.set_major_formatter(mtick.PercentFormatter(100, symbol='%'))
    hline_y_values = [20*(i+1) for i in range(4)] + [90]
    for hline_y_value in hline_y_values:
        ax_acc.axhline(hline_y_value, 0.0, 1.0, color="white", alpha=1.0, linewidth=0.75)


    #matplotlib.axes.Axes.set_ylim(0.85, 1.0)
    ax_acc.plot(np.linspace(0, 1, n_bins), mean_accuracies, '--bo', alpha=0.5, label="Recall Conf $\geq p$")
    ax_acc.plot(np.linspace(0, 1, n_bins), data_removal, '--ys', alpha=0.5, label="Data Preserved Conf $\geq p$")
    ax_acc.fill_between(np.linspace(0, 1, n_bins), mean_accuracies-std_accuracies, mean_accuracies+std_accuracies, alpha=0.5)
    #print("data removal shape: {}".format(data_removal.shape))
    #print("std data removal shape: {}".format(std_data_removal.shape))

    ax_acc.fill_between(np.linspace(0, 1, n_bins), data_removal-std_data_removal, data_removal+std_data_removal, color='yellow', alpha=0.5)
    #ax.legend([hist_correct, hist_wrong], ["Correct", "Wrong"])
    ax.legend(loc=[0.01, 0.21])
    ax.set_yticks([])
    ax_acc.legend(loc=[0.01, 0.41])
    plt.savefig("save_figs/accuracy_confidence/accuracy_confidence_" + model_name + ".png", bbox_inches='tight')




def get_model_accuracy_confidence_cutoff(models, dataloaders, model_name="efficentnetb5"):
    n_bins = 20
    accuracies = [[] for i in range(n_bins)]

    confidence_correct_predictions = []
    confidence_wrong_predictions = []

    for i, model in enumerate(models):
        dataloader = dataloaders[i]

        predictions, targets, predictions_confidence = get_model_confidence_sorted_predictions(model, dataloader)
        for i, pred in enumerate(predictions):
            if pred == targets[i]:
                confidence_correct_predictions.append(predictions_confidence[i])
            else:
                confidence_wrong_predictions.append(predictions_confidence[i])

        step_size = len(predictions) / float(n_bins)


        for i in range(n_bins):
            preds_adjusted = predictions[int(i*step_size):]
            targets_adjusted = targets[int(i*step_size):]
            accuracy = np.mean(preds_adjusted==targets_adjusted)
            #print("remove {} lowest confidence samples, accuracy: {}".format(i*20, accuracy))
            accuracies[i].append(torch.tensor(accuracy))

    accuracies_per_model = []
    for acc in accuracies:
        accuracies_per_model.append(torch.stack(acc, dim=0))
    accuracies_per_model = torch.stack(accuracies_per_model, dim=0)
    print(accuracies_per_model.shape)

    mean_accuracies = []
    std_accuracies = []
    for i in range(accuracies_per_model.shape[0]):
        mean_acc = torch.mean(accuracies_per_model[i, :])
        if len(accuracies_per_model[i]) > 0:
            std_acc = torch.std(accuracies_per_model[i, :])
        else:
            std_acc = 0

        print("remove {} lowest confidence samples, mean accuracy: {}".format(int(i*step_size), mean_acc))
        print("remove {} lowest confidence samples, std accuracy: {}".format(int(i*step_size), std_acc))

        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)

    mean_accuracies_print = [float(mean_accuracies[i]) for i in range(len(mean_accuracies))]
    print("mean accuracies: {}".format(mean_accuracies_print))
    print("std accuracies: {}".format(std_accuracies))

    data_removal = [int(i*step_size) for i in range(n_bins)]
    data_removal = [d/len(predictions) for d in data_removal]

    plt.figure(dpi=300)
    ax = plt.gca()
    bins = np.linspace(min(min(confidence_correct_predictions), min(confidence_wrong_predictions)), max(max(confidence_correct_predictions), max(confidence_wrong_predictions)), n_bins)

    plt.hist(confidence_correct_predictions, bins=bins, alpha=0.5, color=(0, 1, 0), histtype='bar', ec='white')
    plt.hist(confidence_wrong_predictions, bins=bins, alpha=0.5, color=(1, 0, 0), histtype='bar', ec='white')
    #ax.set_ylim([0.84, 1.0])
    ax_acc = ax.twinx()
    #ax_acc.set_ylim([0.0, 1.0])

    ax_acc.set_ylim([0.84, 1.0])
    #ax.set_ylim([0.0, 1.0])
    plt.title(model_name)
    ax.set_xlabel('Data Percentage Removed', fontsize=10)
    ax_acc.set_ylabel('Accuracy', fontsize=10)

    #matplotlib.axes.Axes.set_ylim(0.85, 1.0)
    ax_acc.plot(data_removal, mean_accuracies, '--bo')
    ax_acc.fill_between(data_removal, np.array(mean_accuracies)-np.array(std_accuracies), np.array(mean_accuracies)+np.array(std_accuracies), alpha=0.5)
    plt.savefig("save_figs/accuracy_confidence/accuracy_confidence_" + model_name + ".png")




def get_model_accuracy_confidence_cutoff_old(models, dataloaders, model_name="efficentnetb5"):
    n_bins = 21
    accuracies = [[] for i in range(n_bins)]
    for i, model in enumerate(models):
        dataloader = dataloaders[i]

        predictions, targets, predictions_confidence = get_model_confidence_sorted_predictions(model, dataloader)
        step_size = len(predictions) / float(n_bins)

        for i in range(n_bins):
            preds_adjusted = predictions[int(i*step_size):]
            targets_adjusted = targets[int(i*step_size):]
            accuracy = np.mean(preds_adjusted==targets_adjusted)
            #print("remove {} lowest confidence samples, accuracy: {}".format(i*20, accuracy))
            accuracies[i].append(torch.tensor(accuracy))

    accuracies_per_model = []
    for acc in accuracies:
        accuracies_per_model.append(torch.stack(acc, dim=0))
    accuracies_per_model = torch.stack(accuracies_per_model, dim=0)
    print(accuracies_per_model.shape)

    mean_accuracies = []
    std_accuracies = []
    for i in range(accuracies_per_model.shape[0]):
        mean_acc = torch.mean(accuracies_per_model[i, :])
        if len(accuracies_per_model[i]) > 0:
            std_acc = torch.std(accuracies_per_model[i, :])
        else:
            std_acc = 0

        print("remove {} lowest confidence samples, mean accuracy: {}".format(int(i*step_size), mean_acc))
        print("remove {} lowest confidence samples, std accuracy: {}".format(int(i*step_size), std_acc))

        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)

    print("mean accuracies: {}".format(mean_accuracies))
    print("std accuracies: {}".format(std_accuracies))

    data_removal = [int(i*step_size) for i in range(n_bins)]
    data_removal = [d/len(predictions) for d in data_removal]

    plt.figure(dpi=300)
    ax = plt.gca()
    ax.set_ylim([0.84, 1.0])
    #ax.set_ylim([0.0, 1.0])
    plt.title(model_name)
    ax.set_xlabel('Data Percentage Removed', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)

    #matplotlib.axes.Axes.set_ylim(0.85, 1.0)
    plt.plot(data_removal, mean_accuracies, '--bo')
    plt.fill_between(data_removal, np.array(mean_accuracies)-np.array(std_accuracies), np.array(mean_accuracies)+np.array(std_accuracies))
    plt.savefig("save_figs/accuracy_confidence/accuracy_confidence_" + model_name + ".png")



def get_model_confidence(model, dataset):
    confidence_values_correct = []
    confidence_values_wrong = []
    with torch.no_grad():
        for x, targets in dataset:
            x = x.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(x)
            preds_class = torch.argmax(preds, dim=1)
            targets = torch.argmax(targets, dim=1)-1

            for i, pred in enumerate(preds_class):
                target = targets[i]

                if pred == target:
                    confidence_values_correct.append(preds[i][pred])
                else:
                    confidence_values_wrong.append(preds[i][pred])

    return torch.stack(confidence_values_correct, dim=0).cpu(), torch.stack(confidence_values_wrong, dim=0).cpu()


def plot_model_confidence(model, dataset, n_bins=30):
    confidence_values_correct, confidence_values_wrong = get_model_confidence(model, dataset)

    #print(confidence_values_wrong)

    fig = plt.hist(confidence_values_correct, bins=n_bins, color=(0, 1, 0))
    #plt.imshow()
    fig = plt.hist(confidence_values_wrong, bins=n_bins, color=(1, 0, 0))
    plt.show()
    #plt.imshow()

