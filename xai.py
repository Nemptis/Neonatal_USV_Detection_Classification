
from classification_net_cnn import classification_net_cnn, classification_net_cnn_image_lightning, classification_net_cnn_image_lightning_double_channels, classification_net_cnn_image_lightning_EfficentNetB5, classification_net_cnn_image_lightning_no_regularization, classification_net_cnn_image_lightning_ResNet50, classification_net_cnn_image_lightning_ResNet34, classification_net_cnn_image_lightning_ViT_B
import captum
import torch
from mouse_dataset import mouse_data_module, mouse_dataset
spectrogram_specs = None
import time
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torchvision
import random
from torchvision.models.feature_extraction import create_feature_extractor

from train_delete import filter_classes
from train_mel_specs import train_val_test_split, print_accuracy, get_data_loader, get_dataloader_adversarial_examples
from mel_dataset import MouseAudioDataset, MouseAudioDataset_RegularSpectrogram, get_mean_std
from tqdm import tqdm

import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pathlib import Path
import os
from temperature_scaling import ModelWithTemperature

from utils.utils import cross_validation_split, compute_scores, print_models_stats, cross_validation_split_mouse_visu, get_model_accuracy_confidence_cutoff, print_models_accuracy

from config import DEVICE

#from eval_ import backward_fix_hook_model

CATEGORIES = [1, 2, 3, 4, 5]
CATEGORY_MAP = {'0': 6, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}


hook_output = None

def layer_hook(module, input_, output):
    global hook_output
    hook_output = output


def access_activations_forward_hook(x, forward_function, forward_hook_point, at_channel=None):
    handle = forward_hook_point.register_forward_hook(layer_hook)

    with torch.no_grad():
        forward_function(*x)

    handle.remove()

    if at_channel is None:
        return hook_output.detach().cpu()
    else:
        return hook_output[:, at_channel].detach().cpu()



def extract_mean_activation_vectors(model, layer_hook, dataloader):
    mean_activation_vecs = []
    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(DEVICE)
            activations = access_activations_forward_hook([x], model.forward, layer_hook)
            mean_activation_vec = torch.mean(activations, dim=[2, 3])
            mean_activation_vecs.append(mean_activation_vec)

    mean_activation_vecs = torch.cat(mean_activation_vecs, dim=0)
    return mean_activation_vecs


def get_closest_image_indices(model, test_sample, layer_hook, mean_activation_vecs, n_images_to_sample=8):
    activation_vec_test = access_activations_forward_hook([test_sample.to(DEVICE)], model.forward, layer_hook)
    mean_activation_vec_test = torch.mean(activation_vec_test, dim=[2, 3])

    cossim = -F.cosine_similarity(mean_activation_vecs, mean_activation_vec_test)
    indices = torch.argsort(cossim)
    return indices[:n_images_to_sample], test_sample.cpu()

    #print("cossim shape: {}".format(cossim.shape))


def plot_close_image_samples(model, layer_hook, summary_writer=None):

    train_ds, val_ds, test_ds = get_high_res_datasets()

    dataloader_library = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False,
                                            num_workers=8)

    mean_activation_vecs = extract_mean_activation_vectors(model, layer_hook, dataloader_library).cpu()

    dataloader_test = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True,
                                            num_workers=0)

    j = 0
    for x, target in dataloader_test:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "plot_similar_images" + ' ' + current_time + '')

        indices, test_sample = get_closest_image_indices(model, x, layer_hook, mean_activation_vecs)
        plot_images = [test_sample[0]] + [train_ds[index][0] for index in indices]

        save_path_specs = "save_figs/closests_specs/" + str(j) + "/specs/"
        save_path_specs_db = "save_figs/closests_specs/" + str(j) + "/specs_db/"

        Path(save_path_specs).mkdir(parents=True, exist_ok=True)
        Path(save_path_specs_db).mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(plot_images):
            print(image.shape)
            fig = plt.figure(figsize=(20, 10))
            plt.imshow(image[0])
            plt.colorbar()
            plt.savefig(save_path_specs + str(i) + ".png")
            summary_writer.add_figure("spectogram", fig, global_step=i)


            fig = plt.figure(figsize=(20, 10))
            plt.imshow(image[1])
            plt.colorbar()
            plt.savefig(save_path_specs_db + str(i) + ".png")
            summary_writer.add_figure("spectogram db", fig, global_step=i)



            summary_writer.close()
            plt.close()

        j += 1
        if j > 10:
            break

        #break


def overlay_image_with_activation(spectogram_single, activation_spatial):
    activation_spatial = F.relu(activation_spatial)
    activation_spatial = activation_spatial / torch.max(activation_spatial)

    activation_spatial = F.interpolate(activation_spatial.unsqueeze(dim=0).unsqueeze(dim=0), size=spectogram_single.shape)[0][0]

    spectogram_single = spectogram_single * activation_spatial

    return spectogram_single




def plot_highest_activating_channels_from_data(model, layer_hook, n_channels_to_plot=4):
    train_ds, val_ds, test_ds = get_high_res_datasets()

    dataloader_test = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True,
                                            num_workers=0)

    j = 0

    for x, target in dataloader_test:
        activations = access_activations_forward_hook([x.to(DEVICE)], model.forward, layer_hook)
        mean_activation_vec = torch.mean(activations, dim=[0, 2, 3])
        indices_highest_activating_channels = torch.argsort(-mean_activation_vec)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "plot_highest_activating_channels" + ' ' + current_time + '')

        fig = plt.figure(figsize=(20, 10))
        plt.imshow(x[0][0])
        plt.colorbar()
        #plt.savefig(save_path_specs + str(i) + ".png")
        summary_writer.add_figure("spectogram", fig, global_step=0)


        fig = plt.figure(figsize=(20, 10))
        plt.imshow(x[0][1])
        plt.colorbar()
        #plt.savefig(save_path_specs_db + str(i) + ".png")
        summary_writer.add_figure("spectogram db", fig, global_step=0)


        for i in range(n_channels_to_plot):
            index = indices_highest_activating_channels[i]
            channel_vis = optim_vis_high_res_model(model, layer_hook, index+1, False)[0].cpu().detach()


            fig = plt.figure(figsize=(20, 10))
            plt.imshow(channel_vis[1])
            plt.colorbar()
            #plt.savefig(save_path_specs + str(i) + ".png")
            summary_writer.add_figure("channel vis db", fig, global_step=i)


            fig = plt.figure(figsize=(20, 10))
            plt.imshow(activations[0][i].cpu())
            plt.colorbar()
            #plt.savefig(save_path_specs + str(i) + ".png")
            summary_writer.add_figure("channel activation", fig, global_step=i)


            overlay = overlay_image_with_activation(x[0][1], activations[0][i].cpu())
            fig = plt.figure(figsize=(20, 10))
            plt.imshow(overlay)
            plt.colorbar()
            #plt.savefig(save_path_specs + str(i) + ".png")
            summary_writer.add_figure("overlay", fig, global_step=i)



            summary_writer.close()
            #plt.close()
        j += 1

        if j > 10:
            break



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


def plot_model_confidence(model, dataset):
    confidence_values_correct, confidence_values_wrong = get_model_confidence(model, dataset)

    #print(confidence_values_wrong)

    fig = plt.hist(confidence_values_correct, bins=30, color=(0, 1, 0))
    #plt.imshow()
    fig = plt.hist(confidence_values_wrong, bins=30, color=(1, 0, 0))
    plt.show()
    #plt.imshow()


def loss_func_adv(pred, target):
    return -pred[:, target]


def load_labeled_ds():
    labeled_ds = mouse_dataset.from_folder('data/labeled', name='manu-set', categories=CATEGORIES, category_map=CATEGORY_MAP, wav_ext='_SOX.WAV', pad_start_ms=60, pad_end_ms=60)


    # extract mean std from training data for data normalization
    dataset_for_normalization = MouseAudioDataset_RegularSpectrogram(labeled_ds.data, use_augmentations=False, do_normalization=False,
                                                                        pad_to_same_size=True,
                                                                        final_crop_size_no_aug=170)

    mean_spectrogram, std_spectrogram, mean_scaled_spectrogram, std_scaled_spectrogram = get_mean_std(dataset_for_normalization)

    labeled_ds = MouseAudioDataset_RegularSpectrogram(labeled_ds.data, mean_spectogram=mean_spectrogram, std_spectogram=std_spectrogram,
                                            mean_scaled_spectogram=mean_scaled_spectrogram, std_scaled_spectogram=std_scaled_spectrogram,
                                            final_crop_size_no_aug=170, use_augmentations=False)

    return labeled_ds




def generate_adversarial_examples_dataset(target_map={0:1, 1:2, 2:4, 3:0, 4:3}):
    model_0 = classification_net_cnn_image_lightning.load_from_checkpoint("logs/full_labeled_data_10_cross_validation/lightning_logs/version_0/checkpoints/epoch=139-step=12880.ckpt").to(DEVICE).eval()


    train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=170)

    labeled_train_dataset = load_labeled_ds()

    #train_dataset = train_datasets[0]
    val_ds = val_datasets[0]
    #train_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    #datasets = [train_datasets[0], labeled_train_dataset, val_ds]
    #save_names = ["train_ds", "train_labeled_ds", "val_ds"]
    datasets = [labeled_train_dataset, val_ds]
    save_names = ["train_labeled_ds", "val_ds"]

    for i, dataset in enumerate(datasets):
        save_name = save_names[i]
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        adversarial_ds = []
        adversarial_targets = []

        #j = 0
        for x, targets in tqdm(train_loader):

            x = x.to(DEVICE)
            #target = torch.tensor(0).unsqueeze(dim=0).to(DEVICE)
            for target in targets:
                adv_target = target_map[int(torch.argmax(target))]
                adversarial_targets.append(adv_target)
                #adversarial_targets.append(torch.tensor(adv_target).long())
            #adversarial_targets = torch.stack(adversarial_targets, dim=0).to(DEVICE)

            PGD_Generator = captum.robust.PGD(model_0.forward_logits, loss_func=loss_func_adv)
            x_adv = PGD_Generator.perturb(x, radius=1.0, step_size=0.005, step_num=300,
                                target=adv_target, targeted=True).cpu().detach()

            adversarial_ds.append(x_adv)
            #j += 1
            #if j > 2:
            #    break

        adversarial_ds = torch.cat(adversarial_ds, dim=0).numpy()
        np.save("data/adversarial_generated/epsilon_1_0/" + save_name + ".npy", adversarial_ds)
        np.save("data/adversarial_generated/epsilon_1_0/" + save_name + "_targets.npy", adversarial_targets)

        #np.save("data/adversarial_generated/val_ds.npy", adversarial_ds)
        #np.save("data/adversarial_generated/val_ds_targets.npy", adversarial_targets)




def generate_adversarial_examples():
    model_0 = classification_net_cnn_image_lightning.load_from_checkpoint("logs/full_labeled_data_10_cross_validation/lightning_logs/version_0/checkpoints/epoch=139-step=12880.ckpt").to(DEVICE).eval()
    model_1 = classification_net_cnn_image_lightning.load_from_checkpoint("logs/full_labeled_data_10_cross_validation/lightning_logs/version_1/checkpoints/epoch=330-step=30452.ckpt").to(DEVICE).eval()

    train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=170)

    val_ds = val_datasets[0]

    val_loader = DataLoader(val_datasets[1], batch_size=1, shuffle=True)


    i = 0
    for x, target in val_loader:
        print(target)
        #raise RuntimeError
        x = x.to(DEVICE)
        #with torch.no_grad():
        #    pred = model_0(x)
        #    print(pred)

        plt.figure(dpi=400)
        plt.imshow(x[0, 1].cpu())
        plt.savefig("save_figs/adv/x_" + str(i) + ".png")

        target = torch.tensor(0).unsqueeze(dim=0).to(DEVICE)
        print("x.shape: {}".format(x.shape))
        print("target.shape: {}".format(target.shape))

        with torch.no_grad():
            pred = model_0.forward_logits(x)
        print("pred.shape: {}".format(pred.shape))

        PGD_Generator = captum.robust.PGD(model_0.forward_logits, loss_func=loss_func_adv)
        x_adv = PGD_Generator.perturb(x, radius=0.5, step_size=0.005, step_num=300,
                            target=target, targeted=True)

        with torch.no_grad():
            print("pred: {}".format(model_0(x)))
            print("pred adv: {}".format(model_0(x_adv)))
            print("pred 1: {}".format(model_1(x)))
            print("pred 1 adv: {}".format(model_1(x_adv)))

        plt.figure(dpi=400)
        plt.imshow(x_adv[0, 1].cpu())
        plt.savefig("save_figs/adv/x_adv_" + str(i) + ".png")


        i += 1
        if i > 10:
            break
        #break


def get_mean_std_accuracy(models, dataloader):
    accuracies = []
    for model in models:
        accuracy = print_accuracy(model, dataloader).cpu().numpy()
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)



def print_mean_std_accuracies(models, dataloaders):
    mean_accs = []
    std_accs = []
    for dataloader in dataloaders:
        mean_acc, std_acc = get_mean_std_accuracy(models, dataloader)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("mean accuracies: {}".format(mean_accs))
    print("std accuracies: {}".format(std_accs))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++")



def plot_highest_attributing_channels_from_data(model, layer_hook, n_channels_to_plot=4, wanted_target_class_number=1):
    train_ds, val_ds, test_ds = get_high_res_datasets()

    dataloader_test = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True,
                                            num_workers=0)

    j = 0

    for x, target in dataloader_test:
        x = x.to(DEVICE)
        with torch.no_grad():
            pred = model(x)
        if torch.argmax(pred) == wanted_target_class_number:
            if torch.argmax(target) == wanted_target_class_number:
                baseline = torch.zeros_like(x)
                baseline[:, -1] = x[:, -1]
                attribution = compute_attribution_hidden_layer(model, layer_hook, x, baseline, wanted_target_class_number).detach().cpu()

                #print(attribution.shape)
                #activations = access_activations_forward_hook([x.to(DEVICE)], model.forward, layer_hook)
                mean_attribution_vec = torch.mean(attribution, dim=[0, 2, 3])
                #print(mean_attribution_vec.shape)
                indices_highest_activating_channels = torch.argsort(-mean_attribution_vec)
                #print(len(indices_highest_activating_channels))

                #raise RuntimeError

                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "plot_highest_activating_channels" + ' ' + current_time + '')

                x = x.cpu()
                fig = plt.figure(figsize=(20, 10))
                plt.imshow(x[0][0])
                plt.colorbar()
                #plt.savefig(save_path_specs + str(i) + ".png")
                summary_writer.add_figure("spectogram", fig, global_step=0)

                fig = plt.figure(figsize=(20, 10))
                plt.imshow(x[0][1])
                plt.colorbar()
                #plt.savefig(save_path_specs_db + str(i) + ".png")
                summary_writer.add_figure("spectogram db", fig, global_step=0)


                for i in range(n_channels_to_plot):
                    print(len(indices_highest_activating_channels))
                    index = indices_highest_activating_channels[i]
                    channel_vis = optim_vis_high_res_model(model, layer_hook, index+1, False)[0].cpu().detach()


                    fig = plt.figure(figsize=(20, 10))
                    plt.imshow(channel_vis[1])
                    plt.colorbar()
                    #plt.savefig(save_path_specs + str(i) + ".png")
                    summary_writer.add_figure("channel vis db", fig, global_step=i)


                    fig = plt.figure(figsize=(20, 10))
                    plt.imshow(attribution[0][i].cpu())
                    plt.colorbar()
                    #plt.savefig(save_path_specs + str(i) + ".png")
                    summary_writer.add_figure("channel activation", fig, global_step=i)


                    overlay = overlay_image_with_activation(x[0][1], attribution[0][i].cpu())
                    fig = plt.figure(figsize=(20, 10))
                    plt.imshow(overlay)
                    plt.colorbar()
                    #plt.savefig(save_path_specs + str(i) + ".png")
                    summary_writer.add_figure("overlay", fig, global_step=i)



                    summary_writer.close()
                    #plt.close()
                j += 1

                if j > 10:
                    break







def compute_tsne_embedding(data):
    perplexity_values = [5, 10, 20, 30, 40, 50, 100]
    X_embeddings = []
    j = 0
    for j in range(len(perplexity_values)):
        #color = []
        perplexity = perplexity_values[j]

        tsne = TSNE(n_jobs=19, perplexity=perplexity, n_iter=1000, metric='cosine')
        X_embedded = tsne.fit_transform(data)
        print("x_embedded shape: {}".format(X_embedded.shape))

        X_embeddings.append(X_embedded)

    return X_embeddings, perplexity_values


def tsne(data, colors, save_name=""):

    perplexity_values = [5, 10, 20, 30, 40, 50, 100]
    #perplexity_values = [30]
    j = 0
    for j in range(len(perplexity_values)):
        #color = []
        perplexity = perplexity_values[j]

        tsne = TSNE(n_jobs=19, perplexity=perplexity, n_iter=1000, metric='cosine')
        X_embedded = tsne.fit_transform(data)
        print("x_embedded shape: {}".format(X_embedded.shape))

        color_new = []
        for k in range(X_embedded.shape[0]):
            color_new.append(colors[k])
        color = color_new

        x, y = zip(*X_embedded)
        fig = plt.figure(figsize=(10, 10))


        #print("len color: {}".format(len(color)))
        #color_by_attribution = [(torch.clamp(-torch.sum(X[index])/10.0, 0, 1), 0.0, torch.clamp(torch.sum(X[index])/10.0, 0, 1)) for index in range(len(X))]
        #print("len color: {}".format(len(color_by_attribution)))

        plt.scatter(x, y, c=color)#, s=4.0)#[(X_softmaxed[i][0], 0.0, 0.0) for i in range(len(X_softmaxed))])
        #plt.scatter(x, y, c = color_by_attribution)
        plt.show()
        plt.savefig("save_figs/tsne/" + save_name + "_" + str(perplexity) + ".png")

        #i += 1
        #if i > 5:
        #    break

    time.sleep(1)











def load_all_data(data_loader):
    features = []
    targets = []

    for x, target in data_loader:
        features.append(x)
        targets.append(target)

    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)

    return features, targets



def get_test_data():
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
    )
    data.setup()
    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()
    test_dl = data.test_dataloader()

    features, targets = load_all_data(test_dl)

    return features, targets


def get_val_data():
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
    )
    data.setup()
    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()
    test_dl = data.test_dataloader()

    features, targets = load_all_data(val_dl)

    return features, targets


def get_train_data():
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
    )
    data.setup()
    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()
    test_dl = data.test_dataloader()

    features, targets = load_all_data(train_dl)

    return features, targets



def plot_tsne():
    #model = classification_net.load_from_checkpoint("logs/lightning_logs/version_14/checkpoints/epoch=39-step=3520.ckpt").to(DEVICE).eval()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

    #features, targets = get_train_data()

    model = torch.load("ckpts/model.pt").to(DEVICE).eval()
    #features = torch.load("ckpts/train_features.pt").to(DEVICE)
    #targets = torch.load("ckpts/train_targets.pt").to(DEVICE)
    features = torch.load("ckpts/val_features.pt").to(DEVICE)
    targets = torch.load("ckpts/val_targets.pt").to(DEVICE)

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)

    #for name, param in model.named_parameters():
    #    print(name)
    #raise RuntimeError

    class_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    layer_hooks = [model.identity_layer_for_input_hook, model.layers.layers[1], model.layers.layers[4], model.layers.layers[7], model.layers.layers[10]]

    #layer_hooks = [model.identity_layer_for_input_hook, model.layers.layers[1], model.layers.layers[4], model.layers.layers[7], model.layers.layers[10]]
    log_names = ["input", "layer_1", "layer_4", "layer_7", "logits"]

    for i, layer_hook in enumerate(layer_hooks):

        activation_vecs = access_activations_forward_hook([features], model.forward, layer_hook)

        colors = []
        for target in targets:
            colors.append(class_colors[target.argmax()])


        tsne(activation_vecs, colors, summary_writer, log_tag="tsne " + log_names[i])




def plot_tsne_train_val():
    #model = classification_net.load_from_checkpoint("logs/lightning_logs/version_14/checkpoints/epoch=39-step=3520.ckpt").to(DEVICE).eval()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

    #features, targets = get_train_data()

    model = torch.load("ckpts/model.pt").to(DEVICE).eval()
    features = torch.load("ckpts/train_features.pt").to(DEVICE)
    targets = torch.load("ckpts/train_targets.pt").to(DEVICE)

    val_features = torch.load("ckpts/val_features.pt").to(DEVICE)
    colors = np.array([np.array([1, 0, 0]) for i in range(len(features))] + [np.array([0, 1, 0]) for i in range(len(val_features))])

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)

    #for name, param in model.named_parameters():
    #    print(name)
    #raise RuntimeError

    class_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    layer_hooks = [model.identity_layer_for_input_hook, model.layers.layers[1], model.layers.layers[4], model.layers.layers[7], model.layers.layers[10]]

    #layer_hooks = [model.identity_layer_for_input_hook, model.layers.layers[1], model.layers.layers[4], model.layers.layers[7], model.layers.layers[10]]
    log_names = ["input", "layer_1", "layer_4", "layer_7", "logits"]

    features = torch.cat([features, val_features], dim=0)

    for i, layer_hook in enumerate(layer_hooks):

        activation_vecs = access_activations_forward_hook([features], model.forward, layer_hook)

        #colors = []
        #for target in targets:
        #    colors.append(class_colors[target.argmax()])


        tsne(activation_vecs, colors, summary_writer, log_tag="tsne " + log_names[i])



def get_high_res_datasets():
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
        pad_start_ms=60,
        pad_end_ms=60,
    )
    data.setup("test")

    print("len dataset: {}".format(len(data.dataset.data)))
    dataset = MouseAudioDataset(data.dataset.data, use_augmentations=False)
    train_ds, val_ds, test_ds = train_val_test_split(dataset)

    return train_ds, val_ds, test_ds




def get_high_res_val_data():
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
        pad_start_ms=60,
        pad_end_ms=60,
    )
    data.setup("test")

    print("len dataset: {}".format(len(data.dataset.data)))
    dataset = MouseAudioDataset(data.dataset.data, use_augmentations=True)

    train_ds, _, _ = train_val_test_split(dataset)

    dataset = MouseAudioDataset(data.dataset.data, use_augmentations=False)
    _, val_ds, test_ds = train_val_test_split(dataset)

    features = []
    targets = []

    for i in range(len(val_ds)):
        feature, target = val_ds[i]
        features.append(feature)
        targets.append(target)

    return features, targets



def print_accuracy_images(merge_class_1_and_5=False, model=None, classes_to_use=[1, 2, 3, 4, 5]):
    if model is None:
        model = torch.load("ckpts/resnet_model_conv_cls1_to_4.pt").to(DEVICE).eval()

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
        pad_start_ms=60,
        pad_end_ms=60,
    )
    data.setup("test")

    print("len dataset: {}".format(len(data.dataset.data)))
    dataset = MouseAudioDataset(data.dataset.data, use_augmentations=True)

    train_ds, _, _ = train_val_test_split(dataset)

    dataset = MouseAudioDataset(data.dataset.data, use_augmentations=False)
    _, val_ds, test_ds = train_val_test_split(dataset)


    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True,
                                            num_workers=8)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False,
                                            num_workers=8)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False,
                                            num_workers=8)



    num_samples = 0
    num_correct_samples = 0
    with torch.no_grad():
        for x, time_feature, targets in test_loader:
            x = x.to(DEVICE)
            time_feature = time_feature.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(x, time_feature)

            preds = torch.argmax(preds, dim=1)
            targets = torch.argmax(targets, dim=1)

            num_correct_samples += torch.sum(preds == targets)
            num_samples +=  len(preds)
        accuracy = num_correct_samples / num_samples

        print("accuracy: {}".format(accuracy))
        return accuracy


    """
    print(preds[0])
    print(targets[0])

    print(preds.shape)
    print(targets.shape)

    preds = torch.argmax(preds, dim=1)
    if merge_class_1_and_5:
        for i, pred in enumerate(preds):
            if pred == 4:
                preds[i] = 0

    targets = torch.argmax(targets, dim=1)

    print(torch.unique(targets))

    accuracy = torch.sum(preds == targets) / len(preds)
    print("accuracy: {}".format(accuracy))


    num_classes = 5
    if merge_class_1_and_5:
        num_classes = 4

    for i in range(num_classes):
        target_class = i
        accuracy = torch.sum(preds == target_class) / torch.sum(targets == target_class)
        print("pred weight class {}: {}".format(target_class+1, accuracy))

    for i in range(num_classes):
        target_class = i
        accuracy = torch.sum((preds == target_class) * (targets == target_class)) / torch.sum(targets == target_class)
        #accuracy = torch.sum(preds == target_class) / torch.sum(targets == target_class)
        print("accuracy class {}: {}".format(target_class+1, accuracy))

    for i in range(num_classes):
        num_samples = torch.sum(torch.where(targets == i, 1.0, 0.0))
        print("num samples for class {}: {}".format(i+1, num_samples))
    """


def filter_dataset_by_softmax_score(dataset, model, cutoff_value=0.6):
    i = 0
    indices = []
    with torch.no_grad():
        for x, target in dataset:
            x = x.to(DEVICE).unsqueeze(dim=0)
            pred = model(x)
            pred = torch.max(pred)
            if pred > cutoff_value:
                indices.append(i)

            i += 1

    return torch.utils.data.Subset(dataset, indices)

    for x, target in dataset:
        x = x.to(DEVICE).unsqueeze(dim=0)
        pred = model(x)
        pred = torch.max(pred)
        if pred > cutoff_value:
            print("pred: {}".format(pred))
            raise RuntimeError

        i += 1




"""
def print_accuracy(merge_class_1_and_5=False, model=None, classes_to_use=[1, 2, 3, 4, 5]):
    if model is None:
        model = torch.load("ckpts/resnet_model_conv_cls1_to_4.pt").to(DEVICE).eval()

    features, targets = get_test_data()

    features, targets = filter_classes(features, targets, classes_to_use)


    if merge_class_1_and_5:
        for target in targets:
            if target[-1] == 1:
                target[0] = 1
        #targets = torch.where(targets == torch.tensor([0, 0, 0, 0, 1]), torch.tensor([1, 0, 0, 0, 0]), targets)
        targets = targets[:, :4]

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)

    preds = model(features.to(DEVICE))
    #preds = model._forward(features.to(DEVICE))
    #preds[:, 3] += 3.0
    #preds = torch.nn.functional.softmax(preds)

    print(preds[0])
    print(targets[0])

    print(preds.shape)
    print(targets.shape)

    preds = torch.argmax(preds, dim=1)
    if merge_class_1_and_5:
        for i, pred in enumerate(preds):
            if pred == 4:
                preds[i] = 0

    targets = torch.argmax(targets, dim=1)

    print(torch.unique(targets))

    accuracy = torch.sum(preds == targets) / len(preds)
    print("accuracy: {}".format(accuracy))


    num_classes = 5
    if merge_class_1_and_5:
        num_classes = 4

    for i in range(num_classes):
        target_class = i
        accuracy = torch.sum(preds == target_class) / torch.sum(targets == target_class)
        print("pred weight class {}: {}".format(target_class+1, accuracy))

    for i in range(num_classes):
        target_class = i
        accuracy = torch.sum((preds == target_class) * (targets == target_class)) / torch.sum(targets == target_class)
        #accuracy = torch.sum(preds == target_class) / torch.sum(targets == target_class)
        print("accuracy class {}: {}".format(target_class+1, accuracy))

    for i in range(num_classes):
        num_samples = torch.sum(torch.where(targets == i, 1.0, 0.0))
        print("num samples for class {}: {}".format(i+1, num_samples))
"""

def print_attribution():
    #model = classification_net.load_from_checkpoint("logs/lightning_logs/version_14/checkpoints/epoch=39-step=3520.ckpt").to(DEVICE).eval()

    #features, targets = get_val_data()

    #features = features.to(DEVICE)
    #targets = targets.to(DEVICE)

    model = torch.load("ckpts/model.pt").to(DEVICE).eval()
    features = torch.load("ckpts/train_features.pt").to(DEVICE)
    targets = torch.load("ckpts/train_targets.pt").to(DEVICE)

    preds = model(features.to(DEVICE))
    print(preds.shape)
    print(targets.shape)

    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)

    accuracy = torch.sum(preds == targets) / len(preds)

    print("accuracy: {}".format(accuracy))


    DeepLift = captum.attr.DeepLift(model, multiply_by_inputs=True)
    attribution = DeepLift.attribute(features, target=preds)

    attribution = torch.mean(torch.abs(attribution), dim=0).detach()

    print(attribution)
    print(torch.argmax(attribution))




    mask = torch.where(attribution < 0.02, torch.zeros_like(attribution), torch.ones_like(attribution))
    print(mask)
    print("prune num values: {}".format(len(mask) - torch.sum(mask)))



    mask = torch.unsqueeze(mask, dim=0).to(DEVICE)



    preds = model(features*mask)
    #print(preds.shape)
    #print(targets.shape)

    preds = torch.argmax(preds, dim=1)

    accuracy = torch.sum(preds == targets) / len(preds)

    print("accuracy pruned: {}".format(accuracy))





def get_attribution_class(class_idx=1):
    model = torch.load("ckpts/resnet_model_conv_new_data_07.pt").to(DEVICE).eval()
    features, targets = get_val_data()

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)

    preds = model(features.to(DEVICE))
    print(preds.shape)
    print(targets.shape)

    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)

    datapoints_for_attribution = []
    for i, pred in enumerate(preds):
        target = targets[i]
        if target == class_idx:
            if pred == target:
                datapoints_for_attribution.append(features[i])
    datapoints_for_attribution = torch.stack(datapoints_for_attribution)

    baseline = torch.zeros_like(datapoints_for_attribution[0]) - 1.0
    baseline[-1] = 0.033
    baseline = baseline.unsqueeze(dim=0)
    print(baseline.shape)
    print(datapoints_for_attribution.shape)

    DeepLift = captum.attr.DeepLift(model, multiply_by_inputs=True)
    attribution = DeepLift.attribute(datapoints_for_attribution, target=class_idx, baselines=baseline)

    return attribution


class ModelHookBoostClass:
    def __init__(self, layer_hook, boost_direction, boost_factor=1.0):
        layer_hook.register_forward_hook(self.forward_hook)
        self.boost_direction = boost_direction.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        self.boost_factor = boost_factor


    def forward_hook(self, module, input_, output):
        print("forward hook")
        return self.boost_direction * torch.mean(torch.nn.functional.cosine_similarity(self.boost_direction, output)) * self.boost_factor + output
        #return torch.zeros_like(output)



class ModelHookAddChannel:
    def __init__(self, layer_hook, channel=3, constant_add=2.0):
        self.hook = layer_hook.register_forward_hook(self.forward_hook)
        self.channel = channel
        self.constant_add = constant_add


    def forward_hook(self, module, input_, output):
        output[: self.channel] += self.constant_add
        #raise RuntimeError
        return output


    def remove_hook(self):
        self.hook.remove()



def hook_model_boost_class(class_idx, layer_hook, model, boost_factor=1000.0, features=None, targets=None):

    if features is None:
        features, targets = get_val_data()

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)

    preds = model(features.to(DEVICE))
    print(preds.shape)
    print(targets.shape)

    preds = torch.argmax(preds, dim=1)
    #targets = torch.argmax(targets, dim=1)

    datapoints_for_attribution = []
    for i, pred in enumerate(preds):
        target = targets[i]
        if target == class_idx:
            if pred != target:
                datapoints_for_attribution.append(features[i])
    datapoints_for_attribution = torch.stack(datapoints_for_attribution)

    attribution = get_layer_attribution_class(class_idx, layer_hook, datapoints_for_attribution, model)
    #channel_attribution = torch.mean(torch.clamp(attribution, min=0.0, max=1000000000.0), dim=[0, 2, 3])
    channel_attribution = torch.mean(attribution, dim=[0, 2, 3])

    ModelHookBoostClass(layer_hook, channel_attribution, boost_factor)



def get_layer_attribution_class(class_idx, layer_hook, datapoints_for_attribution, model):
    baseline = torch.zeros_like(datapoints_for_attribution[0]) - 1.0
    baseline[-1] = 0.033
    baseline = baseline.unsqueeze(dim=0)
    print(baseline.shape)
    print(datapoints_for_attribution.shape)

    DeepLift = captum.attr.LayerDeepLift(model, layer=layer_hook, multiply_by_inputs=True)
    attribution = DeepLift.attribute(datapoints_for_attribution, target=class_idx, baselines=baseline)

    return attribution




def plot_mean_attribution_class(class_idx=1, summary_writer=None):
    if summary_writer is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

    fig = plt.figure()

    attribution = get_attribution_class(class_idx)
    mean_attribution = torch.mean(attribution, dim=0).unsqueeze(dim=0)

    mean_attribution = torch.reshape(input=mean_attribution[:, :200], shape=(mean_attribution.shape[0], 1, 25, 8))

    mean_attribution = torch.nn.functional.interpolate(mean_attribution, scale_factor=16).detach().cpu()

    plt.imshow(mean_attribution[0][0])
    plt.colorbar()
    summary_writer.add_figure("mean attribution", fig, global_step=class_idx)
    summary_writer.close()


def plot_tsne_of_highest_n_features(class_idx, n_features=3, summary_writer=None):
    if summary_writer is None:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

    attribution = get_attribution_class(class_idx)
    mean_attribution = torch.mean(attribution, dim=0).unsqueeze(dim=0)

    highest_indize = np.argsort(torch.abs(mean_attribution)[0].cpu().detach().numpy())

    features, targets = get_val_data()

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)

    features = torch.stack([features[highest_indize[i]] for i in range(n_features)], dim=1)
    #targets = torch.stack([targets[highest_indize[i]] for i in range(targets)], dim=1)

    #class_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    class_colors = [[1, 0, 0] for i in range(5)]
    class_colors[class_idx] = [0, 0, 1]


    #layer_hooks = [model.identity_layer_for_input_hook, model.layers.layers[1], model.layers.layers[4], model.layers.layers[7], model.layers.layers[10]]
    log_names = ["input", "layer_1", "layer_4", "layer_7", "logits"]

    colors = []
    for target in targets:
        colors.append(class_colors[target.argmax()])

    tsne(features.cpu().detach(), colors, summary_writer, log_tag="tsne " + str(class_idx))







def vary_time_feature():
    with torch.no_grad():
        model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()
        #features = torch.load("ckpts/train_features.pt").to(DEVICE)
        #targets = torch.load("ckpts/train_targets.pt").to(DEVICE)

        features, targets = get_val_data()

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        original_time = torch.clone(features[:, -1])


        for i in range(1):
            features[:, -1] = 5.0#original_time*(i/10)

            preds = model(features)
            #print(preds.shape)

            print("mean preds: {}".format(torch.mean(preds, dim=0)))
            print("time factor: {}".format(i/10))



def plot_image():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

    fig = plt.figure()

    features, targets = get_val_data()

    x = features[1].unsqueeze(dim=0)
    #x = torch.reshape(input=x[:, :200], shape=(x.shape[0], 1, 25, 8))
    x = torch.reshape(input=x[:, :200], shape=(x.shape[0], 1, 25, 8))

    x = torch.nn.functional.interpolate(x, scale_factor=16)

    plt.imshow(x[0][0])
    plt.colorbar()
    summary_writer.add_figure("test", fig)
    summary_writer.close()

    #x = torchvision.utils.save_image(x, "imgs/test.png")







hook_output = None
def hook_func(module, _input, output):
    global hook_output
    hook_output = output


def optim_vis(model, class_number=1):
    global hook_output

    x = -4.0*torch.ones((1, 200), device=DEVICE)
    x += 0.05*torch.randn_like(x)
    x = torch.nn.functional.tanh(x)
    time_feature = torch.tensor(0.0, ).to(DEVICE).unsqueeze(dim=0).unsqueeze(dim=0)

    x.requires_grad = True
    time_feature.requires_grad = False

    #optim = torch.optim.Adam([x, time_feature], lr=5e-2)
    optim = torch.optim.Adam([x], lr=5e-2)

    #model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()
    model.layers.linear_last.register_forward_hook(hook_func)

    #model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()
    #feature_extractor = create_feature_extractor(
    #model, {'layers.layers': 'output'})

    for i in range(128):
        clamped = torch.nn.functional.tanh(x)
        #print(clamped.shape)
        #print(time_feature.shape)

        clamped = torch.reshape(input=clamped, shape=(clamped.shape[0], 1, 25, 8))
        clamped = torch.nn.functional.pad(clamped, (1, 1, 1, 1), mode='constant', value=-1.0)#value=0.5, mode='reflect')

        clamped = torch.roll(clamped, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))
        clamped = clamped[:, :, 1:26, 1:9]
        clamped = torch.reshape(clamped, shape=(clamped.shape[0], 200))

        x_inc_time_feature = torch.cat([clamped, time_feature], dim=1)

        pred = model(x_inc_time_feature)
        print(pred)
        pred_logits = hook_output
        #print(pred_logits)
        loss = -torch.mean(pred_logits[:, class_number-1]) + 2.5*torch.mean(clamped)# + 0.5*torch.mean(pred_logits[:, 2])
        #print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()


    with torch.no_grad():
        print("time feature:  {}".format(time_feature))
        x = torch.nn.functional.tanh(x.detach())


        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

        fig = plt.figure()

        x = torch.reshape(input=x, shape=(x.shape[0], 1, 25, 8))

        x = torch.nn.functional.interpolate(x, scale_factor=16).cpu()

        plt.imshow(x[0][0], vmin=-1.0, vmax=1.0)
        plt.colorbar()
        summary_writer.add_figure("optim vis", fig, global_step=class_number)
        summary_writer.close()



def optim_vis_high_res_model(model, layer, layer_relu_hook_point=None, class_number=1, plot_image=False, save_path=None, layer_str=None, initialization=None):
    global hook_output

    if layer_relu_hook_point is not None:
        pass_grad_hook = layer_relu_hook_point.register_backward_hook(backward_hook_pass_grad_directly)#model.layers.resnet_block_01.activation_01.register_backward_hook(backward_hook_pass_grad_directly)

    #x = -4.0*torch.ones((1, 200), device=DEVICE)
    if initialization is None:
        x = torch.zeros((1, 2, 201, 150), device=DEVICE)
        x = 0.05*torch.randn_like(x)
    else:
        x = initialization.to(DEVICE)
    #x = torch.nn.functional.tanh(x)
    time_feature = torch.tensor(0.18816110491752625, ).to(DEVICE).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)

    #print("time feature shape: {}".format(time_feature.shape))
    #print("x shape: {}".format(x.shape))

    x.requires_grad = True
    time_feature.requires_grad = False

    #optim = torch.optim.Adam([x, time_feature], lr=5e-2)
    optim = torch.optim.Adam([x], lr=5e-2)

    #model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()
    #model.layers.linear_last.register_forward_hook(hook_func)
    hook = layer.register_forward_hook(hook_func)

    #model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()
    #feature_extractor = create_feature_extractor(
    #model, {'layers.layers': 'output'})

    for i in tqdm(range(128), ascii=True):
        if i > 16:
            pass_grad_hook.remove()
        #clamped = torch.nn.functional.tanh(x)
        #print(clamped.shape)
        #print(time_feature.shape)

        #with torch.no_grad():
        #    x[:, 0] = torch.clamp(x[:, 0], min=-0.02, max=10000000)

        with torch.no_grad():
            #x[:, -1, :, :] = 0.33#0.18816110491752625
            #x[:, 0, :, :] = 0.0
            pass
        x_step = torch.cat([x, time_feature*torch.ones_like(x)[:, 0].unsqueeze(dim=1)], dim=1)

        #x_step = torch.nn.functional.pad(x_step, (1, 1, 1, 1), mode='constant', value=0.0)#value=0.5, mode='reflect')
        #x_step[:, 1] = torch.clamp(x_step[:, 1], min=0.0, max=10000000)

        x_step = torch.roll(x_step, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))
        x_step = torch.nn.functional.interpolate(x_step, scale_factor=random.randint(90, 110) / 100.0)

        pred_logits = model.forward_logits(x_step)
        activation = hook_output
        #print(pred_logits)
        loss =  -torch.mean(activation[:, class_number-1])# - 10.0*torch.mean(activation) #+ 30.0*torch.mean(F.l1_loss(x_step, 0.0*torch.ones_like(x_step))) #20*torch.mean(x_step)#50.0*torch.mean(F.l1_loss(x_step, -3.0*torch.ones_like(x_step))) # + 2.5*torch.mean(x_step[:, 1])# + 0.5*torch.mean(pred_logits[:, 2])
        #print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()



    with torch.no_grad():
        #pred = F.softmax(pred_logits, dim=1)
        #model_1 = classification_net_cnn_image_lightning.load_from_checkpoint("logs/full_labeled_data_10_cross_validation/lightning_logs/version_1/checkpoints/epoch=330-step=30452.ckpt").to(DEVICE).eval()
        #print(pred)

        #with torch.no_grad():
        #    print("pred model_1 : {}".format(model_1(x_step)))

        print("time feature:  {}".format(time_feature))
        #x = torch.nn.functional.tanh(x.detach())
        x = x.detach().cpu()
        #x[:, 0] = torch.clamp(x[:, 0], min=-0.02, max=1000000)

        if plot_image:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

            fig = plt.figure()
            plt.imshow(x[0][0])
            plt.colorbar()
            summary_writer.add_figure("optim vis spectogram", fig, global_step=class_number)

            fig = plt.figure()
            plt.imshow(x[0][1])
            plt.colorbar()
            summary_writer.add_figure("optim vis spectogram db", fig, global_step=class_number)


            summary_writer.close()
        else:
            if save_path is not None:
                if layer_str is not None:
                    save_path = save_path + layer_str
                import pathlib
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

                fig = plt.figure()
                plt.imshow(x[0][0])
                #plt.colorbar()
                plt.axis('off')
                plt.savefig(save_path + "spec_" + str(class_number) + ".png")
                plt.close()

                fig = plt.figure()
                plt.imshow(x[0][1])
                plt.axis('off')
                #plt.colorbar()
                plt.savefig(save_path + "scaled_spec_" + str(class_number) + ".png")
                plt.close()

                x = torch.cat([x.cuda(), time_feature.cuda()*torch.ones_like(x.cuda())[:, 0].unsqueeze(dim=1)], dim=1)


                baseline = torch.zeros_like(x)
                baseline[:, -1] = 0.18816110491752625
                """
                attribution = compute_attribution(model, x, baseline, class_number-1).cpu().detach()

                fig = plt.figure()
                plt.imshow(attribution[0][1])
                plt.colorbar()
                plt.savefig(save_path + "attribution_scaled_spec_" + str(class_number) + ".png")
                plt.close()
                """

        hook.remove()

        return x




def reshape_to_spatial(x):
    time_feature = x[:, -1]
    x = torch.reshape(input=x[:, :200], shape=(x.shape[0], 1, 25, 8))
    return x, time_feature


def plot_attribution_overlay(model, features, target_class, baseline, n_images=5):
    #model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "img" + ' ' + current_time + '')


    #baseline = get_baseline()
    #print("baseline pred: {}".format(model(baseline)))

    #features, targets = get_val_data()

    #features = features.to(DEVICE)
    #targets = targets.to(DEVICE)

    baseline = baseline.to(DEVICE)
    n_plotted = 0

    for i, feature in enumerate(features):
        feature = torch.unsqueeze(feature, dim=0).to(DEVICE)

        #print("feature shape: {}".format(feature.shape))
        pred = model(feature)
        pred = torch.argmax(pred, dim=1)
        #target = torch.argmax(targets[i].unsqueeze(dim=0), dim=1).to(DEVICE)
        if pred == target_class:
            #if pred == target:
            attribution = compute_attribution(model, feature, baseline, target_class)

            print("attribution: {}".format(torch.mean(attribution, dim=[0, 2, 3])))

            #print(attribution)
            #attribution, attribution_time = reshape_to_spatial(attribution)
            attribution = attribution.detach().cpu()

            #img, time_feature = reshape_to_spatial(feature)
            img = feature
            img = img.detach().cpu()

            fig = plt.figure(figsize=(20, 10))

            plt.imshow(img[0][1])
            plt.colorbar()
            summary_writer.add_figure("img", fig, global_step=n_plotted)
            summary_writer.close()



            fig = plt.figure(figsize=(20, 10))

            plt.imshow(attribution[0][1])
            plt.colorbar()
            summary_writer.add_figure("attribution", fig, global_step=n_plotted)
            summary_writer.close()

            #print("time feature: {}, attribution time feature: {}".format(time_feature, attribution_time))
            n_plotted += 1

            if n_plotted >= n_images:
                return




def compute_attribution(model, x, baseline, target_class):
    ig = captum.attr.IntegratedGradients(model, multiply_by_inputs=True)
    attribution_method = captum.attr.NoiseTunnel(ig)
    attribution = attribution_method.attribute(inputs=x, baselines=baseline, target=target_class, stdevs=0.1, internal_batch_size=1, nt_samples=10)

    return attribution



def compute_attribution_hidden_layer(model, layer, x, baseline, target_class):
    ig = captum.attr.LayerIntegratedGradients(model, layer, multiply_by_inputs=True)
    attribution_method = captum.attr.NoiseTunnel(ig)
    attribution = attribution_method.attribute(inputs=x, baselines=baseline, target=target_class, stdevs=0.1, internal_batch_size=1, nt_samples=10)

    return attribution



def get_baseline():
    train_features, _ = get_train_data()

    mean_data = torch.mean(train_features.to(DEVICE), dim=0).unsqueeze(dim=0)*0.45

    baseline = -torch.ones((1, 201))
    baseline[:, -1] = 0.0
    return baseline.to(DEVICE)





def print_attribution_target_class(target_class=0, absolute_value=True):

    model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()
    #features = torch.load("ckpts/train_features.pt").to(DEVICE)
    #targets = torch.load("ckpts/train_targets.pt").to(DEVICE)

    features, targets = get_val_data()

    features = features.to(DEVICE)
    targets = targets.to(DEVICE)


    train_features, _ = get_train_data()


    baseline = torch.mean(train_features.to(DEVICE), dim=0).unsqueeze(dim=0)*0.45

    print("baseline pred: {}".format(model(baseline)))
    #raise RuntimeError

    preds = model(features.to(DEVICE))
    print(preds.shape)
    print(targets.shape)

    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)

    inputs_for_attribution = []
    for i, pred in enumerate(preds):
        target = targets[i]
        if pred == target:
            inputs_for_attribution.append(features[i])

    inputs_for_attribution = torch.stack(inputs_for_attribution)

    accuracy = torch.sum(preds == targets) / len(preds)

    print("accuracy: {}".format(accuracy))


    DeepLift = captum.attr.DeepLift(model, multiply_by_inputs=True)
    ig = captum.attr.IntegratedGradients(model, multiply_by_inputs=True)
    attribution_method = captum.attr.NoiseTunnel(ig)
    attribution = attribution_method.attribute(inputs=inputs_for_attribution, baselines=baseline, target=target_class, stdevs=0.1)

    if absolute_value:
        attribution = torch.mean(torch.abs(attribution), dim=0).detach()
    else:
        attribution = torch.mean(attribution, dim=0).detach()
    print(attribution)





def test_baselines():
    #model = torch.load("ckpts/model.pt").to(DEVICE).eval()
    model = torch.load("ckpts/model_conv.pt").to(DEVICE).eval()

    features, targets = get_train_data()

    #features = features.to(DEVICE)
    #targets = targets.to(DEVICE)

    #features = torch.load("ckpts/train_features.pt").to(DEVICE)

    #print(features[0])

    baseline = -torch.ones((1, 201), device=DEVICE)
    baseline[:, -1] = 0.05
    pred = model(baseline)
    print(pred)



def backward_hook_pass_grad_directly(module, input_, output):
    return output



def plot_channel_vis(model, layers, layer_names, layers_relu=None, initialization=None):
    for j, layer in enumerate(layers):
        #num_channels = access_activations_forward_hook([torch.zeros((1, 2, 128, 150), device=DEVICE), torch.zeros((1, 1, 1, 1), device=DEVICE)], model, layer).shape[1]
        num_channels = access_activations_forward_hook([torch.zeros((1, 3, 201, 150), device=DEVICE)], model, layer).shape[1]

        channel_visualizations = []
        for i in range(num_channels):
            layer_str = layer_names[j]
            if layers_relu is not None:
                layer_relu = layers_relu[j]
            channel_vis = optim_vis_high_res_model(model, layer=layer, layer_relu_hook_point=layer_relu, class_number=i+1, plot_image=False, save_path="save_figs/channel_vis/", layer_str=layer_str,
                                                   initialization=initialization)
            #break
            #channel_visualizations.append(channel_vis[0])

        #channel_visualizations = torch.stack(channel_visualizations, dim=0)
        #channel_visualizations_spec = torch.reshape(channel_visualizations[:, 0], (channel_visualizations.shape[0] * channel_visualizations.shape[2], channel_visualizations.shape[3]))
        #channel_visualizations_spec_scaled = torch.reshape(channel_visualizations[:, 1], (channel_visualizations.shape[0] * channel_visualizations.shape[2], channel_visualizations.shape[3]))

        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = SummaryWriter('/home/rherdt/logs/gradient_tape/dummy ' + "tsne" + ' ' + current_time + '')

        fig = plt.figure()
        plt.imshow(channel_visualizations_spec)
        plt.colorbar()
        summary_writer.add_figure("optim vis spectogram", fig, global_step=i)

        fig = plt.figure()
        plt.imshow(channel_visualizations_spec_scaled)
        plt.colorbar()
        summary_writer.add_figure("optim vis spectogram db", fig, global_step=i)


        summary_writer.close()
        """

def get_datasets_cross_val(val_crop_size=170, normalize_smooth_spec_individually=False, resize_size=None,
                           categories=[1, 2, 3, 4, 5]):
    data_dir1 = 'data/automatic_detection_manual_classification'
    data_dir2 = 'data/labeled'

    auto_manu_ds = mouse_dataset.from_folder(data_dir1, name='auto-manu-set', categories=categories, category_map=CATEGORY_MAP, pad_start_ms=60, pad_end_ms=60)
    labeled_ds = mouse_dataset.from_folder(data_dir2, name='manu-set', categories=categories, category_map=CATEGORY_MAP, wav_ext='_SOX.WAV', pad_start_ms=60, pad_end_ms=60)

    """
    data = mouse_data_module(
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=32,
        num_workers=8,
        #categories=[-1, 1, 2, 3, 4, 5],
        #augmentations=None,
        #pct_left=1,
        #pad_same_length=False,
        #pad_factor=None,
        #max_pad=None,
        spectrogram_specs=spectrogram_specs,
        pad_start_ms=60,
        pad_end_ms=60,
    )
    data.setup("test")
    """
    # extract mean std from training data for data normalization
    dataset_for_normalization = MouseAudioDataset_RegularSpectrogram(labeled_ds.data, use_augmentations=False, do_normalization=False,
                                                                        pad_to_same_size=True,
                                                                        final_crop_size_no_aug=170)
    #train_ds, _, _ = train_val_test_split(dataset_for_normalization, prefix=str(train_index) + "_")
    #print("len dataset: {}".format(len(train_ds)))
    #mean_spectrogram, std_spectrogram, mean_scaled_spectrogram, std_scaled_spectrogram = get_mean_std(train_ds)
    mean_spectrogram, std_spectrogram, mean_scaled_spectrogram, std_scaled_spectrogram = get_mean_std(dataset_for_normalization)

    """
    data = mouse_data_module(
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=32,
        num_workers=8,
        category_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
        #augmentations=None,
        #pct_left=1,
        #pad_same_length=False,
        #pad_factor=None,
        #max_pad=None,
        spectrogram_specs=spectrogram_specs,
        pad_start_ms=60,
        pad_end_ms=60,
        data_dir="./data/automatic_detection_manual_classification",
        wav_ext=".WAV",
    )
    data.setup("test")
    """

    train_datasets = []
    val_datasets = []
    print("start validation split...")
    train_datasets_mouse, val_datasets_mouse = cross_validation_split_mouse_visu(auto_manu_ds)
    for i, train_ds in enumerate(train_datasets_mouse):
        val_ds = val_datasets_mouse[i]

        dataset = MouseAudioDataset_RegularSpectrogram(val_ds.data, mean_spectogram=mean_spectrogram, std_spectogram=std_spectrogram,
                                                mean_scaled_spectogram=mean_scaled_spectrogram, std_scaled_spectogram=std_scaled_spectrogram,
                                                final_crop_size_no_aug=val_crop_size, normalize_smooth_spec_individually=normalize_smooth_spec_individually,
                                                resize_size=resize_size)
        val_datasets.append(dataset)

        dataset = MouseAudioDataset_RegularSpectrogram(train_ds.data, mean_spectogram=mean_spectrogram, std_spectogram=std_spectrogram,
                                                mean_scaled_spectogram=mean_scaled_spectrogram, std_scaled_spectogram=std_scaled_spectrogram,
                                                final_crop_size_no_aug=val_crop_size, normalize_smooth_spec_individually=normalize_smooth_spec_individually,
                                                resize_size=resize_size)
        train_datasets.append(dataset)

    print("finished validation split!")

    """
    #print("len dataset: {}".format(len(data.dataset.data)))
    dataset = MouseAudioDataset_RegularSpectrogram(auto_manu_ds.data, mean_spectogram=mean_spectrogram, std_spectogram=std_spectrogram,
                                            mean_scaled_spectogram=mean_scaled_spectrogram, std_scaled_spectogram=std_scaled_spectrogram,
                                            final_crop_size_no_aug=val_crop_size)

    train_datasets, val_datasets = cross_validation_split(dataset, num_splits=10, seed=42)
    """
    return train_datasets, val_datasets


def datasets_to_dataloaders(datasets):
    dataloaders = []

    for dataset in datasets:
        dl = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,
                                                num_workers=8)
        dataloaders.append(dl)

    return dataloaders



def filter_dataset_by_target_class(dataset, wanted_class):
    indices = []
    i = 0
    for data, target in dataset:
        if torch.argmax(target) == wanted_class:
            indices.append(i)
        i += 1

    return torch.utils.data.Subset(dataset, indices)





def get_dataloaders(labeled=True, prefix='0_', cutoff_value=None):

    if not labeled:
        prefix += "automatic_detection_"

    if labeled:
        data = mouse_data_module(
            train_val_test_split=[0.8, 0.1, 0.1],
            batch_size=32,
            num_workers=8,
            category_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
            #augmentations=None,
            #pct_left=1,
            #pad_same_length=False,
            #pad_factor=None,
            #max_pad=None,
            spectrogram_specs=spectrogram_specs,
            pad_start_ms=60,
            pad_end_ms=60,
            #data_dir="./data/automatic_detection_manual_classification",
            #wav_ext=".WAV",
        )
    else:
        data = mouse_data_module(
            train_val_test_split=[0.8, 0.1, 0.1],
            batch_size=32,
            num_workers=8,
            category_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
            #augmentations=None,
            #pct_left=1,
            #pad_same_length=False,
            #pad_factor=None,
            #max_pad=None,
            spectrogram_specs=spectrogram_specs,
            pad_start_ms=60,
            pad_end_ms=60,
            data_dir="./data/automatic_detection_manual_classification",
            wav_ext=".WAV",
        )
    data.setup("test")


    data_normalization = mouse_data_module(
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=32,
        num_workers=8,
        #categories=[-1, 1, 2, 3, 4, 5],
        #augmentations=None,
        #pct_left=1,
        #pad_same_length=False,
        #pad_factor=None,
        #max_pad=None,
        spectrogram_specs=spectrogram_specs,
        pad_start_ms=60,
        pad_end_ms=60,
    )
    data_normalization.setup("test")

    # extract mean std from training data for data normalization
    dataset_for_normalization = MouseAudioDataset_RegularSpectrogram(data_normalization.dataset.data, use_augmentations=False, do_normalization=False,
                                                                     pad_to_same_size=True,
                                                                     final_crop_size_no_aug=170)
    train_ds, _, _ = train_val_test_split(dataset_for_normalization, prefix=prefix[0] + "_")
    #print("len dataset: {}".format(len(train_ds)))
    mean_spectrogram, std_spectrogram, mean_scaled_spectrogram, std_scaled_spectrogram = get_mean_std(train_ds)


    dataset = MouseAudioDataset_RegularSpectrogram(data.dataset.data, mean_spectogram=201266192.0, std_spectogram=28543963136.0,
                                            mean_scaled_spectogram=57.99601745605469, std_scaled_spectogram=6.7896833419799805)


    #print("len dataset: {}".format(len(data.dataset.data)))
    #dataset = MouseAudioDataset_RegularSpectrogram(data.dataset.data, mean_spectogram=mean_spectrogram, std_spectogram=std_spectrogram,
    #                                        mean_scaled_spectogram=mean_scaled_spectrogram, std_scaled_spectogram=std_scaled_spectrogram)

    train_ds, val_ds, test_ds = train_val_test_split(dataset, force_new=False, prefix=prefix)

    if cutoff_value is not None:
        val_ds = filter_dataset_by_softmax_score(val_ds, model, cutoff_value=cutoff_value)
        test_ds = filter_dataset_by_softmax_score(test_ds, model, cutoff_value=cutoff_value)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True,
                                            num_workers=8)

    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False,
                                            num_workers=8)

    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False,
                                            num_workers=8)


    return train_dl, val_dl, test_dl, len(val_ds), len(test_ds)



def plot_model_attribution(model, dataset, baseline, n_samples=4):
    save_path_specs = "save_figs/attribution/"

    Path(save_path_specs).mkdir(parents=True, exist_ok=True)

    i = 0
    for data, target in dataset:
        target = torch.argmax(target)
        data = torch.unsqueeze(data, dim=0).to(DEVICE)

        IntegratedGrad = captum.attr.NoiseTunnel(captum.attr.IntegratedGradients(model, multiply_by_inputs=True))
        attribution = IntegratedGrad.attribute(data, target=target, internal_batch_size=8, nt_samples=5, nt_samples_batch_size=1, stdevs=0.1,
                                         baselines=baseline).detach().cpu()


        print(attribution.shape)
        fig = plt.figure(dpi=200)
        plt.imshow(attribution[0][1])
        plt.colorbar()
        plt.savefig(save_path_specs + str(i) + ".png")

        fig = plt.figure(dpi=200)
        plt.imshow(data[0][1].cpu().detach().numpy())
        plt.colorbar()
        plt.savefig(save_path_specs + "data_" + str(i) + ".png")


        j = 0
        for layer in [model.layers.layer0, model.layers.layer1, model.layers.layer2]:
            IntegratedGrad = captum.attr.NoiseTunnel(captum.attr.LayerIntegratedGradients(model, layer=layer, multiply_by_inputs=True))
            attribution = IntegratedGrad.attribute(data, target=target, internal_batch_size=8, nt_samples=5, nt_samples_batch_size=1, stdevs=0.1,
                                            baselines=baseline).detach().cpu()

            print(attribution.shape)
            fig = plt.figure(dpi=200)
            plt.imshow(attribution[0][1])
            plt.colorbar()
            plt.savefig(save_path_specs + "layer_" + str(j) + " " + str(i) + ".png")

            overlay = overlay_image_with_activation(data[0][1].detach().cpu(), attribution[0][1])
            fig = plt.figure(dpi=200)
            plt.imshow(overlay)
            plt.colorbar()
            plt.savefig(save_path_specs + "overlay layer_" + str(j) + " " + str(i) + ".png")


            j += 1


        i += 1
        if i > n_samples:
            return


def example_plot_attribution_samples():
    print("example plot attribution samples...")
    logs_dir = "logs/full_labeled_data_10_cross_validation"

    models = get_lightning_models_from_folder(logs_dir)
    print("load datasets...")
    train_datasets, val_datasets = get_datasets_cross_val()
    print("done loading datasets!")

    #print(val_datasets[0][0][0].shape)
    #raise RuntimeError

    baseline = torch.zeros_like(val_datasets[0][0][0]).unsqueeze(dim=0).to(DEVICE)
    baseline[:, 2] = 0.18816110491752625

    with torch.no_grad():
        print("baseline pred: {}".format(models[0](baseline)))

    print("filter by target class...")
    val_ds = filter_dataset_by_target_class(val_datasets[0], wanted_class=0)
    print("plot attribution...")
    #backward_fix_hook_model(models[0])
    plot_model_attribution(models[0], val_ds, baseline=baseline, n_samples=4)




def attribution_towards_inputs(model, dataloader):
    attributions = []
    model = model.eval().to(DEVICE)

    for data, target in dataloader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        target = torch.argmax(target, dim=-1)
        IntegratedGrad = captum.attr.NoiseTunnel(captum.attr.IntegratedGradients(model, multiply_by_inputs=True))
        baseline = torch.zeros_like(data)
        baseline[:, 2] = 0.18816110491752625
        attribution = IntegratedGrad.attribute(data, target=target, internal_batch_size=32, nt_samples=5, nt_samples_batch_size=1, stdevs=0.1,
                                         baselines=baseline).detach().cpu()
        attributions.append(attribution)

    with torch.no_grad():
        baseline = torch.zeros_like(data)[0].unsqueeze(dim=0)
        baseline_pred = model(baseline)
        #print("baseline pred: {}".format(baseline_pred))

    final_attribution = torch.cat(attributions, dim=0)
    #print("final attribution shape: {}".format(final_attribution.shape))

    #print("final attribution: {}".format(torch.mean(final_attribution, dim=[0, 2, 3])))

    final_attribution_abs = torch.mean(torch.abs(final_attribution), dim=[0])
    final_attribution = torch.mean(final_attribution, dim=[0])

    return torch.sum(final_attribution_abs, dim=[1, 2]), torch.sum(final_attribution, dim=[1, 2])



def get_lightning_ckpt_files_step(file_path, ckpt_paths):
    if ".ckpt" in file_path:
        ckpt_paths.append(file_path)
    elif os.path.isdir(file_path):
        for file in sorted(os.listdir(file_path)):
            get_lightning_ckpt_files_step(os.path.join(file_path, file), ckpt_paths)

def get_lightning_ckpt_file_paths_recursively(folder_path):
    ckpt_paths = []
    get_lightning_ckpt_files_step(folder_path, ckpt_paths)
    return ckpt_paths


def plot_lightning_models_accuracy_over_val_size(logs_dir, model_class=classification_net_cnn_image_lightning,
                                                  val_crop_sizes=[130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190]):

    models = get_lightning_models_from_folder(logs_dir, model_class=model_class)
    mean_val_accuracies = []
    std_val_accuracies = []
    #i = 0
    for val_crop_size in tqdm(val_crop_sizes):
        train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=val_crop_size, normalize_smooth_spec_individually=True)
        train_loaders = datasets_to_dataloaders(train_datasets)
        val_loaders = datasets_to_dataloaders(val_datasets)

        val_mean_acc, val_std_acc = print_models_accuracy(models, train_loaders, val_loaders)

        mean_val_accuracies.append(val_mean_acc)
        std_val_accuracies.append(val_std_acc)
        #i += 1
        #if i > 1:
        #    break


    print("val_mean_acc: {}".format(mean_val_accuracies))
    print("val_std_acc: {}".format(std_val_accuracies))

    fig = plt.figure(dpi=200)
    #plt.plot(val_crop_sizes, mean_val_accuracies)
    plt.errorbar(val_crop_sizes, mean_val_accuracies, yerr=std_val_accuracies)
    plt.savefig("save_figs/acc_val_size/efficentnetb5_img.png")




def print_lightning_models_stats(logs_dir, model_class=classification_net_cnn_image_lightning, val_crop_size=170, normalize_smooth_spec_individually=False,
                                 resize_size=None):
    models = get_lightning_models_from_folder(logs_dir, model_class=model_class)
    train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=val_crop_size, normalize_smooth_spec_individually=normalize_smooth_spec_individually,
                                                          resize_size=resize_size)
    #train_loaders = datasets_to_dataloaders(train_datasets)
    val_loaders = datasets_to_dataloaders(val_datasets)

    print_models_stats(models, val_loaders)

    #benchmark_dataloader_time(models[0], val_loaders[0])


def print_lightning_models_stats_confidence(logs_dir, model_class=classification_net_cnn_image_lightning, val_crop_size=170, normalize_smooth_spec_individually=False,
                                 resize_size=None, model_name_for_savefig="efficentnetb5", use_temperature_scaling=False, run_individual_models=False):
    models = get_lightning_models_from_folder(logs_dir, model_class=model_class)
    train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=val_crop_size, normalize_smooth_spec_individually=normalize_smooth_spec_individually,
                                                          resize_size=resize_size)
    #train_loaders = datasets_to_dataloaders(train_datasets)
    val_loaders = datasets_to_dataloaders(val_datasets)

    if use_temperature_scaling:
        models_temperature_scaled = []
        for i, model in enumerate(models):
            val_dl = val_loaders[i]
            model = ModelWithTemperature(model)
            model.set_temperature(val_dl, num_steps=10000)
            model = SoftmaxWrapper(model)
            models_temperature_scaled.append(model)
        models = models_temperature_scaled

    if run_individual_models:
        for i, model in enumerate(models):
            models_step = [model]
            val_loaders_step = [val_loaders[i]]
            get_model_accuracy_confidence_cutoff(models_step, val_loaders_step, model_name_for_savefig + "_" + str(i))

    else:
        get_model_accuracy_confidence_cutoff(models, val_loaders, model_name_for_savefig)


class SoftmaxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.softmax(x)


def benchmark_dataloader_time(model, val_loader):
    start_time = time.time()
    with torch.no_grad():
        for spectrogram, target in val_loader:
            model(spectrogram.to(DEVICE))
    print("time taken with gpu: {}".format(time.time() - start_time))

    model = model.cpu()
    start_time = time.time()
    with torch.no_grad():
        for spectrogram, target in val_loader:
            model(spectrogram)
    print("time taken with cpu: {}".format(time.time() - start_time))










def print_lightning_models_attribution(logs_dir):
    models = get_lightning_models_from_folder(logs_dir)
    train_datasets, val_datasets = get_datasets_cross_val()

    absolute_attributions = []
    attributions = []

    absolute_attributions_class = [[] for i in range(5)]
    attributions_class = [[] for i in range(5)]

    for j, model in tqdm(enumerate(models)):
        val_loaders = datasets_to_dataloaders(val_datasets)
        absolute_attribution, attribution = attribution_towards_inputs(model, val_loaders[j])

        absolute_attributions.append(absolute_attribution)
        attributions.append(attribution)

        #print("absolute attribution: {},  class: {}".format(absolute_attribution, "all"))
        #print("attribution: {},  class: {}".format(attribution, "all"))

        for i in range(5):
            filtered_val_datasets = []
            for val_ds in val_datasets:
                val_ds = filter_dataset_by_target_class(val_ds, i)
                filtered_val_datasets.append(val_ds)

            val_loaders = datasets_to_dataloaders(filtered_val_datasets)

            absolute_attribution, attribution = attribution_towards_inputs(models[j], val_loaders[j])
            absolute_attributions_class[i].append(absolute_attribution)
            attributions_class[i].append(attribution)
        #break

    absolute_attributions = torch.stack(absolute_attributions, dim=0)
    attributions = torch.stack(attributions, dim=0)

    print("absolute attribution: {},  class: {}".format(torch.mean(absolute_attributions, dim=0), "all"))
    print("std absolute attribution: {},  class: {}".format(torch.std(absolute_attributions, dim=0), "all"))
    print("attribution: {},  class: {}".format(torch.mean(attributions, dim=0), "all"))
    print("std attribution: {},  class: {}".format(torch.std(attributions, dim=0), "all"))


    for i in range(5):
        absolute_attributions_class[i] = torch.stack(absolute_attributions_class[i], dim=0)
        attributions_class[i] = torch.stack(attributions_class[i], dim=0)

        print("absolute attribution: {},  class: {}".format(torch.mean(absolute_attributions_class[i], dim=0), i))
        print("std absolute attribution: {},  class: {}".format(torch.std(absolute_attributions_class[i], dim=0), i))
        print("attribution: {},  class: {}".format(torch.mean(attributions_class[i], dim=0), i))
        print("std attribution: {},  class: {}".format(torch.std(attributions_class[i], dim=0), i))


def get_filtered_datasets_by_gen_line(val_dataset):
    glu_ds = val_dataset.select_wav_files_by_name(["glu397"])
    ko_ds = val_dataset.select_wav_files_by_name(["KO"])
    #ko9_ds = val_dataset.select_wav_files_by_name(["KO_9"])
    #ko10_ds = val_dataset.select_wav_files_by_name(["KO_10"])
    r142L_ds = val_dataset.select_wav_files_by_name(["r142l", "R142L"])

    datasets = [glu_ds, ko_ds, r142L_ds]
    return datasets

def get_filtered_datasets_by_age(val_dataset):
    p4_ds = val_dataset.select_wav_files_by_name(["p4"])
    p8_ds = val_dataset.select_wav_files_by_name(["p8"])
    p12_ds = val_dataset.select_wav_files_by_name(["p12"])

    datasets = [p4_ds, p8_ds, p12_ds]
    return datasets

def get_filtered_datasets_by_mouse_id(val_dataset):
    R142L_ds = val_dataset.select_wav_files_by_name(["R142L", "r142l"])
    id2_ds = R142L_ds.select_wav_files_by_mouse_id("2")
    id4_ds = R142L_ds.select_wav_files_by_mouse_id("4")
    id8_ds = R142L_ds.select_wav_files_by_mouse_id("8")

    glu397_ds = val_dataset.select_wav_files_by_name(["glu397"])
    mother_21 = glu397_ds.select_wav_files_by_mother_id("21")
    mother_28 = glu397_ds.select_wav_files_by_mother_id("28")
    mother_35 = glu397_ds.select_wav_files_by_mother_id("35")
    mother_40 = glu397_ds.select_wav_files_by_mother_id("40")
    mother_9874 = glu397_ds.select_wav_files_by_mother_id("9874")

    mother_21_id_1 = mother_21.select_wav_files_by_mouse_id("1")
    mother_21_id_2 = mother_21.select_wav_files_by_mouse_id("2")
    mother_28_id_2 = mother_28.select_wav_files_by_mouse_id("2")
    mother_35_id_7 = mother_35.select_wav_files_by_mouse_id("7")
    mother_35_id_5 = mother_35.select_wav_files_by_mouse_id("5")
    mother_40_id_4 = mother_40.select_wav_files_by_mouse_id("4")
    mother_40_id_3 = mother_40.select_wav_files_by_mouse_id("3")
    mother_9874_id_3 = mother_9874.select_wav_files_by_mouse_id("3")
    mother_9874_id_1 = mother_9874.select_wav_files_by_mouse_id("1")

    ko_ds = val_dataset.select_wav_files_by_name(["KO"])
    id1_ds = ko_ds.select_wav_files_by_mouse_id("1")
    id3_ds = ko_ds.select_wav_files_by_mouse_id("3")
    id6_ds = ko_ds.select_wav_files_by_mouse_id("6")


    datasets = [id2_ds, id4_ds, id8_ds, id1_ds, id3_ds, id6_ds, mother_21_id_1, mother_21_id_2, mother_28_id_2,
                mother_35_id_7, mother_35_id_5, mother_40_id_4, mother_40_id_3, mother_9874_id_3,
                mother_9874_id_1]
    return datasets



def get_filtered_datasets_by_genotype(val_dataset):
    genotype_0_ds = val_dataset.select_wav_files_by_genotype("0")
    genotype_1_ds = val_dataset.select_wav_files_by_genotype("1")

    #genotype_f_ds = val_dataset.select_wav_files_by_genotype("f")
    #genotype_m_ds = val_dataset.select_wav_files_by_genotype("m")


    datasets = [genotype_0_ds, genotype_1_ds]#, genotype_f_ds, genotype_m_ds]
    return datasets


def get_filtered_datasets_by_class(val_dataset):
    datasets = []
    for i in range(5):
        ds = val_dataset.select_data_by_class(i+1)
        datasets.append(ds)
    """
    for i in range(6):
        ds = val_dataset.select_data_by_class(i)
        datasets.append(ds)
    """
    return datasets



def run_tsne_models():
    model = get_lightning_models_from_folder("logs/efficentnetb5", model_class=classification_net_cnn_image_lightning_EfficentNetB5)[0].to(DEVICE).eval()
    print("num features: {}".format(len(model.layers.model.features)))

    layer_hooks = [model.layers.model.features[i] for i in range(9)]
    layer_names = ["features[" + str(i) + "]" for i in range(9)]

    #raise RuntimeError

    train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=170)

    #layer_hooks = [model.layers.resnet_block_01, model.layers.layer0, model.layers.layer1, model.layers.layer2, model.layers.linear_last]#[model.layers.resnet_block_01]
    #layer_names = ["resnet_block_01", "layer0", "layer1", "layer2", "linear_last"]#["resnet_block_01"]

    run_tsne_gen_types(model, layer_hooks, layer_names)#(categories=[i for i in range(6)])



def run_tsne_gen_types(model, layer_hooks, layer_names, categories=[1, 2, 3, 4, 5]):
    #model = get_lightning_models_from_folder("logs/full_labeled_data_10_cross_validation")[0].to(DEVICE).eval()
    train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=170, categories=categories)
    val_dataset = val_datasets[0]

    val_loader = datasets_to_dataloaders([val_dataset])[0]

    class_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                    [0, 0.0, 0.0], [0, 0.25, 1], [0, 1.0, 0.25]]


    #layer_hooks = [model.layers.resnet_block_01, model.layers.layer0, model.layers.layer1, model.layers.layer2, model.layers.linear_last]#[model.layers.resnet_block_01]
    #layer_names = ["resnet_block_01", "layer0", "layer1", "layer2", "linear_last"]#["resnet_block_01"]
    feature_types = ["class", "gen", "age", "genotype", "mouse"]#, "mouse"]
    filter_functions = [get_filtered_datasets_by_class, get_filtered_datasets_by_gen_line, get_filtered_datasets_by_age, get_filtered_datasets_by_genotype, get_filtered_datasets_by_mouse_id]#, get_filtered_datasets_by_mouse_id]

    for layer_index, layer_hook in enumerate(layer_hooks):
        X_embedded_train = None
        X_embedded_val = None

        for feature_index, feature_type in enumerate(feature_types):
            log_name = feature_type + "_" + layer_names[layer_index]

            datasets = filter_functions[feature_index](train_datasets[0])
            activation_vecs_all = compute_tsne_data(datasets, layer_hook, "train_" + log_name, class_colors, model)
            colors = compute_tsne_colors(datasets, layer_hook, class_colors, model)
            activation_vecs_all, colors = sort_activations_colors(activation_vecs_all, colors)
            if X_embedded_train is None:
                X_embedded_train, perplexity_values = compute_tsne_embedding(activation_vecs_all)
            for embedding_index, X_embedding in enumerate(X_embedded_train):
                plot_resulting_tsne(colors, X_embedding, save_name="train_" + log_name, perplexity=perplexity_values[embedding_index])


            datasets = filter_functions[feature_index](val_datasets[0])
            activation_vecs_all = compute_tsne_data(datasets, layer_hook, log_name, class_colors, model)
            colors_val = compute_tsne_colors(datasets, layer_hook, class_colors, model)
            activation_vecs_all, colors = sort_activations_colors(activation_vecs_all, colors)
            if X_embedded_val is None:
                X_embedded_val, perplexity_values = compute_tsne_embedding(activation_vecs_all)
            for embedding_index, X_embedding in enumerate(X_embedded_val):
                plot_resulting_tsne(colors_val, X_embedding, save_name=log_name, perplexity=perplexity_values[embedding_index])

            #run_tsne_step(datasets, layer_hook, log_name, class_colors, model)


def sort_activations_colors(activation_vecs_all, colors):
    activation_vecs_mean = torch.mean(activation_vecs_all, dim=1).numpy()
    activation_vecs_all = activation_vecs_all.numpy()
    sorting_indice = activation_vecs_mean.argsort()
    #print("sorting_indice shape: {}".format(sorting_indice.shape))
    activation_vecs_all = activation_vecs_all[sorting_indice]
    colors = np.array(colors)
    #print("colors shape: {}".format(colors.shape))
    colors = colors[sorting_indice]

    return torch.from_numpy(activation_vecs_all), colors



def compute_tsne_data(datasets, layer_hook, log_name, class_colors, model):
    val_loaders = datasets_to_dataloaders(datasets)

    activation_vecs_all, colors, targets_out = build_activation_vecs(model, layer_hook, val_loaders, colors=class_colors)
    np.save("data/activations/" + log_name + "_activation_vectors" + ".npy", activation_vecs_all)
    np.save("data/activations/" + log_name + "_targets" + ".npy", targets_out)

    return activation_vecs_all


def compute_tsne_colors(datasets, layer_hook, class_colors, model):
    val_loaders = datasets_to_dataloaders(datasets)

    _, colors, _ = build_activation_vecs(model, layer_hook, val_loaders, colors=class_colors, compute_activation_vecs=False)

    return colors


def plot_resulting_tsne(colors, X_embedded, save_name, perplexity):
        color_new = []
        for k in range(X_embedded.shape[0]):
            color_new.append(colors[k])
        color = color_new

        x, y = zip(*X_embedded)
        fig = plt.figure(figsize=(10, 10))

        plt.scatter(x, y, c=color)#, s=4.0)#[(X_softmaxed[i][0], 0.0, 0.0) for i in range(len(X_softmaxed))])
        plt.axis("off")
        plt.show()
        plt.savefig("save_figs/tsne/" + save_name + "_" + str(perplexity) + ".png")
        plt.close()


def build_activation_vecs(model, layer_hook, data_loaders, colors, compute_activation_vecs=True):
    colors_out = []
    targets_out = []
    activation_vecs_out = []
    for i, data_loader in enumerate(data_loaders):
        if compute_activation_vecs:
            activation_vecs_all = []
            for x, _ in data_loader:
                activations = access_activations_forward_hook([x.to(DEVICE)], model.forward, layer_hook)
                if len(activations.shape) == 4:
                    activation_vecs = torch.mean(activations, dim=[2, 3])
                else:
                    activation_vecs = activations
                activation_vecs_all.append(activation_vecs)

            activation_vecs_all = torch.cat(activation_vecs_all, dim=0)
            activation_vecs_out.append(activation_vecs_all)

        #for _ in range(len(activation_vecs_all)):
        for _ in range(len(data_loader.dataset)):
            colors_out.append(colors[i])
            targets_out.append(torch.tensor(i))

    if compute_activation_vecs:
        activation_vecs_out = torch.cat(activation_vecs_out, dim=0)
    return activation_vecs_out, colors_out, torch.stack(targets_out, dim=0)






def get_lightning_models_from_folder(folder_dir, model_class=classification_net_cnn_image_lightning):
    ckpt_paths = get_lightning_ckpt_file_paths_recursively(folder_dir)
    models = []

    for i, ckpt_path in enumerate(ckpt_paths):
        model = model_class.load_from_checkpoint(ckpt_path).eval().to(DEVICE)
        #model = classification_net_cnn_image_lightning.load_from_checkpoint(ckpt_path).eval().to(DEVICE)
        models.append(model)

    return models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":


    log_dirs = ["logs/full_labeled_data_10_cross_validation", "logs/efficentnetb5", "logs/resnet50", "logs/resnet34", "logs/vitb16"]
    model_classes = [classification_net_cnn_image_lightning, classification_net_cnn_image_lightning_EfficentNetB5, classification_net_cnn_image_lightning_ResNet50, classification_net_cnn_image_lightning_ResNet34, classification_net_cnn_image_lightning_ViT_B]
    model_names = ["Custom CNN", "EfficientNet B5", "ResNet50", "ResNet34", "ViTB/16"]
    normalize_smooth_spec_individually = [False, True, True, True, True]

    log_dirs = ["logs/vitb16", "logs/full_labeled_data_10_cross_validation_25_8_data", "logs/full_labeled_data_10_cross_validation_double_channels", "logs/full_labeled_data_10_cross_validation_no_reg_aug_label_smoothing_500_epochs", "logs/full_labeled_data_10_cross_validation_no_db_limit"]
    model_classes = [classification_net_cnn_image_lightning_ViT_B, classification_net_cnn_image_lightning, classification_net_cnn_image_lightning_double_channels, classification_net_cnn_image_lightning, classification_net_cnn_image_lightning]
    model_names = ["ViTB/16", "custom cnn 25x8", "custom cnn x2 channels", "custom cnn no aug reg", "custom cnn no db clip"]
    normalize_smooth_spec_individually = [True, False, False, False, False]
    resize_sizes = [(160, 160), (25, 8), None, None, None]
    val_crop_sizes = [150, 170, 170, 170, 170]


    log_dirs = ["logs/full_labeled_data_10_cross_validation_no_db_limit"]
    model_classes = [classification_net_cnn_image_lightning]
    model_names = ["Custom CNN"]
    normalize_smooth_spec_individually = [False]
    resize_sizes = [None]
    val_crop_sizes = [170]



    for i, log_dir in enumerate(log_dirs):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(model_names[i])
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        #model_class = model_classes[i]
        #model = model_class()
        #num_params = count_parameters(model)
        #print("{} parameters: {}".format(model_names[i], num_params))

        print_lightning_models_stats(log_dir, model_class=model_classes[i], normalize_smooth_spec_individually=normalize_smooth_spec_individually[i],
                                        resize_size=resize_sizes[i], val_crop_size=val_crop_sizes[i])

        #print_lightning_models_stats_confidence(log_dir, model_class=model_classes[i], normalize_smooth_spec_individually=normalize_smooth_spec_individually[i],
        #                                        model_name_for_savefig=model_names[i], use_temperature_scaling=False, run_individual_models=True)

    #logs_dir = "logs/full_labeled_data_10_cross_validation"
    #print_lightning_models_attribution(logs_dir)
    #print_lightning_models_stats_confidence(logs_dir, model_class=classification_net_cnn_image_lightning, normalize_smooth_spec_individually=False,
    #                                        model_name_for_savefig="custom_cnn")#,
                                 #resize_size=(25, 8))
                                #resize_size=(160, 160), val_crop_size=150)
    #example_plot_attribution_samples()


    #def run_tsne_gen_types(model, layer_hooks, layer_names, categories=[1, 2, 3, 4, 5]):

    #run_tsne_models()
    #logs_dir = "logs/efficentnetb5"
    #plot_lightning_models_accuracy_over_val_size(logs_dir, model_class=classification_net_cnn_image_lightning_EfficentNetB5,
    #                                              val_crop_sizes=[130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190])


    #model = torch.load("trained_models/cnn/resnet_model_conv_new_data_07.pt").to(DEVICE).eval()
    #model = torch.load("trained_models/cnn/mel_specs/resnet.pt").to(DEVICE).eval()

    #hook_model_boost_class(3, model.layers.layer1, model)
    #print_accuracy(merge_class_1_and_5=True, model=model, classes_to_use=[1, 2, 3, 4, 5])

    #print_accuracy_images(merge_class_1_and_5=False, model=model, classes_to_use=[1, 2, 3, 4, 5])

    #for i in range(5):
    #    optim_vis_high_res_model(model, layer=model.layers.layer0, class_number=i+1, plot_image=False)

    #train_datasets, val_datasets = get_datasets_cross_val(val_crop_size=150)
    #for x, target in val_datasets[0]:
    #    break
    #print(x.shape)

    #generate_adversarial_examples_dataset()


    #model = classification_net_cnn_image_lightning.load_from_checkpoint("logs/full_labeled_data_10_cross_validation/lightning_logs/version_0/checkpoints/epoch=139-step=12880.ckpt").to(DEVICE).eval()
    #model = get_lightning_models_from_folder("logs/full_labeled_data_10_cross_validation/")[0]
    #backward_fix_hook_model(model)
    #plot_channel_vis(model, layers=[model.layers.layer2, model.layers.layer1, model.layers.layer0, model.layers.linear_last], layer_names=["layer2/", "layer1/", "layer0/", "linear_last/"], layers_relu=[model.layers.layer2.resnet_blocks[1].activation_01, model.layers.layer1.resnet_blocks[1].activation_01, model.layers.layer0.resnet_blocks[1].activation_01, None])
    #plot_channel_vis(model, layers=[model.layers.linear_last], layer_names=["linear_last/"])
    #plot_channel_vis(model, layers=[model.layers.resnet_block_01], layer_names=["resnet_block_01/"], layers_relu=[model.layers.resnet_block_01.activation_01])



    #plot_close_image_samples(model, layer_hook=model.layers.layer2)
    #plot_highest_activating_channels_from_data(model, layer_hook=model.layers.layer2, n_channels_to_plot=4)

    #plot_highest_attributing_channels_from_data(model, layer_hook=model.layers.layer2, n_channels_to_plot=4, wanted_target_class_number=1)


    #test_baselines()


    #vary_time_feature()
    #plot_image()
    #for i in range(4):
    #    optim_vis(model, class_number=i+1)



    #for class_idx in range(5):
    #    print("class " + str(class_idx))
    #    #print_attribution_target_class(class_idx, absolute_value=False)
    #    plot_attribution_overlay(class_idx)
