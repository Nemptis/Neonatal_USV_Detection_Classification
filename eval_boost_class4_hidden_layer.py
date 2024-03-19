
import torch
from mouse_dataset import mouse_dataset, mouse_data_module, spectrogram_specs
from xai import load_all_data, access_activations_forward_hook, hook_model_boost_class
import matplotlib.pyplot as plt

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
data.setup("test")
train_dl = data.train_dataloader()
val_dl = data.val_dataloader()
test_dl = data.test_dataloader()

features, targets = load_all_data(val_dl)
features = features.to("cuda")
targets = targets.to("cuda")
targets = torch.argmax(targets, dim=1)

target_class = 3

model = torch.load("trained_models/cnn/focal_loss/resnet_model_conv_new_split_" + "1.4" + ".pt").to("cuda").eval()

#hook_model_boost_class(3, model.layers.layer1, model)

accuracies = []
accuracies_class_4 = []

with torch.no_grad():
    preds_logits = access_activations_forward_hook([features], model, model.layers.linear_last).to("cuda")

bias_class_4 = [i*1000 for i in range(30)]

for i, bias in enumerate(bias_class_4):
    model = torch.load("trained_models/cnn/focal_loss/resnet_model_conv_new_split_" + "1.4" + ".pt").to("cuda").eval()

    hook_model_boost_class(3, model.layers.layer1, model, bias, features, targets)

    with torch.no_grad():
        preds_logits = access_activations_forward_hook([features], model, model.layers.linear_last).to("cuda")


    preds = torch.clone(preds_logits)
    #preds[:, 3] += bias
    preds = torch.nn.functional.softmax(preds)

    preds = torch.argmax(preds, dim=1)
    #print("preds shape: {}".format(preds.shape))
    #print("targets shape: {}".format(targets.shape))
    accuracy_class = torch.sum((preds == target_class) * (targets == target_class)) / torch.sum(targets == target_class)
    accuracy = torch.sum(preds == targets) / len(preds)

    accuracies.append(accuracy.cpu())
    accuracies_class_4.append(accuracy_class.cpu())

    print(accuracy)

fig = plt.figure(figsize=(20, 10))
plt.plot(bias_class_4, accuracies, '--bo', bias_class_4, accuracies_class_4, '--go')
plt.savefig("save_figs/fig.png")





