
import torch
from mouse_dataset import mouse_dataset, mouse_data_module, spectrogram_specs
from xai import load_all_data
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
raise RuntimeError

target_class = 3
focal_loss_gamma_values = [i*0.2 for i in range(20)]
class_weight_powers = [i*0.2 - 1.0 for i in range(16)]
class_weights_4 = [i*0.1 for i in range(10)] + [i for i in range(10)]
batch_size = [8, 16, 32, 64, 128, 256]
#print(class_weight_powers)
#raise RuntimeError
#for focal_loss_gamma_value in focal_loss_gamma_values:
#    focal_loss_gamma_value = str(focal_loss_gamma_value)[:3]
accuracies = []
accuracies_class_4 = []

for i, class_weight_power in enumerate(class_weights_4):
    #if class_weight_power < 0:
    #    class_weight_power = str(class_weight_power)[:4]
    #else:
    #    class_weight_power = str(class_weight_power)[:3]
    class_weight_power = str(round(class_weight_power, 1))
    #print(class_weight_power)
    #model = torch.load("trained_models/cnn/focal_loss/resnet_model_conv_new_split_" + focal_loss_gamma_value + ".pt").to("cuda").eval()
    model = torch.load("trained_models/cnn/class_weight_4/resnet_model_conv_new_split_" + class_weight_power + ".pt").to("cuda").eval()

    with torch.no_grad():
        preds = model(features.to("cuda"))
    preds = torch.argmax(preds, dim=1)
    #print("preds shape: {}".format(preds.shape))
    #print("targets shape: {}".format(targets.shape))
    accuracy_class = torch.sum((preds == target_class) * (targets == target_class)) / torch.sum(targets == target_class)
    accuracy = torch.sum(preds == targets) / len(preds)

    accuracies.append(accuracy.cpu())
    accuracies_class_4.append(accuracy_class.cpu())

    print(accuracy)

class_weights_4 = [str(class_weight_4) for class_weight_4 in class_weights_4]
fig = plt.figure(figsize=(20, 10))
plt.plot(class_weights_4, accuracies, '--bo', class_weights_4, accuracies_class_4, '--go')
plt.savefig("save_figs/fig.png")





