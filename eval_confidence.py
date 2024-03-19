
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
    data_dir="./data/automatic_detection_manual_classification",
    spectrogram_specs=spectrogram_specs,
    categories=[0], ignore_categories=[],
)
data.setup("test")
train_dl = data.train_dataloader()
val_dl = data.val_dataloader()
test_dl = data.test_dataloader()

features, targets = load_all_data(train_dl)
features = features.to("cuda")
targets = targets.to("cuda")
targets = torch.argmax(targets, dim=1)

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

model = torch.load("trained_models/cnn/resnet_model_conv_new_data_07.pt").to("cuda").eval()

with torch.no_grad():
    preds = model(features.to("cuda"))
preds_argmax = torch.argmax(preds, dim=1)


confidence_correct = []
confidence_false = []

for i, pred in enumerate(preds_argmax):
    target = targets[i]
    if pred == target:
        confidence_correct.append(preds[i][pred])
    else:
        confidence_false.append(preds[i][pred])

confidence_correct = torch.stack(confidence_correct)
confidence_false = torch.stack(confidence_false)

print("mean confidence correct: {}".format(torch.mean(confidence_correct)))
print("std confidence correct: {}".format(torch.std(confidence_correct)))

print("mean confidence false: {}".format(torch.mean(confidence_false)))
print("std confidence false: {}".format(torch.std(confidence_false)))


