
#%% load the background
from __future__ import print_function, division
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import torch.nn as nn

#%% define the dataset
dataset2use = "val"
if dataset2use == 'val':
    data_dir = "/home/cw9/sds_hd/sd18a006/Marlen/datasets/stainNormalization/patchCamelyon"
elif dataset2use == "test":
    data_dir = "/home/cw9/sds_hd/sd18a006/Marlen/datasets/stainNormalization/tumorLymphnode/"

# define the function to get the data
def get_datatransform(inputSize, data_dir):

    data_transforms = {
        dataset2use: transforms.Compose([
            transforms.Resize([inputSize, inputSize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [dataset2use]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=False, num_workers=4)
                   for x in [dataset2use]}

    return(data_transforms, image_datasets, dataloaders)

#%% prepare the transformations and the dataset
data_transforms , image_datasets, dataloaders= get_datatransform(259, data_dir)

class_names = dataloaders[dataset2use].dataset.classes
nb_classes = len(class_names)
confusion_matrix = torch.zeros(nb_classes, nb_classes)

#%% visualize the input data (to look if evey class is evenly)
class_names =  ['normal', 'tumor']

df = pd.DataFrame(dataloaders[dataset2use].dataset.samples)
df.columns = ['file', 'class_nr']

df.class_nr = np.array(df.class_nr)

class_labels = ['NaN' for x in range(df.shape[0])]
for i in range(0,df.shape[0]):
    class_labels[i] = class_names[df.class_nr[int(i)]]
df = df.assign(class_labels = class_labels)
sns.set_palette("Set1", n_colors = 12)
sns.countplot(df.class_labels)
plt.xlabel('Pattern')
plt.ylabel('Count [n]')
plt.savefig('DataBase_' + dataset2use + '.jpg')
plt.show()
plt.close()

#%% define the models to load
model_list = ['/home/cw9/sds_hd/sd18a006/Marlen/datasets/stainNormalization/stainGAN_camelyon16/trainedModels' + '/model_ResNet152.pt',
            '/home/cw9/sds_hd/sd18a006/Marlen/datasets/stainNormalization/patchCamelyon/trainedModels' +'/model_ResNet152.pt'
]
model_names = ['Norm', "non-Norm"]
accurancy = list(range(0, len(model_list)))
kappa = list(range(0, len(model_list)))
loss = list(range(0, len(model_list)))

#%% iterate over the models
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n = 0
df_values = pd.DataFrame(list(range(0,len(dataloaders[dataset2use].sampler.data_source.imgs))))

for imodel in model_list:

    #%% prepare the dataset
    inputSize = 224
    data_transforms, image_datasets, dataloaders = get_datatransform(inputSize, data_dir)

    #%% apply model on test data set (and get a confusion matrix)
    model_ft = torch.load(imodel)
    model_ft.eval()
    vector_prd = []
    vector_exp = []
    #del(outputs_)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders[dataset2use]):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            if i == 0:
                outputs_matrix = outputs
            else:
                outputs_matrix = torch.cat((outputs_matrix, outputs), 0)

            vector_prd = vector_prd + preds.view(-1).cpu().tolist()
            #vector_prd = vector_prd + [0]
            vector_exp = vector_exp + classes.view(-1).cpu().tolist()

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for x, y in zip(vector_exp, vector_prd):
        confusion_matrix[y, x] += 1

    loss_function = nn.CrossEntropyLoss()
    loss_value = loss_function(outputs_matrix.to('cpu'), torch.tensor(vector_exp))
    print(confusion_matrix)

    #%% calcualte the comparison values
    kappa[n] = cohen_kappa_score(vector_prd, vector_exp)
    accurancy[n] = accuracy_score(vector_prd, vector_exp)
    loss[n] = loss_value.tolist()
    print(kappa[n])
    print(accurancy[n])

    #%% plot a confusion matrix
    matrix2plot = confusion_matrix.numpy()
    matrix2plot = matrix2plot.astype(int)
    #matrix2plot = normalize(matrix2plot, axis =1, norm = 'l1')
    #matrix2plot = matrix2plot.round(decimals=2)
    # create seabvorn heatmap with required labels
    ax = sns.heatmap(matrix2plot,
                     annot = True, linewidths=5, annot_kws={"size": 10},
                     xticklabels=class_names, yticklabels=class_names,
                     cmap = "Blues")
    plt.xlabel('Ground Truth')
    plt.ylabel('Pattern predicted by model ' + model_names[n])
    plt.savefig('ConfMat_' + model_names[n] + '.jpg')
    plt.show()
    plt.close()
    n +=1
