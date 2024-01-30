# Import packages
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# Command-line parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="dataset",
                help="path to input dataset")
ap.add_argument("-s", "--stats", type=str, default="model",
                help="path to output stats files")
ap.add_argument("-m", "--model", type=str, default="model\\covid-vgg.h5",
                help="output model file name")
args = vars(ap.parse_args())

# Training constant parameters
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# Create output dirs
print("[INFO] Creating output dirs...")
os.makedirs(os.path.abspath(args['stats']), exist_ok=True)
os.makedirs(os.path.join(os.path.abspath(args['stats']), 'augment'), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args['model'])), exist_ok=True)

# Enumerate files
print("[INFO] Loading images...")
data = []
labels = []

# loop over the image paths
for root, _, files in os.walk(args['dataset']):
    for file in files:
        path = os.path.join(root, file)
        img = load_img(path, target_size=(224, 224))

        data.append(img_to_array(img))
        labels.append(os.path.basename(root))

# Convert to NumPy array and rescale data
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, 
                                                stratify=labels, random_state=42)
trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

# Load the VGG16 network
baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Freeze base model
for layer in baseModel.layers:
    layer.trainable = False

# Build model
model = Model(inputs=baseModel.input, outputs=headModel)

# Print model summary
print(model.summary())

# Compile model
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] Training...")
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS, 
                  save_to_dir=os.path.join(args['stats'], 'augment')),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Run predictions
print("[INFO] Evaluating...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# Show classification report (accuracy, precision, recall, f1-score)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))



# Compute the confusion matrix
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
df_cm = pd.DataFrame(cm, lb.classes_, lb.classes_)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(args['stats'], "confusion-matrix.png"))


# Plot training loss and accuracy
n = EPOCHS
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, n), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), H.history["val_loss"], label="val_loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()
plt.savefig(os.path.join(args['stats'], "epoch-loss.png"))

plt.figure()
plt.plot(np.arange(0, n), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n), H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig(os.path.join(args['stats'], "epoch-accuracy.png"))

# Serialize the model to disk
print("[INFO] Saving COVID-19 detector model...")
model.save(args['model'])


import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

#define the transformations that we want to apply
transformations = {'train' : transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.Grayscale(),
                                                 transforms.ToTensor()])}

#define the path to our datasets
train_path = "D:\software\X-Ray LSTM\dataset\covid"
valid_path = "D:\software\X-Ray LSTM\dataset\normal"
dataset = {'train' : datasets.ImageFolder(train_path, transform = transformations['train']),
          'valid' : datasets.ImageFolder(valid_path, transform = transformations['train'])}

#load the dataset
data_loader = {'train' : DataLoader(dataset['train'], batch_size = 32, shuffle = True),
              'valid' : DataLoader(dataset['valid'], batch_size = 32, shuffle = False)}

images, labels = next(iter(data_loader['train']))

images.shape

#see if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers):
        super(LSTM, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        #initialize the hidden state and the cell state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        
        out, (hidden, cell) = self.lstm(x, (h0, c0))
        
        output_ = out[:, -1, :]
        
        output = self.fc(output_)
        
        return output
#instatiate the model
SEQ_LEN = 32
INPUT_DIM = 32
OUTPUT_DIM = 2
HIDDEN_DIM = 128
N_LAYERS = 2
model = LSTM(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, hidden_dim = HIDDEN_DIM, n_layers = N_LAYERS)
model = model.to(device)
print(model)

#define the optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#define the loss function
criterion = nn.CrossEntropyLoss()

#now we will TRAIN the model
def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for input_, label in data_loader[phase]:
                inputs = input_.squeeze(1).to(device)
                labels = label.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects.float() / len(dataset[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss.item(), epoch_acc.item()))
    return model

training = train_model(model = model, criterion = criterion, optimizer = optimizer, num_epochs = 10)