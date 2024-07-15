from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# #intialising the learning rate , number of epochs and batch size
# INIT_LR = 1e-4
# EPOCHS = 20
# BS = 32

# DIRECTORY = r"F:\Projects\Face Mask Detection\dataset"
# CATEGORIES = ["with_mask","without_mask"]

# # Grab the list of images in our dataset directory, and then initialize
# # the list of data (i.e images) and class images 
# print("[INFO] loading images....")

# data = []
# labels = []

# for category in CATEGORIES:
#     path = os.path.join(DIRECTORY,category)
#     for img in os.listdir(path):
#         img_path = os.path.join(path,img)
#         image = load_img(img_path,target_size=(224,224))
#         image = img_to_array(image)
#         image = preprocess_input(image)

#         data.append(image)
#         labels.append(category)


# data = np.array(data, dtype="float32")
# labels = np.array(labels)


# # perform one-hot encoding 
# lb = LabelBinarizer()             #LabelBinarizer is first used to convert the categorical string labels ("with_mask" and "without_mask")
# labels = lb.fit_transform(labels) # into binary format

# labels = to_categorical(labels)   # to_categorical is essential for neural networks, which often require the labels to be in a one-hot encoded format 
#                                   #to calculate loss and optimize during training.


# (trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=.25,stratify=labels,random_state=7)

# # constructing trainig image generator for data augmentation 

# aug = ImageDataGenerator(
#     rotation_range = 20,
#     zoom_range = 0.15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.15,
#     horizontal_flip=True,
#     fill_mode="nearest")

# # load the MobileNet network, ensuring the head FC layer sets are left off

# baseModel = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

# # contruct the head of the model that will be placed on the top of the base model

# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7,7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128,activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2,activation="softmax")(headModel)

# # place the head FC model on top of the base model (this will become the actual model we will train)
# model = Model(inputs=baseModel.input,outputs=headModel)

# # loop over all layers in the base model and freeze them so they will not be updated during the first trainig process

# for layer in baseModel.layers:
#     layer.trainable = False

# # compile our model 
# print("[INFO] conmpiling model....")
# opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
# model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

# #train the head of the network
# print("[INFO] training head")
# H = model.fit(
#     aug.flow(trainX,trainY,batch_size=BS),
#     steps_per_epoch = len(trainX) // BS,
#     validation_data= (testX,testY),
#     validation_steps = len(testX) // BS,
#     epochs = EPOCHS
# )

# # make predictions on testing set
# print("[INFO] evaluating network.. ")
# predIdxs = model.predict(testX,batch_size=BS)

# # Convert predictions from probabilities to class labels
# predIdxs = np.argmax(predIdxs, axis=1)
# testY = np.argmax(testY, axis=1)

# # show a nicely formatted classification report
# print(classification_report(testY.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))

# # serialize the model to disk
# print("[INFO] saving mask detector model...")
# model.save("mask_detector.model", save_format="h5")

# # plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# Initialize constants
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Dataset directory and categories
DIRECTORY = r"F:\Projects\Face Mask Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Loading images and labels
print("[INFO] loading images....")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

# Convert lists to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Splitting data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.25, stratify=labels, random_state=7)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Loading MobileNetV2 base model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Constructing the head of the model to be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Combining base model and head model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freezing layers in the base model
for layer in baseModel.layers:
    layer.trainable = False

# Compiling the model
print("[INFO] compiling model....")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Training the model
print("[INFO] training head")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)
print("[INFO] saving mask detector model...")
model.save("mask_detector.h5")

# Evaluating the model on the test set
print("[INFO] evaluating network.. ")
predIdxs = model.predict(testX, batch_size=BS)

# Converting predictions from probabilities to class labels
predIdxs = np.argmax(predIdxs, axis=1)
testY = np.argmax(testY, axis=1)  # Ensure testY is also in one-hot encoded format

# Printing classification report
print(classification_report(testY, predIdxs, target_names=lb.classes_))

# # Saving the model to disk
# print("[INFO] saving mask detector model...")
# model.save("mask_detector.model", save_format="h5")

# Plotting the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
