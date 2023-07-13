# To check if GPU is active
from tensorflow.python.client import device_lib

# Load Data
import os
import cv2
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt

# Model Training
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

print(device_lib.list_local_devices())

train_dir = 'asl_alphabet_train'
test_dir = 'asl_alphabet_test'


def get_data(data_dir):
    images = []
    labels = []

    dir_list = os.listdir(data_dir)
    for i in range(len(dir_list)):
        print("Obtaining images of", dir_list[i], "...")
        for image in os.listdir(data_dir + "/" + dir_list[i]):
            img = cv2.imread(data_dir + '/' + dir_list[i] + '/' + image)
            img = cv2.resize(img, (32, 32))
            images.append(img)
            labels.append(i)

    return images, labels


X, y = get_data(train_dir)

print(len(X), len(y))

classes = ['A', 'B', 'C', 'del']


def plot_sample_images():
    figure = plt.figure()
    plt.figure(figsize=(16, 5))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        path = train_dir + "/{0}/{0}1.jpg".format(classes[i])
        img = plt.imread(path)
        plt.imshow(img)
        plt.xlabel(classes[i])


plot_sample_images()


def preprocess_data(X, y):
    np_X = np.array(X)
    # This normalization step scales the pixel values between 0 and 1, which can help improve the model's training process
    normalised_X = np_X.astype('float32') / 255.0
    # Each label is converted to a binary vector where only one element is 1, representing the corresponding class
    label_encoded_y = utils.to_categorical(y)
    # 10% of the data is allocated for testing, while the remaining 90% is used for training
    x_train, x_test, y_train, y_test = train_test_split(normalised_X, label_encoded_y, test_size=0.1)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = preprocess_data(X, y)

print("Training data:", x_train.shape)
print("Test data:", x_test.shape)

# Convolutional Neural Network (CNN) model architecture using the Keras Sequential API
# Model Configuration
classes = 4
batch = 32
epochs = 15 # Number of epochs to train the model
learning_rate = 0.001

# Initializes a sequential model, which is a linear stack of layers
model = Sequential()

# Adds a 2D convolutional layer with 64 filters, each having a 3x3 kernel size. Activation function used is ReLU.
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
# Helps reduce the spatial dimensions of the feature maps
model.add(MaxPooling2D(pool_size=(2, 2)))
# Normalize the activations of the previous layer
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
# Dropout helps prevent over fitting by randomly setting a fraction of input units to 0 during training,
# which reduces the interdependencies between neurons
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# This pattern of adding convolutional, pooling, batch normalization, and dropout layers is repeated twice more
# with different filter sizes (128 and 256) to capture more complex patterns and features in the data.

# This layer flattens the output from the previous layer into a 1D vector, preparing it to be fed into a fully
# connected (dense) layer
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
# The activation function used is softmax, which produces a probability distribution over the classes,
# indicating the model's predicted probabilities for each class.
model.add(Dense(classes, activation='softmax'))

# Model Compilation
# Adam optimizer with the provided learning rate, categorical cross-entropy loss function (suitable for
# multi-class classification), and accuracy as the evaluation metric
adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model's architecture, including the layer types, output shapes, and number of parameters
model.summary()

history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.2, shuffle = True, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


def plot_results(model):
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


plot_results(model)

model.save("model.h5")
loaded_model = load_model('model.h5')
cap = cv2.VideoCapture(0)
class_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'del'}

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Preprocess the frame
    img = cv2.resize(frame, (32, 32))  # Resize the frame to match the input size of the model
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add a batch dimension

    # Make predictions
    prediction = loaded_model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    # Display the prediction
    cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
