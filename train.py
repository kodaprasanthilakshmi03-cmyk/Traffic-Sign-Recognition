import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Configuration
DATASET_PATH = "Dataset"
LABEL_FILE = "labels.csv"
MODEL_PATH = "model/model.h5"
IMAGE_DIMENSIONS = (32, 32, 3)
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 10
# Step 1: Load dataset
print("Loading dataset...")
images, classNo = [], []
class_folders = os.listdir(DATASET_PATH)
num_classes = len(class_folders)
for class_id in range(num_classes):
    image_files = os.listdir(os.path.join(DATASET_PATH, str(class_id)))
    for img_file in image_files:
        img_path = os.path.join(DATASET_PATH, str(class_id), img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            classNo.append(class_id)
    print(f"Loaded class {class_id} with {len(image_files)} images")
images = np.array(images)
classNo = np.array(classNo)
# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_RATIO)
# Step 3: Preprocessing
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img): return cv2.equalizeHist(img)
def preprocessing(img): return equalize(grayscale(img)) / 255
X_train = np.array(list(map(preprocessing, X_train))).reshape(-1, 32, 32, 1)
X_val = np.array(list(map(preprocessing, X_val))).reshape(-1, 32, 32, 1)
X_test = np.array(list(map(preprocessing, X_test))).reshape(-1, 32, 32, 1)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)
# Step 4: Data Augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
# Step 5: Model
def build_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model = build_model()
model.summary()
# Step 6: Train
history = model.fit(dataGen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    shuffle=True)

# Step 7: Evaluate
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
# Step 8: Save model
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
# Step 9: Plot and save metrics
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("model/loss_curve.png")
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("model/accuracy_curve.png")
print("Training complete.")
