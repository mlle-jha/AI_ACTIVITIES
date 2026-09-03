import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, classification_report

#DATA ACQUISITION
#Load the cat breed dataset
dataset_path = "AI_ACTIVITIES/cat_breeds"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

#Print the class names
print("Cat breeds:", train_ds.class_names)
#Print the number of classes
print("Number of cat breeds:", len(train_ds.class_names))

#EXPLORATORY DATA ANALYSIS
#Display dataset information

print("Image size: 180 x 180")
print("Number of cat breeds:", len(train_ds.class_names))
print("Cat breeds:", train_ds.class_names)

#Display the first 9 images
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(
            f"Label: {train_ds.class_names[labels[i]]}"
        )
        plt.axis("off")
plt.show()

#DATA PREPROCESSING
#Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

#MODELING
#Number of cat breeds
num_classes = len(train_ds.class_names)

#Load the MobileNetV2 model
base_model = keras.applications.MobileNetV2(
    input_shape=(180, 180, 3),
    include_top=False,
    weights="imagenet"
)

#Freeze the pretrained model
base_model.trainable = False

#Create the neural network
model = keras.Sequential([
    layers.Input(shape=(180, 180, 3)),
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(
        num_classes,
        activation="softmax"
    )
])

#Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Display model information
model.summary()

#MODEL TRAINING
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=validation_ds
)


#DISPLAY ACCURACY PER EPOCH
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy per epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#DISPLAY LOSS PER EPOCH
plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#PREDICTIONS
#Make predictions using the validation dataset
y_true = []
y_pred = []

#DISPLAY PREDICTIONS
#Display the first 9 predictions
plt.figure(figsize=(10, 10))
for images, labels in validation_ds.take(1):
    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(
            images[i].numpy().astype("uint8")
        )
        plt.title(
            f"Predicted: {train_ds.class_names[predicted_labels[i]]}\n"
            f"Actual: {train_ds.class_names[labels[i]]}"
        )
        plt.axis("off")
plt.tight_layout()
plt.show()

for images, labels in validation_ds:
    predictions = model.predict(
        images,
        verbose=0
    )
    predicted_labels = np.argmax(
        predictions,
        axis=1
    )
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

#MODEL EVALUATION
#Evaluate the model

loss, accuracy = model.evaluate(
    validation_ds
)

print("Validation Loss:", loss)
print(
    "Validation Accuracy:",
    accuracy
)
print(
    "Validation Accuracy:",
    accuracy * 100,
    "%"
)

#Display classification report
print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=train_ds.class_names
    )
)

#CONFUSION MATRIX
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(12, 10))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
plt.xticks(
    range(num_classes),
    train_ds.class_names,
    rotation=90
)
plt.yticks(
    range(num_classes),
    train_ds.class_names
)
plt.show()

#DEPLOYMENT
#Save the trained model
model.save("AI_ACTIVITIES/cat_breed_model.keras")
print("Model saved successfully!")

#Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#Save the TensorFlow Lite model
with open("cat_breed_model.tflite", "wb") as file:
    file.write(tflite_model)
print("TensorFlow Lite model saved successfully!")

#MONITORING
print("\nModel monitoring:")
print("- Check validation accuracy")
print("- Check validation loss")
print("- Check for overfitting")
print("- Check which cat breeds are commonly misclassified")
print("- Retrain the model when more images are available")