import tensorflow as tf
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt


# Parametry i baza
num_classes = 10
input_shape = (28, 28, 1)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Obraz wejściowy do warstwy liniowej
x_train = x_train.astype("float32") /255
x_test = x_test.astype("float32") /255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train Kształt: ", x_train.shape)
print("Próbki Train: ", x_train.shape[0])
print("Próbki Test: ", x_test.shape[0])

# Obraz wejściowy do warstwy spłaszczenia
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Tworzenie modelu
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

# Kompilowanie modelu
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Trenowanie modelu
batch_size = 128
epochs = 15
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Ocena modelu
score = model.evaluate(x_test, y_test, verbose=0)
print("Test strat: ", score[0])
print("Test dokładności")

# Prognozowanie
prediction = model.predict(x_test)
img = plt.imshow(1-x_test[1])
img.set_cmap("gray")
plt.axis("off")
plt.show()