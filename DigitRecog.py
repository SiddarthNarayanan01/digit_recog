import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Tuple unpacking for data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # Prints (60000, 28, 28)

single_image = x_train[0]

# Matplotlib has an image plotter for 2D arrays
plt.imshow(single_image, cmap='Greys')
plt.show()


# One hot encoding the y labels
y_cat_test = to_categorical(y_test, 10)  # You can state number of classes
y_cat_train = to_categorical(y_train, 10)  # The shapes are now (60000, 10)

# Normalizing the data
# The images will be between 0 and 1
x_train = x_train/255
x_test = x_test/255

# The shape of the data is (60000, 28, 28)
# It includes the batch dimension, width, height. We need to add the color dimension
x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train[1])

# Creating the model

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(28, 28, 1),
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Flattening to a single array
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# Output layer
model.add(Dense(10, activation='softmax'))  # Softmax because it is a multi-class problem

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping('val_loss', patience=1, verbose=0)

model.fit(x_train, y_cat_train, batch_size=500, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
metrics[['loss', 'val_loss']].plot()
plt.show()

metrics[['accuracy', 'val_accuracy']].plot()
plt.show()

predictions = model.predict_classes(x_test)
class_report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

sns.heatmap(conf_matrix, cmap='Blues')
plt.show()

print('\n\n\n')
print(class_report, conf_matrix)

model.save('DigitRecognition.h5', include_optimizer=True)
