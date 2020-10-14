from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import os

batch_size = 32
num_classes = 10
epochs = 10
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Simple dense with different hidden layers
model_0 = Sequential(
	  [
	  	  Flatten(),
	  	  Dense(num_classes),
	  	  Activation('softmax'),
	  ]
)

model_1 = Sequential(
	  [
	  	  Flatten(),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(num_classes),
	  	  Activation('softmax'),
	  ]
)

model_2 = Sequential(
	  [
	  	  Flatten(),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(num_classes),
	  	  Activation('softmax'),
	  ]
)

model_3 = Sequential(
	  [
	  	  Flatten(),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(num_classes),
	  	  Activation('softmax'),
	  ]
)

model_4 = Sequential(
	  [
	  	  Flatten(),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(512),
	  	  Activation('relu'),
	  	  Dropout(0.5),
	  	  Dense(num_classes),
	  	  Activation('softmax'),
	  ]
)

# CNN
model_cnn = Sequential(
	  [
		Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]),
		Activation('relu'),
		Conv2D(32, (3, 3)),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Dropout(0.25),

		Conv2D(64, (3, 3), padding='same'),
		Activation('relu'),
		Conv2D(64, (3, 3)),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Dropout(0.25),

		Flatten(),
		Dense(512),
		Activation('relu'),
		Dropout(0.5),
		Dense(num_classes),
		Activation('softmax'),
	  ]
)


# Train the model using RMSprop
model_0.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history0 = model_0.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model_1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history1 = model_1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model_2.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history2 = model_2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model_3.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history3 = model_3.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model_4.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history4 = model_4.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model_cnn.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history_cnn = model_cnn.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

plt.plot(history0.history['val_accuracy'], color = 'r')
plt.plot(history1.history['val_accuracy'], color = 'b')
plt.plot(history2.history['val_accuracy'], color = 'y')
plt.plot(history3.history['val_accuracy'], color = 'g')
plt.plot(history4.history['val_accuracy'], color = 'brown')
plt.plot(history_cnn.history['val_accuracy'], color = 'purple')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Simple dense - 0 hiddlen', 'Simple dense - 1 hiddlen',
			'Simple dense - 2 hiddlen', 'Simple dense - 3 hiddlen',
			'Simple dense - 4 hiddlen', 'CNN'], loc = 'upper left')
plt.show()

