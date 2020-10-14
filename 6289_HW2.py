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


# For question 1
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


# For question 2
model_cnn_sig.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
history_cnn_sig = model_cnn_sig.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

plt.plot(history_cnn_relu_drop_noaug.history['val_accuracy'], color = 'r')
plt.plot(history_cnn_sig.history['val_accuracy'], color = 'b')
plt.title('Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['CNN using relu', 'CNN using sigmoid'], loc = 'upper left')
plt.show()


# For question 3
# real-time data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

model_cnn_relu_drop.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history_cnn_relu_drop_noaug_2 = model_cnn_relu_drop.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=(x_test, y_test),
          shuffle=True)

history_cnn_relu_drop_aug = model_cnn_relu_drop.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=20,
                        validation_data=(x_test, y_test),
                        workers=4)

model_cnn_relu_nodrop.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history_cnn_relu_nodrop_noaug = model_cnn_relu_nodrop.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=(x_test, y_test),
          shuffle=True)

history_cnn_relu_nodrop_aug = model_cnn_relu_nodrop.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=20,
                        validation_data=(x_test, y_test),
                        workers=4)

# Traing accuracy
plt.plot(history_cnn_relu_drop_noaug_2.history['accuracy'], color = 'r')
plt.plot(history_cnn_relu_drop_aug.history['accuracy'], color = 'b')
plt.plot(history_cnn_relu_nodrop_noaug.history['accuracy'], color = 'y')
plt.plot(history_cnn_relu_nodrop_aug.history['accuracy'], color = 'g')
plt.title('Training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['dropout & no data augmentation', 'dropout & data augmentation',
			'no dropout & no data augmentation', 'no dropout & data augmentation'], loc = 'upper left')
plt.show()

# Testing accuracy
plt.plot(history_cnn_relu_drop_noaug_2.history['val_accuracy'], color = 'r')
plt.plot(history_cnn_relu_drop_aug.history['val_accuracy'], color = 'b')
plt.plot(history_cnn_relu_nodrop_noaug.history['val_accuracy'], color = 'y')
plt.plot(history_cnn_relu_nodrop_aug.history['val_accuracy'], color = 'g')
plt.title('Testing accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['dropout & no data augmentation', 'dropout & data augmentation',
			'no dropout & no data augmentation', 'no dropout & data augmentation'], loc = 'upper left')
plt.show()