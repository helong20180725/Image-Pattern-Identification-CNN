from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import keras
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# dimensions of our images. 1024,768 => 720, 540 => 120, 80
img_width, img_height = 120, 80

# 99 => 595 => 260 => 95
train_data_dir = 'puredata/train/'
# 27 => 159 =>114 => 29
test_data_dir = 'puredata/test/'
# 7 => 38 => 9
validation_data_dir = 'puredata/validation/'
batch_size = 9
epochs = 150

# preprocess: load data from images

train_datagen = ImageDataGenerator(rescale=1./255)
# training data augmentation
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=10,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        color_mode="grayscale",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# Test data don't need to be augmented. rescale=1./255
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        color_mode="grayscale",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        directory=validation_data_dir,
        color_mode="grayscale",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
# model: building model

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

# Enough CNN layers are necessary. 
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# In this problem, we don't need too much layers.
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))


model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# keras.optimizers.Adadelta()， keras.optimizers.Adam()
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
H = model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=epochs
                        )

score = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
print(score)

# model.save_weights('result/homework4_6.h5')

# plot the training loss and accuracy

matplotlib.use("Agg")
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy\nFinal scores: Loss " + str(score[0]) + ", Accuracy " + str(score[1]))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("result/plot_best_4c1n_128")
