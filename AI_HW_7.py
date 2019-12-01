from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import rmsprop
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np
from keras.preprocessing import image

num_classes = 4
image_size = 150
nb_epoch = 10
batch_size = 20
nb_train_samples = 240
nb_validation_samples = 60

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_size, image_size)
else:
    input_shape = (image_size, image_size, 3)

train_data_dir = 'train_zip/train'
validation_data_dir = 'test_zip/test'

train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
    )

model = Sequential()

model.add(ZeroPadding2D((1,1), input_shape=input_shape))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
         optimizer = 'adam',
         metrics = ['accuracy'])


model.fit_generator(
                train_generator,
                steps_per_epoch=(nb_train_samples // batch_size),
                epochs=nb_epoch,
                validation_data=validation_generator,
                validation_steps=(nb_validation_samples // batch_size)
               )

model.save_weights('first_try.h5')

img_pred = image.load_img('banana.jpg', target_size=(150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)

result = model.predict(img_pred)
print(result)

if result[0][0] == 1:
    prediction = 'apple'
elif result[0][1] == 1:
    prediction = 'banana'
elif result[0][2] == 1:
    prediction = 'orange'
elif result[0][3] == 1:
    prediction = 'mixed'
    
print(prediction)