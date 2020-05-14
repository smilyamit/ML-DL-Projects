
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3), activation='relu'))


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
#classifier.add(Dropout(0.2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# Augmenting images for the Training set
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

# Augmenting images for the Test set
test_datagen = ImageDataGenerator(rescale=1. / 255)


# Creating the Training set
training_set = train_datagen.flow_from_directory('/Users/koro/Desktop/Ml Practise /DL/CNN/dogcat_Deploy/dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')


# Creating the Test set
test_set = test_datagen.flow_from_directory('/Users/koro/Desktop/Ml Practise /DL/CNN/dogcat_Deploy/dataset/test_set',
                                            target_size=(64, 64),
                                                           
                                            batch_size=32,         # batch size is a number of samples processed before the model is updated for first time than so on
                                            class_mode='binary')   # within 1 epoch at first only batches of 32 images are processed, than again next 32 (until 8000)


#Training the CNN on the Training set and evaluating it on the Test set
model = classifier.fit_generator(training_set,
                                  steps_per_epoch=250,  # 8000/32
                                  epochs=10,             # number of epochs is the number of complete passes through the training dataset
                                  validation_data=test_set,
                                  validation_steps=62)  # It is equal to size of test data/batch size  = 2000/32


#Part 3 Saving the model
classifier.save("model2.h5")
print("Saved model to disk")


#Part 4 - Making new predictions

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('model2.h5')

test_image = image.load_img('/Users/koro/Desktop/Ml Practise /DL/CNN/dogcat_Deploy/dataset/test_image/cat.4003.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

training_set.class_indices

  # By just typing this line u will get value for your classes (eg cat:0 , dog:1)                             
if result[0][0] == 1:                          
    prediction = 'dog'        #that [0][0] means index position of first row and 1st column, u can see in spyder
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)


#Note
#In spyder go to edit to see shortcut key command (eg cmd+1 for commenting)
# samples_per_epoch = the number of training images

# nb_val_samples = number of validation images

# These two arguments used by the author are valid if you're using older version of keras.

# but in the latest keras update, these arguments have been replaced by:

# steps_per_epoch = samples_per_epoch // batch_size

# validation_steps = nb_val_samples // batch_size
#https://www.udemy.com/course/deeplearning/learn/lecture/6798970#questions/9304844