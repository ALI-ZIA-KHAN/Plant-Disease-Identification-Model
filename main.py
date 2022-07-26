


import matplotlib
matplotlib.use('Agg')

#2
# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

import os
from flask import Flask,request,send_file,jsonify
app = Flask(__name__)


# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 3
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 4
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 38, activation = 'softmax'))
classifier.summary()




classifier.load_weights('C:/Users/PSF/PycharmProjects/CVF/archive/AlexNetModel.hdf5')




# let's visualize layer names and layer indices to see how many layers
# we should freeze:
from keras import layers
for i, layer in enumerate(classifier.layers):
   print(i, layer.name)



# we chose to train the top 2 conv blocks, i.e. we will freeze
# the first 8 layers and unfreeze the rest:
print("Freezed layers:")
for i, layer in enumerate(classifier.layers[:20]):
    print(i, layer.name)
    layer.trainable = False


classifier.summary()





# Compiling the Model
from tensorflow.keras import optimizers
classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128
base_dir = "C:/Users/PSF/PycharmProjects/CVF/PlantVillage/PlantVillage"

training_set = train_datagen.flow_from_directory(base_dir+'/train',
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical',
                                                 shuffle=False)

valid_set = valid_datagen.flow_from_directory(base_dir+'/val',
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False)


class_dict = training_set.class_indices
print(class_dict)


li = list(class_dict.keys())
print(li)


li = list(class_dict.keys())
print(li)


train_num = training_set.samples
valid_num = valid_set.samples



# checkpoint
from keras.callbacks import ModelCheckpoint
weightpath = "C:/Users/PSF/PycharmProjects/CVF/archive/best_weights_9.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

#fitting images to CNN
history = classifier.fit_generator(training_set,
                        steps_per_epoch=10, #train_num//batch_size,
                         validation_data=valid_set,
                         epochs=2, #25,
                         validation_steps=5, #valid_num//batch_size,
                         callbacks=callbacks_list)
#saving model
filepath="C:/Users/PSF/PycharmProjects/CVF/archive/AlexNetModel.hdf5"
classifier.save(filepath)




#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# #accuracy plot
# plt.plot(epochs, acc, color='green', label='Training Accuracy')
# plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend()
#
# plt.figure()
# #loss plot
# plt.plot(epochs, loss, color='pink', label='Training Loss')
# plt.plot(epochs, val_loss, color='red', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()






# # predicting an image
# from keras.preprocessing import image
# import tensorflow as tf
# import numpy as np
# image_path = "C:/Users/PSF/PycharmProjects/CVF/PlantVillage/PlantVillage/train/Apple___Apple_scab/d480dbfb-ce18-40cf-a183-9bc52ec9195d___FREC_Scab 3303.JPG"
# new_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
# img = image.img_to_array(new_img)
# img = np.expand_dims(img, axis=0)
# img = img/255
#
# print("Following is our prediction:")
# prediction = classifier.predict(img)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# d = prediction.flatten()
# j = d.max()
# for index,item in enumerate(d):
#     if item == j:
#         class_name = li[index]
#
# ##Another way
# # img_class = classifier.predict_classes(img)
# # img_prob = classifier.predict_proba(img)
# # print(img_class ,img_prob )
#
#
# #ploting image with predicted class name
# plt.figure(figsize = (4,4))
# plt.imshow(new_img)
# plt.axis('off')
# plt.title(class_name)
# plt.show()




# predicting an image
from keras.preprocessing import image
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
UPLOAD_FOLDER = "./UPLOADFOLDER"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/",methods=['POST'])
def upload_file():
    file = request.files["file"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)
    print("path",path)
    # image_path = "C:/Users/PSF/PycharmProjects/CVF/PlantVillage/PlantVillage/train\Apple___Black_rot/fc5f6bd3-bd4e-4a8c-94d1-d46ea92a3941___JR_FrgE.S 3017.JPG"
    new_img = image.image_utils.load_img(path, target_size=(224, 224))
    img = image.image_utils.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    print("Following is our prediction:")
    prediction = classifier.predict(img)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = li[index]

    ##Another way
    # img_class = classifier.predict_classes(img)
    # img_prob = classifier.predict_proba(img)
    # print(img_class ,img_prob )


    #ploting image with predicted class name
    plt.figure(figsize = (4,4))
    plt.imshow(new_img)
    plt.axis('off')
    plt.title(class_name)
    plt.show()
    print("newimg",new_img)
    return jsonify({"Class_Name":class_name})





if __name__ == "__main__":
    app.run(host='192.168.43.34',port=5000,debug=True)
