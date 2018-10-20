import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import Augmentor

img_width, img_height = 400, 400
model_path = './models/mtg-model-real.h5'

model_weights_path = './models/
first_real.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)


def predict(file):
#   x = load_img(file, target_size=(img_height, img_width))
#   x = img_to_array(x)
  x = file
#   x = np.expand_dims(x, axis=0)
  array = model.predict(x)
#   print(file)
#   print(array)
#   result = array[0]
#   answer = np.argmax(result)
#   if answer == 0:
#     print("Label: 2ed")
#   elif answer == 1:
#     print("Labels: 3ed")
#   elif answer == 3:
#     print("Label: 4ed")
#   elif answer == 4:
#     print("Labels: 5ed")
#   elif answer == 5:
#     print("Label: 6ed")
#   elif answer == 6:
#     print("Labels: 7ed")
#   elif answer == 7:
#     print("Label: 8ed")
#   elif answer == 8:
#     print("Labels: 9ed")
  return array



q = Augmentor.Pipeline(source_directory="../test")
# added on server
q.greyscale(probability=1)
q.resize(probability=1, width=610, height=800, resample_filter=u'BICUBIC')
q.zoom(probability=1, min_factor=0.55, max_factor=0.55)
q.rotate_without_crop(probability=1, max_left_rotation=90, max_right_rotation=90)
q.skew(probability=0.8, magnitude=0.15)
q.zoom(probability=1, min_factor=0.5, max_factor=1)
q.crop_by_size(probability=1, width=610, height=610, centre=True)
q.random_brightness(probability=0.9, min_factor=0.6, max_factor=1.4)
q.resize(probability=1, width=400, height=400, resample_filter=u'BICUBIC')
g_validate = q.keras_generator(batch_size=1)
images, labels = next(g_validate)
print("---")
print(labels[0])
print(predict(images[0]))

#
# for i, ret in enumerate(os.walk('../test')):
#   for i, filename in enumerate(ret[2]):
#     if filename.startswith("."):
#       continue
#     result = predict(ret[0] + '/' + filename)