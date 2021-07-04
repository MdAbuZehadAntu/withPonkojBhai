import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img

data = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

img = load_img("temp_image/dog.jpg") #PIL image
x = img_to_array(img)# shape (288, 374, 3)
# print(x.shape)
x = x.reshape((1,)+x.shape) # shape (1, 288, 374, 3)
# print(x.shape)
i = 0

# flow( function generate batches of randomly transformed images
for batch in data.flow(x,batch_size=10,save_to_dir="result",save_prefix="dog",save_format="jpeg"):
    i += 1
    if i>20:
        break