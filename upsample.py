from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import glob

# After this generation, I will have more pictures.
def generate_more_samples(data_generator, input_dir, output_dir, save_prefix):
    images_path = glob.glob(input_dir+"*.jpg")
    for path in images_path:
        image = load_img(path)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # the .flow() command below generates batches of randomly transformed images
        i = 0
        for batch in data_generator.flow(image,
                                         batch_size=1,
                                         save_to_dir=output_dir,
                                         save_prefix=save_prefix,
                                         save_format='jpg'):
            i += 1
            if i > 4:
                break  # otherwise the generator would loop indefinitely
    print("done")


datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


generate_more_samples(datagen, "Powder/", "Powder/", "L4_generated")
generate_more_samples(datagen, "Biological/", "Biological/", "L7_generated")
generate_more_samples(datagen, "Fibres/", "Fibres/", "L9_generated")
