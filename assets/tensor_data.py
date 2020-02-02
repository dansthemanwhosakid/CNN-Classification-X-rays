
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_directory = '../data/train/'
test_directory = '../data/test/'

train_img_gen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip = True,
                                   rotation_range = 30,
                                   zoom_range = .30)

train_data = train_img_gen = train_img_gen.flow_from_directory(batch_size = 32,
                                                               directory = train_directory,
                                                               shuffle = True,
                                                               target_size = (224,224),
                                                               class_mode = 'binary'
                                                              )
test_img_gen = ImageDataGenerator(rescale=1./255)

test_data = test_img_gen.flow_from_directory(batch_size = 32,
                                             directory = test_directory,
                                             shuffle = True,
                                             target_size = (224,224),
                                             class_mode = 'binary'
                                                              )
