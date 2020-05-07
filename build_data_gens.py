from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_gen_config(desc):
    '''
    Returns: Tuple of (train_gen_config, val_gen_config)
    '''
    return data_gen_configs[desc]


def list_variants():
    return list(data_gen_configs.keys())


data_gen_configs = {}


## !!! No name= attr in IMAGEDATAGENERATORS

######################################
######### No Augmentation

desc1 = 'No Augmentation'

train_gen_config1 = ImageDataGenerator(rescale=1./255)
val_gen_config1 = ImageDataGenerator(rescale=1./255)

data_gen_configs[desc1] = (train_gen_config1, val_gen_config1)


######################################
######### Baseline Augmentation

desc2 = 'Baseline Augmentation'

train_gen_config2 = ImageDataGenerator(rescale=1./255,
                                rotation_range=45,
                                width_shift_range=.15,
                                height_shift_range=.15,
                                #shear_range=.2,
                                horizontal_flip=True,
                                zoom_range=0.5,
                                fill_mode='nearest') #default
val_gen_config2 = ImageDataGenerator(rescale=1./255)


data_gen_configs[desc2] = (train_gen_config2, val_gen_config2)