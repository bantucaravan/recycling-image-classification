#from tensorflow.keras.callbacks import ModelCheckpoint#, TensorBoard
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import InceptionResNetV2, VGG16
from tensorflow.keras import optimizers

import os
import re
import glob




def get_model(desc, OPT, IMG_SHAPE):   #**kwargs):
    '''
    ###Must be keyword args, arbitray 


    Issue: combine initilize_models() and get_model() so you pass 
    init args and if condition to only initialize and return relevant model
    '''

    ## SET UP VARIABLE ENVIRONMENT
    #assert ('IMG_SHAPE' in kwargs) and ('IMG_SHAPE' in kwargs)
    
    # make the args available in func scope namespace
    #locals().update(kwargs) # why does this not work?????
    # I should just assing from the kwargs dict below!!!
    #globals().update(kwargs) # successful
    
    #print(locals())
    #print(globals())

    # extract learning rate
    # alt: try/except
    lr= 0.01
    lr_str = str(lr)
    tmp = re.findall(r'lr([0-9\-e\.]+)', OPT)
    if len(tmp) == 1:
        lr = float(tmp[0])
        lr_str = tmp[0]
    print(lr)
    opts = {
    # Note: 2e-5 --> 2e-05
    'opt-rmsprop-lr2e-05' : optimizers.RMSprop(lr=2e-05),
    'opt-adam' : 'adam',
    f'opt-adam-lr{lr_str}' : optimizers.Adam(learning_rate=lr), 
    'opt-SGD' : optimizers.SGD(lr=0.01, nesterov=True), # lr is default
    'opt-SGD-lr%s' %(str(lr)) : optimizers.SGD(lr=lr, nesterov=True),
    }
    
    
    print('IMG_SHAPE:',IMG_SHAPE)
    print('OPT:',OPT)


    OPT = opts[OPT]

    # Make it available globally for get_model func
    #global models
    #models = {}

    ###############################
    ######### Baseline

    if desc == 'Baseline Model':
    #desc1 = 'Baseline Model'

        model1 = Sequential()
        model1.add(Conv2D(32, (3, 3), activation='relu',input_shape=(*IMG_SHAPE, 3)))
        model1.add(MaxPooling2D((2, 2)))
        #model1.add(Dropout(.2))
        model1.add(Conv2D(64, (3, 3), activation='relu'))
        model1.add(MaxPooling2D((2, 2)))
        #model1.add(Dropout(.2))
        model1.add(Conv2D(128, (3, 3), activation='relu'))
        model1.add(MaxPooling2D((2, 2)))
        #model1.add(Dropout(.2))
        model1.add(Conv2D(128, (3, 3), activation='relu'))
        model1.add(MaxPooling2D((2, 2)))
        model1.add(Dropout(.2))
        model1.add(Flatten()) 
        model1.add(Dense(512, activation='relu'))
        model1.add(Dense(6, activation='softmax'))

        #model1.summary()
        model1.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model1.compile(**opt_test)
        return model1

        #models[desc1] = model1

    ###########################################
    ########## Baseline + Dropout

    if desc == 'Baseline Model + Dropout':
    #desc2 = 'Baseline Model + Dropout'

        model2 = Sequential()
        model2.add(Conv2D(32, (3, 3), activation='relu',input_shape=(*IMG_SHAPE, 3)))
        model2.add(MaxPooling2D((2, 2)))
        model2.add(Dropout(.2))
        model2.add(Conv2D(64, (3, 3), activation='relu'))
        model2.add(MaxPooling2D((2, 2)))
        model2.add(Dropout(.2))
        model2.add(Conv2D(128, (3, 3), activation='relu'))
        model2.add(MaxPooling2D((2, 2)))
        model2.add(Dropout(.2))
        model2.add(Conv2D(128, (3, 3), activation='relu'))
        model2.add(MaxPooling2D((2, 2)))
        model2.add(Dropout(.2))
        model2.add(Flatten())
        model2.add(Dense(512, activation='relu'))
        model2.add(Dense(6, activation='softmax'))

        #model2.summary()
        model2.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model2.compile(**opt_test)
        return model2

        #models[desc2] = model2





    ###############################################
    ############ Smaller Baseline 

    #desc3 = 'Smaller Baseline Model'
    if desc == 'Smaller Baseline Model':

        model3 = Sequential()
        model3.add(Conv2D(16, (3, 3), activation='relu',input_shape=(*IMG_SHAPE, 3)))
        model3.add(MaxPooling2D((2, 2)))
        model3.add(Dropout(.2))
        model3.add(Conv2D(32, (3, 3), activation='relu'))
        model3.add(MaxPooling2D((2, 2)))
        model3.add(Dropout(.2))
        model3.add(Conv2D(64, (3, 3), activation='relu'))
        model3.add(MaxPooling2D((2, 2)))
        model3.add(Dropout(.2))
        model3.add(Flatten())
        model3.add(Dense(256, activation='relu'))
        model3.add(Dense(6, activation='softmax'))

        #model3.summary()
        model3.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model3.compile(**opt_test)
        return model3

        #models[desc3]= model3

    ###############################################
    ############ Lite Test 

    #desc4 ='Lite Test'
    if desc == 'Lite Test':

        model4 = Sequential()
        model4.add(Conv2D(16, (3, 3), activation='relu',input_shape=(*IMG_SHAPE, 3)))
        model4.add(MaxPooling2D((2, 2)))
        model4.add(Flatten())
        model4.add(Dense(256, activation='relu'))
        model4.add(Dense(6, activation='softmax'))

        #model4.summary()
        model4.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model4.compile(**opt_test)
        return model4

        #models[desc4] = model4


    ###############################################
    ############ Inception-ResNet V2 

    #desc ='Inception-ResNet V2 Model'
    if desc == 'Inception-ResNet V2 Model':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        #if not os.path.exists('~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context

        conv_base = InceptionResNetV2(weights='imagenet', #default
                    include_top=False,
                    input_shape=(*IMG_SHAPE, 3),
                    pooling='avg') #global avg pooling? instead of flatten later? 'avg'
        #conv_base.summary()
        conv_base.trainable = False
        model5 = Sequential()
        model5.add(conv_base)
        #model5.add(layers.Flatten()) # global avg pooling instead of flatten
        model5.add(Dense(256, activation='relu')) # 512?
        model5.add(Dense(6, activation='softmax'))
        
        #assert = len(model5.trainable_weights) == 4 bias + wts arrays per 2 dense layers
        #model5.summary()
        model5.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model5.compile(**opt_test)
        return model5

        #models[desc5] = model5

    ###############################################
    ############ 

    #desc6 ='Inception-ResNet V2 flattened'
    if desc == 'Inception-ResNet V2 flattened':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        #if not os.path.exists('/Users/noahchasekmacfoy/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context


    
        conv_base = InceptionResNetV2(weights='imagenet', #default
                    include_top=False,
                    input_shape=(*IMG_SHAPE, 3),
                    pooling=None) #global avg pooling? instead of flatten later? 'avg'
        #conv_base.summary()
        conv_base.trainable = False
        model6 = Sequential()
        model6.add(conv_base)
        model6.add(Flatten())
        model6.add(Dense(256, activation='relu')) # 512?
        model6.add(Dense(6, activation='softmax'))
        
        #assert = len(model6.trainable_weights) == 4 bias + wts arrays per 2 dense layers
        #model6.summary()
        model6.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model6.compile(**opt_test)
        return model6

        #models[desc8] = model6

    ##################################################
    ############################
    #desc ='Inception-ResNet V2 Model'
    if desc == 'VGG16 Model flattened':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        if not os.path.exists('/Users/noahchasekmacfoy/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'):
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            print('Downloading weights!!!!')

        conv_base = VGG16(weights='imagenet', #default
                    include_top=False,
                    input_shape=(*IMG_SHAPE, 3),
                    pooling=None) #global avg pooling? instead of flatten later? 'avg'
        #conv_base.summary()
        conv_base.trainable = False
        model = Sequential()
        model.add(conv_base)
        model.add(Flatten()) # global avg pooling instead of flatten
        model.add(Dense(256, activation='relu')) # 512?
        model.add(Dense(6, activation='softmax'))
        
        #assert = len(model5.trainable_weights) == 4 bias + wts arrays per 2 dense layers
        #model.summary()
        model.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model.compile(**opt_test)
        return model

        #models[desc5] = model

    ##################################################
    ############################
    #desc ='Inception-ResNet V2 Model'
    if desc == 'VGG16 Model':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        if not os.path.exists('/Users/noahchasekmacfoy/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'):
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            print('Downloading weights!!!!')

        conv_base = VGG16(weights='imagenet', #default
                    include_top=False,
                    input_shape=(*IMG_SHAPE, 3),
                    pooling='avg') #global avg pooling? instead of flatten later? 'avg'
        #conv_base.summary()
        conv_base.trainable = False
        model = Sequential()
        model.add(conv_base)
        model.add(Dense(256, activation='relu')) # 512?
        model.add(Dense(6, activation='softmax'))
        
        #assert = len(model5.trainable_weights) == 4 bias + wts arrays per 2 dense layers
        #model.summary()
        model.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model.compile(**opt_test)
        return model

        #models[desc5] = model


    ###############################################
    ############ 

    if desc == 'Inception-ResNet V2 deeper top Model':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        #if not os.path.exists('~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context


    
        conv_base = InceptionResNetV2(weights='imagenet', #default
                    include_top=False,
                    input_shape=(*IMG_SHAPE, 3),
                    pooling='avg') #global avg pooling? instead of flatten later? 'avg'
        #conv_base.summary()
        conv_base.trainable = False
        model7 = Sequential()
        model7.add(conv_base)
        model7.add(Dense(512, activation='relu')) 
        model7.add(Dense(256, activation='relu'))
        model7.add(Dropout(.2))
        model7.add(Dense(50, activation='relu')) # 512?
        model7.add(Dropout(.2))
        model7.add(Dense(6, activation='softmax'))
        
        #assert = len(model7.trainable_weights) == 4 # bias + wt array per dense layer
        #model7.summary()
        model7.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model7.compile(**opt_test)
        return model7

        #models[desc8] = model7


    ###############################################
    ############ Inception-ResNet V2 

    #desc ='Inception-ResNet V2 Model'
    if desc == 'Inception-ResNet V2 w. Dropout Model':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        #if not os.path.exists('~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context

        conv_base = InceptionResNetV2(weights='imagenet', #default
                    include_top=False,
                    input_shape=(*IMG_SHAPE, 3),
                    pooling='avg') #global avg pooling? instead of flatten later? 'avg'
        #conv_base.summary()
        conv_base.trainable = False
        model = Sequential()
        model.add(conv_base)
        model.add(Dropout(.2))
        model.add(Dense(256, activation='relu')) # 512?
        model.add(Dense(6, activation='softmax'))
        
        #assert = len(model.trainable_weights) == 4 bias + wts arrays per 2 dense layers
        #model.summary()
        model.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        #model.compile(**opt_test)
        return model

        #models[desc5] = model

    ###############################################
    ############################

    #desc ='Inception-ResNet V2 Model'
    if desc == 'Inception-ResNet V2 finetuning final-module':

        # Code from https://github.com/fchollet/deep-learning-models/issues/33#issuecomment-397257502
        # which is a great discussion on the issues downloading files
        #if not os.path.exists('~/.keras/models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        #    import ssl
        #    ssl._create_default_https_context = ssl._create_unverified_context

        #half_cooked_ts = '2020-02-07_03h46m48s' # InceptionResNetV2 60eps opt-adam
        half_cooked_ts = '2020-02-09_01h00m31s' # 35 ep InceptionResNEtV2 w/ Dropout post Base , opt-adam
        get_loss = lambda x: float(re.findall(r'_(\d\.\d{2})_', x)[0])
        pat = '../saved models/model_epoch*_*_%s.h5' %(half_cooked_ts)
        modelpaths = sorted(glob.glob(pat), key=get_loss)
        bestmodelpath = modelpaths[-1]
        model = load_model(bestmodelpath)

        assert len(model.trainable_weights) == 4
        assert sum(i.trainable for i in model.layers[0].layers) == 1 # input layer

        #Set final Inception-ResNet-C Module to trainable
        #see paper: https://arxiv.org/pdf/1602.07261v2.pdf
        #Model summary (saved) and model graph (saved)
        #after 'block8_9_ac', index 761
        for i, l in enumerate(model.layers[0].layers):
            if i > 761:
                l.trainable = True

        assert sum(i.trainable for i in model.layers[0].layers) == 20
        
        model.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        return model

        #models[desc5] = model

    ###############################################
    ############################


    if len(re.findall(r'202\d-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s', desc)) == 1:
        savedir = '../saved models'
        if desc == '2020-02-07_01h36m15s': # baseline dropout
            f = 'model_epoch58_0.67_2020-02-07_01h36m15s.h5'         
            f  = os.path.join(savedir, f)
        elif desc == '2020-02-07_01h10m05s': # baseline only
            f = 'model_epoch58_0.70_2020-02-07_01h10m05s.h5'
            f  = os.path.join(savedir, f)
        
        else:
            get_loss = lambda x: float(re.findall(r'_(\d\.\d{2})_', x)[0])
            opt_idx = -1 # assume acc is reported
            pat = '../saved models/model_epoch*_*_%s.h5' %(desc)
            try:
                f = sorted(glob.glob(pat), key=get_loss)[opt_idx]
            except IndexError:
                msg = 'Saved models from run "%s" were not found in "%s"' %(desc, savedir)
                raise ValueError(msg)
        
        model = load_model(f)
        print('Starting trainng from saved model: %s' %(f))
        return model

    ###############################################
    ############################
        
    if desc == 'VGG16 Fine-tuning': # on final conv block (with dropout added to top)
        half_cooked_ts = '2020-02-08_23h29m06s' # VGG16 flattened 30 eps opt-adam
        get_loss = lambda x: float(re.findall(r'_(\d\.\d{2})_', x)[0])
        pat = '../saved models/model_epoch*_*_%s.h5' %(half_cooked_ts)
        modelpaths = sorted(glob.glob(pat), key=get_loss)
        bestmodelpath = modelpaths[-1]
        model = load_model(bestmodelpath)
        #model.layers[0].summary()

        assert len(model.trainable_weights) == 4
        assert sum(i.trainable for i in model.layers[0].layers) == 1 # input layer

        #Set final Cnv block to trainable
        #at 'block5_conv1', index 15

        #for i, l in enumerate(model.layers[0].layers):
        #   print(i, l.name, l.trainable)

        # I choose not to set trainable by  layer name bc name may 
        # change? if other conv models have been previously loaded 
        # in this python session.
        for i, l in enumerate(model.layers[0].layers):
            if i > 14:
                l.trainable = True

        assert sum(i.trainable for i in model.layers[0].layers) == 5

        # add drop out on penultimate layer
        a = Sequential()
        for i,l in enumerate(model.layers):
            a.add(l)
            if i == 2:
                a.add(Dropout(.2))
        #a.summary()
        model = a
        
        model.compile(optimizer=OPT, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        
        return model


    ################# END #######################


    else:

        raise ValueError('Model identifier "%s" not found.' %(desc))

    # Make models avaiable globally in module
    # draw backs to this?
    #globals().update(locals()) # unnecessary bc I made my own dict for naming clarity





if False:
    # save inception resnet summary bc longer than longest allowed print in vscode
    rec = []
    def test(_str):
        rec.append(_str)
    conv_base.summary(print_fn=test)
    with open('../InceptionResNetV2 Summary.txt', 'w') as f:
        f.write('\n'.join(rec))



    ### get model graph
    from tensorflow.keras.utils import plot_model, model_to_dot
    dpi = 150
    plot_model(conv_base, to_file=f'../InceptionResNetV2 graph {dpi}dpi.pdf', show_shapes=True, expand_nested=True, dpi=dpi)
    # jpg not 96 dpi seems to output a zero byte file!!!!!


    # using the `dot` command line tool (which pydot calls) setting dpi above 102 returns a zero 
    # byte image for jpgs, it might be some internal dimension cut off thing, only with jpg tho (but jpg 
    # is easiet ux when opening in gui)
    dot = model_to_dot(conv_base, show_shapes=True, expand_nested=True, dpi=150, rankdir='TB')
    dpi = 102
    to_file=f'../InceptionResNetV2 expanded graph {dpi}dpi.jpg'
    dot.write(to_file, format='jpg', prog=['dot', f'-Gdpi={dpi}'])












