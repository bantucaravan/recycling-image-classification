
import os
import glob
import re
import datetime as dt
from collections import defaultdict
import sys
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

#import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.callbacks import ModelCheckpoint#, TensorBoard
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense

from noahs_utils import *



def reconstruct_img_gen(log, subset='val'):
    new_gen_config = ImageDataGenerator()
    vars(new_gen_config).update(log[subset+'_data_gen_config'])
    goodargs = try_args(log[subset+'_data_gen'], new_gen_config.flow_from_directory)
    new_gen = new_gen_config.flow_from_directory(**goodargs)
    diffs = {k:(v,vars(new_gen)[k])  for k,v in log[subset+'_data_gen'].items() if not vars(new_gen)[k] == log[subset+'_data_gen'][k]}
    print('Not matching internal vars:\n', diffs)
    return new_gen