%load_ext autoreload
%autoreload 2

import os
import re
import sys
import itertools

# location of nbrun_git_clone
#sys.path.append("/Users/noah.chasek-macfoy@ibm.com/Desktop/projects/Drone proj/code/")

from nbrun_git_clone.nbrun import run_notebook

import build_data_gens
import build_models
from noahs_utils import read_log_json

sys.path.append("/Users/noah.chasek-macfoy@ibm.com/Desktop/projects/Noah's Utils [git repo]")
from noahs.experiment_tools import run_nb




#Create the folder where to save the executed notebooks:
from pathlib import Path
savedir = 'saved experiments'
Path(savedir).mkdir(exist_ok=True)


# Sort out options
build_data_gens.list_variants()


# set base outname
def set_outname(nb_kwargs, pasteps):
    '''
    nb_kwargs (dict)
    '''
    
    locals().update(nb_kwargs)

    warmstart = len(re.findall(r'202\d-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s', MODEL)) == 1
    if not warmstart:
        outname = f'{DATA_GEN_CONFIG} + {MODEL} + epoch{EPOCH} + {OPT}'
    if warmstart:
        log = read_log_json(path='../logs/model_log.json', run_num=MODEL)
        pastmodel = log['MODEL']
        pasteps = pasteps#log['EPOCH']
        outname = f'{DATA_GEN_CONFIG} +  warmstart{pastmodel} + epoch{pasteps}+{EPOCH} + {OPT}'
    return outname


## Execute notebooks

variants = {
    'DATA_GEN_CONFIG': ['Baseline Augmentation'], # 'No Augmentation'] # 
    'MODEL': ['VGG16 Fine-tuning'], # ['VGG16 Fine-tuning', 'Inception-ResNet V2 finetuning final-module', 'Inception-ResNet V2 w. Dropout Model', 'VGG16 Model', 'VGG16 Model flattened', 'Inception-ResNet V2 Model', 'Inception-ResNet V2 flattened', 'Baseline Model', 'Baseline Model + Dropout', 'Lite Test'] # ,
    'EPOCH': [70, 45, 60],
    'OPT': ['opt-adam-lr1e-05'], # ['opt-SGD-lr2e-05', 'opt-SGD-lr0.001', 'opt-rmsprop-lr2e-05', 'opt-adam-lr2e-05'] #
    #'BATCH_SIZE': 84,
    'AUTO': [True]
    }


base_nb = 'Base Experiment.ipynb'   

run_nb(variants, base_nb, outname_func=set_outname, save_html=True,
       save_ipynb=True, savedir=savedir)





# experimentation
if False:
    def test(**kwargs):
            locals().update(kwargs)
            print(locals())

            print(opt)
            print(img_shape)

    test(opt='hello', img_shape=(1,2))

    globals().update({'opt': 'hello', 'img_shape': (1, 2)})











