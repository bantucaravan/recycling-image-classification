import os
import re
import sys

# location of nbrun_git_clone
#sys.path.append("/Users/noah.chasek-macfoy@ibm.com/Desktop/projects/Drone proj/code/")

from nbrun_git_clone.nbrun import run_notebook

import build_data_gens
import build_models
from noahs_utils import read_log_json

import importlib
importlib.reload(build_data_gens)
importlib.reload(build_models)


#Create the folder where to save the executed notebooks:
from pathlib import Path
savedir = 'saved experiments'
Path(savedir).mkdir(exist_ok=True)


# Sort out options
build_data_gens.list_variants()



gen_list = ['Baseline Augmentation'] # 'No Augmentation'] # 
model_list =  ['VGG16 Fine-tuning'] # ['VGG16 Fine-tuning', 'Inception-ResNet V2 finetuning final-module', 'Inception-ResNet V2 w. Dropout Model', 'VGG16 Model', 'VGG16 Model flattened', 'Inception-ResNet V2 Model', 'Inception-ResNet V2 flattened', 'Baseline Model', 'Baseline Model + Dropout', 'Lite Test'] # 
epoch_list = [70]
opt_list =   ['opt-adam-lr1e-05'] # ['opt-SGD-lr2e-05', 'opt-SGD-lr0.001', 'opt-rmsprop-lr2e-05', 'opt-adam-lr2e-05'] #


#Execute notebooks
# use itertools.product() # what about things that are true every time? (I can still add to the inputs to product..)
base_nb = 'Base Experiment.ipynb'
for DATA_GEN_CONFIG in gen_list:
    for MODEL in model_list:
        for EPOCH in epoch_list:
            for OPT in opt_list:
                nb_kwargs = {
                    'DATA_GEN_CONFIG': DATA_GEN_CONFIG,
                    'MODEL': MODEL,
                    'EPOCH': EPOCH,
                    'OPT': OPT,
                    #'BATCH_SIZE': 84,
                    'AUTO': True
                    }
                
                warmstart = len(re.findall(r'202\d-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s', MODEL)) == 1
                if not warmstart:
                    outname = f'{DATA_GEN_CONFIG} + {MODEL} + epoch{EPOCH} + {OPT}'
                if warmstart:
                    log = read_log_json(path='../logs/model_log.json', run_num=MODEL)
                    pastmodel = log['MODEL']
                    pasteps = 30#log['EPOCH']
                    outname = f'{DATA_GEN_CONFIG} +  warmstart{pastmodel} + epoch{pasteps}+{EPOCH} + {OPT}'
                    
                run_notebook(base_nb, 
                            out_path_ipynb=os.path.join(savedir, outname+'.ipynb'),
                            out_path_html=os.path.join(savedir, outname+'.html'),
                            save_html=True,
                            timeout=60*60*6,
                            nb_kwargs=nb_kwargs)
                print('Completed:',nb_kwargs)


#kw_dict {DATA_GEN_CONFIG : gen_list, MODEL : model_list, EPOCH : epoch_list, OPT opt_list:





if False:
    def test(**kwargs):
            locals().update(kwargs)
            print(locals())

            print(opt)
            print(img_shape)

    test(opt='hello', img_shape=(1,2))

    globals().update({'opt': 'hello', 'img_shape': (1, 2)})











