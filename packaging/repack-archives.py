#!/usr/bin/env python

import json
from pathlib import Path
from subprocess import run

OUTPUT_PATH = Path('models-out')  # path for all outputs
BASE_PATH = Path('models-in')  # path to base model store
ASSETS_PATH = Path('packaging')  # path to custom serving handler


# classic repacking (without support for multiple clf)
"""
for path in BASE_PATH.glob('*.json'):
    model_name = path.stem
    tmp_path = '.'+model_name
    clf_path = str(path).replace('.json', '.joblib')

    # recover classifier objects from model archive
    if not Path(clf_path).exists():
        Path(tmp_path).mkdir()
        run(['unzip', str(OUTPUT_PATH)+'/'+model_name+'.mar', '-d', tmp_path])
        run(['cp', tmp_path+'/default.joblib', clf_path])
        run(['rm', '-rf', tmp_path])

    # repack archive
    run(['cp', clf_path, 'default.joblib'])
    OUTPUT_FILE = BASE_PATH / model_name
    command_string = 'torch-model-archiver --model-name '+model_name+' --version 1.0 --serialized-file '+str(Path(OUTPUT_FILE / 'pytorch_model.bin'))+' --handler '+str(Path(ASSETS_PATH /'run_handler.py'))+' --extra-files '+str(Path(ASSETS_PATH / 'handler.py'))+',default.joblib,'+str(OUTPUT_FILE)+' --export-path '+str(OUTPUT_PATH)+' --runtime python3 --force'
    try:
        run(command_string.split(' '))
    except:
        print('failed to create model archive using command:', command_string)
    Path('default.joblib').unlink()
"""


# repacking all models
"""
# build dict: { "base-model1": ["clf1", "clf2", ..], "base-model2": ..}
packaging_dict = {}
for path in BASE_PATH.glob('*.json'):
    config_str = open(str(path), 'r', encoding='utf-8').read()
    if '"custom":' in config_str:
        with open(str(path), 'r', encoding='utf-8') as f:
            config = json.load(f)
        task_key = list(config.keys())[0]
        lang_key = list(config[task_key].keys())[0]
        basemodel_path = config[task_key][lang_key]['base']['model file']  # e.g. 'models-in/example-mod-de'
        custom_clf_path = str(path).replace('.json', '.joblib')
        
        if not basemodel_path in packaging_dict.keys():
            packaging_dict[basemodel_path] = []
        packaging_dict[basemodel_path] += [custom_clf_path]

# build model archives: base models + alle zugehörigen clfs verpacken
for basemodel_path, custom_clf_list in packaging_dict.items():
    basemodel_name = basemodel_path.split('/')[-1]
    default_clf_path = str(basemodel_path)+'.joblib'
    
    run(['cp', default_clf_path, 'default.joblib'])
    extra_files = [str(Path(ASSETS_PATH / 'handler.py')), str(BASE_PATH / basemodel_name), 'default.joblib'] + custom_clf_list
    command_string = 'torch-model-archiver --model-name '+basemodel_name+' --version 1.0 --serialized-file '+basemodel_path+'/pytorch_model.bin --handler '+str(Path(ASSETS_PATH /'run_handler.py'))+' --extra-files '+','.join(extra_files)+' --export-path '+str(OUTPUT_PATH)+' --runtime python3 --force'
    Path('default.joblib').unlink()
"""
