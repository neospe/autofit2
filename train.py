#!/usr/bin/env python

"""
autofit2 training service
"""

from codecarbon import OfflineEmissionsTracker
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import DataLoader
from torch import manual_seed
from datetime import datetime
from collections import defaultdict
from joblib import dump, load
from zipfile import ZipFile
from subprocess import run
from pathlib import Path
import pandas as pd
import numpy as np
import binascii
import torch
import secrets
import random
import json
import sys
import os
from dataload.loader import *  # set DATALAKE_ADDR to load from the data lake
from dataload.preproc import *
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

EPOCHS = 3
LOG_PATH = Path('history')  # path for all configs + logs
OUTPUT_PATH = Path('models-out')  # path for all outputs
BASE_PATH = Path('models-in')  # path to base model store
EVAL_PATH = Path('eval-data')  # path to evaluation data
ASSETS_PATH = Path('packaging')  # path to custom serving handler

task_labels_mapping = {'mod': MAPPING_BINARYCLASS}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)

def sentence_pairs_generation(sentences, labels, pairs):
    # initialize two empty lists to hold the (sentence, sentence) pairs and
    # labels to indicate if a pair is positive or negative

    numClassesList = np.unique(labels)
    idx = [np.where(labels == i)[0] for i in numClassesList]

    for idxA in range(len(sentences)):      
        currentSentence = sentences[idxA]
        label = labels[idxA]
        idxB = np.random.choice(idx[np.where(numClassesList==label)[0][0]])
        posSentence = sentences[idxB]
            # prepare a positive pair and update the sentences and labels
            # lists, respectively
        pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

        negIdx = np.where(labels != label)[0]
        negSentence = sentences[np.random.choice(negIdx)]
            # prepare a negative pair of sentences and update our lists
        pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))
    
        # return a 2-tuple of our sentence pairs and labels
    return (pairs)

def compute_negative_entropy(inputs: torch.torch, attention_mask: torch.torch, return_values: bool = False):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tensor. should be in the form: LBHSS
        - attention_mask. Tensor with dim: BS
    """

    #  average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]

        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy

def get_tokens_entropy(model, data, device='cpu', batch_size=32):
    entropy_scores = defaultdict(list)
    word_occ = defaultdict(int)
        
    with torch.no_grad():
        for text in data:
            output = model.encode(text, output_value=None)
            neg_entropy, entropies = compute_negative_entropy(torch.stack(output["attentions"]), output["attention_mask"], return_values=True)

            curr_e = -entropies[0]
            curr_e = torch.flipud(curr_e)  # flip vector up/down

            input_ids = output["input_ids"]
            #input_ids = input_ids[input_ids != 0]  # not necessary
            tokens = model.tokenizer.convert_ids_to_tokens(input_ids)

            curr_e = curr_e.mean(0).unsqueeze(0)
            curr_e = curr_e[0][0]

            assert curr_e.shape[1] == len(tokens)

            for i, t in enumerate(tokens):
                entr = curr_e[0, i].cpu().item()
                entropy_scores[t].append(entr)
                word_occ[t] += 1

        # return the average
        entropy_scores = {k: np.mean(v) for k, v in entropy_scores.items()}
        return entropy_scores, word_occ

if __name__ == "__main__":

    # check arguments
    if len(sys.argv) < 2:
        print(sys.argv[0], '[path to config.json] [optional: resume]')
        exit(0)
    else:
        CONFIG_PATH = Path(sys.argv[1])
        if len(sys.argv) == 3:
            resume = bool(sys.argv[2] == 'resume')
        else:
            resume = False

    # check config sanity
    config_str = open(CONFIG_PATH, 'r', encoding='utf-8').read()
    if '"all":' in config_str and '"custom":' in config_str:
        print('config error: "all" and "custom" targets cannot be trained in the same job.')
        exit(0)
    if '"all":' not in config_str and '"custom":' not in config_str:
        print('config error: neither "all" nor "custom" target found.')
        exit(0)
    if '"all":' in config_str:  # benchmarks are optional
        train_base_model = True
    else:
        train_base_model = False

    # preliminaries
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if not LOG_PATH.exists():
        LOG_PATH.mkdir()

    if not OUTPUT_PATH.exists():
        OUTPUT_PATH.mkdir()

    if not BASE_PATH.exists():
        BASE_PATH.mkdir()

    # setup logging for base model training
    if train_base_model:
        CURRENT_LOG = Path(CONFIG_PATH.stem+'.log')
        if CURRENT_LOG.exists():
            with open(CURRENT_LOG, 'r') as f:
                log = [line.rstrip() for line in f]
        else:
            log = []

    ARCHIVER_PATH = 'torch-model-archiver'

    set_seed(0)

    # go through the config
    for task, task_config in config.items():

        for lang, model_config in task_config.items():

            BASEMODEL_FILE = model_config['base']['model file']
            labels_mapping = task_labels_mapping[task]

            # read eval data
            EVAL_TERMS = EVAL_PATH / str(lang+'-terms.txt')
            EVAL_CORPUS = EVAL_PATH / str(lang+'-corpus.tsv')
            
            if lang == 'multi':
                if EVAL_TERMS.exists(): run(['rm', str(EVAL_TERMS)])
                os.system('cat '+str(EVAL_PATH)+'/*-terms.txt > '+str(EVAL_PATH)+'/tmp')
                run(['mv', str(EVAL_PATH)+'/tmp', str(EVAL_TERMS)])
                if EVAL_CORPUS.exists(): run(['rm', str(EVAL_CORPUS)])
                os.system('cat '+str(EVAL_PATH)+'/*-corpus.tsv > '+str(EVAL_PATH)+'/tmp')
                run(['mv', str(EVAL_PATH)+'/tmp', str(EVAL_CORPUS)])

            with open(EVAL_TERMS, 'r', encoding='utf-8') as f:
                eval_termlist = [line.rstrip() for line in f]

            with open(EVAL_CORPUS, 'r', encoding='utf-8') as f:
                eval_corpus = [line.split('\t')[1].rstrip() for line in f]
            
            # loop through training targets for the specified base model
            for target_name, target_info in model_config['targets'].items():

                if train_base_model:
                    MODEL_NAME = CONFIG_PATH.stem + '-' + task + '-' + lang  # e.g. example-mod-de
                else:
                    MODEL_NAME = secrets.token_hex(8)  # 16 char lowercase alphanumeric string, e.g. '6dc0f3c1cdf79aa8'

                OUTPUT_FILE = OUTPUT_PATH / MODEL_NAME  #+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                CLF_FILE = MODEL_NAME+'.joblib'
                
                if train_base_model and resume:
                    if lang+'\t'+target_name+'\t' not in log[-1]:  # matches last line in log
                        continue                              # no, continue looping through targets
                    else:                                     # yes, continue with next target
                        resume = False
                        continue

                options = target_info.get('options', False)
                if options:
                    train_embedding = options.get('train embedding', True)
                    embedding_data_ratio = options.get('embedding sample ratio', 1.0)
                    data_ratio = options.get('sample ratio', 1.0)
                    base_clf = options.get('base clf', False)
                else:
                    train_embedding = True
                    base_clf = False
                    embedding_data_ratio = 1.0
                    data_ratio = 1.0

                ###### LOAD DATA

                # TODO: die exec lines in eigenen threads/prozessen starten

                if len(target_info['loader']) == 2 and target_name != 'all':  # train-test split defined in dataset
                    exec_line = 'train_data = '+target_info['loader'][0]
                    exec(exec_line)
                    random.shuffle(train_data)
                    exec_line = 'eval_data = '+target_info['loader'][1]
                    exec(exec_line)
                    random.shuffle(eval_data)
                else:
                    if target_name == 'all':
                        data = []
                        for loader_string in target_info['loader']:  # load all
                            exec_line = 'data += '+loader_string
                            exec(exec_line)
                    else:
                        exec_line = 'data = '+target_info['loader'][0]  # load one
                        exec(exec_line)
                        
                    random.shuffle(data)
                    data = data[:int(len(data)*data_ratio)]  # use only part of the data

                    # split data
                    if target_name == 'all':  # train=test as self-consistency check for 'all'
                        train_data = data
                        eval_data = data
                    else:
                        train_data = data[:int(len(data)*0.7)]  # 70-30 for any single dataset without its own defined train-test split
                        eval_data = data[int(len(data)*0.7):]
                
                # map labels
                x_train = [item['text'] for item in train_data]
                y_train = [labels_mapping[item['label']] for item in train_data]
                x_eval = [item['text'] for item in eval_data]
                y_eval = [labels_mapping[item['label']] for item in eval_data]

                # generate sentence pairs
                if train_embedding:
                    train_examples = []
                    for x in range(5):
                        train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)
                    train_examples = train_examples[:int(len(train_examples)*embedding_data_ratio)]  # use only part of the data for embedding fine-tuning
                    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

                if target_name == 'all':
                    regularization = [False, True]  # we need to train our own base-models two times, without and with regularization
                    # note:
                    # - the order [False, True] should be preserved for the generation of the model card, where old scores are being subtracted from new scores
                    # - the non-regularized version is not logged extra und thus not resumable
                else:
                    regularization = [True]  # all other targets (single dataset benchmarks, customer models) are regularized by default
                scores_list = []

                for reg in regularization:

                    ###### FINETUNE MODEL

                    model = SentenceTransformer(BASEMODEL_FILE)

                    if train_embedding:
                        train_loss = losses.CosineSimilarityLoss(model, regularization=reg)

                        # track emissions start
                        tracker = OfflineEmissionsTracker(country_iso_code='AUT', save_to_file=False)
                        tracker.start()
                        
                        # fit embedding
                        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCHS, warmup_steps=10, show_progress_bar=True)

                        # track emissions end
                        emissions = tracker.stop()  # Emissions as CO₂-equivalents [CO₂eq], in kg
                        del tracker

                    if reg == True:  # for 'all' (excluding nonreg) + benchmarks + custom models
                        run_id = binascii.hexlify(os.urandom(8)).decode('utf-8')
                        emb_dim = model.get_sentence_embedding_dimension()

                        X_train = np.memmap(run_id+'_train', dtype='float32', mode='w+', shape=(len(x_train), emb_dim))
                        X_train[:] = model.encode(x_train)  # cuda if available
                        del X_train  # flush memory changes to disk by deleting the object
                        X_train = np.memmap(run_id+'_train', dtype='float32', mode='r', shape=(len(x_train), emb_dim))

                        X_eval = np.memmap(run_id+'_eval', dtype='float32', mode='w+', shape=(len(x_eval), emb_dim))
                        X_eval[:] = model.encode(x_eval)
                        del X_eval
                        X_eval = np.memmap(run_id+'_eval', dtype='float32', mode='r', shape=(len(x_eval), emb_dim))

                        #print('X_train.shape:', X_train.shape)
                        #print('X_eval.shape:', X_eval.shape)

                        if base_clf != False:
                            clf = load(BASE_PATH / base_clf+'.joblib')
                            clf.warm_start = True
                        else:
                            clf = LogisticRegression(max_iter=500, n_jobs=8)
                        
                        # fit classifier
                        clf.fit(X_train, y_train)

                        # save embedding + classifier
                        if not target_name.startswith ('benchmark '):  # for 'all' + custom models
                            if train_embedding:
                                model.save(path=str(OUTPUT_FILE), model_name=MODEL_NAME, create_model_card=False)
                            dump(clf, CLF_FILE)

                        ###### EVALUATE MODEL

                        y_pred_eval_sgd = clf.predict(X_eval)
                        test_acc = accuracy_score(y_eval, y_pred_eval_sgd)
                        test_f1 = f1_score(y_eval, y_pred_eval_sgd)

                        if not target_name.startswith ('benchmark '):  # for 'all' + custom models
                            if test_acc <= 0.9 or test_f1 <= 0.9:
                                print('WARNING: low self-consistency for', OUTPUT_FILE, 'target:', target_name)
                                print('test_acc:', test_acc, 'test_f1:', test_f1)
                            self_consistency = [test_acc, test_f1]
                        
                        # cleanup
                        run(['rm', run_id+'_train', run_id+'_eval'])

                    if not target_name.startswith ('benchmark '):  # for 'all' (including nonreg) + custom models
                        ntropy_scores, word_occ = get_tokens_entropy(model, eval_corpus)  # returns a dict of every token in the data + its entropy score, word occurence

                        # filter
                        ntropy_scores = {k.replace('▁', ''): [v] for k, v in ntropy_scores.items()}  # v must be array-like or dict
                        ntropy_scores = {k: v for k, v in ntropy_scores.items() if any(term == k for term in eval_termlist)}

                        # sort
                        scores = pd.DataFrame().from_dict(ntropy_scores, orient='index', columns=['score'])
                        scores.sort_index(inplace=True)  # sort by index alphabetically
                        scores_list.append(scores)

                ###### GENERATE MODEL CARD

                # train base model and not embedding -> we want to update a base model
                update_base_model = False

                if train_base_model and not train_embedding:
                    update_base_model = True
                    with open(str(BASEMODEL_FILE)+'.json', 'r', encoding='utf-8') as f:
                        model_card = json.load(f)
                    
                    #benchmarks = {}
                    benchmark_configs = [(name, info) for name, info in model_config['targets'].items() if name.startswith ('benchmark ')]
                    benchmark_loglines = [line for line in log if line.startswith(lang) and '\tall\t' not in line]
                    for config, logline in zip (benchmark_configs, benchmark_loglines):
                        model_card['benchmarks'][config[0]] = {}
                        model_card['benchmarks'][config[0]]['description'] = config[1]['description']
                        model_card['benchmarks'][config[0]]['link'] = config[1]['link']
                        model_card['benchmarks'][config[0]]['scores'] = [float(score) for score in logline.split('\t')[-2:]]
                
                # all other cases
                else:
                    model_card = {}
                    model_card['model info'] = {}

                    if target_name == 'all':  # 'all' models
                        assert scores_list[1].index.tolist() == scores_list[0].index.tolist()
                        scores_diff = scores_list[1] - scores_list[0]  # debiased (entropy) scores should be higher, so we subtract from them
                        
                        benchmarks = {}
                        benchmark_configs = [(name, info) for name, info in model_config['targets'].items() if name.startswith ('benchmark ')]
                        benchmark_loglines = [line for line in log if line.startswith(lang) and '\tall\t' not in line]
                        for config, logline in zip (benchmark_configs, benchmark_loglines):
                            benchmarks[config[0]] = {}
                            benchmarks[config[0]]['description'] = config[1]['description']
                            benchmarks[config[0]]['link'] = config[1]['link']
                            benchmarks[config[0]]['scores'] = [float(score) for score in logline.split('\t')[-2:]]
                        
                        model_card['benchmarks'] = benchmarks
                        model_card['bias_raw'] = {k: v for k, v in scores_list[1].to_dict()['score'].items()}

                    elif not target_name.startswith ('benchmark '):  # 'all' + custom models
                        with open(str(BASEMODEL_FILE)+'.json', 'r', encoding='utf-8') as f:
                            basemodel_card = json.load(f)
                        basemodel_scores = pd.DataFrame().from_dict(basemodel_card['bias_raw'], orient='index', columns=['score'])
                        basemodel_scores.sort_index(inplace=True)
                        assert scores_list[0].index.tolist() == basemodel_scores.index.tolist()
                        scores_diff = scores_list[0] - basemodel_scores

                        model_card['model info']['base model'] = BASEMODEL_FILE.split('/')[-1]
                        model_card['bias_raw'] = {k: v for k, v in scores_list[0].to_dict()['score'].items()}

                if not target_name.startswith ('benchmark '):  # 'all' + custom models

                    if not update_base_model:
                        model_card['model info']['model name'] = MODEL_NAME
                        model_card['model info']['model type'] = model_config['base']['model type']
                        model_card['model info']['pretraining task'] = model_config['base']['pretraining task']
                        model_card['model info']['downstream task'] = model_config['base']['downstream task']
                        model_card['model info']['emissions'] = emissions
                        model_card['benchmarks']['Self-consistency'] = {'scores': self_consistency}
                        
                        scores_diff = scores_diff * 1.4426950408889  # convert entropy nats to bits: 1 nat = 1,4426950408889 bit
                        scores_diff.sort_values(by=['score'], axis='index', ascending=False, inplace=True)
                        model_card['bias_change'] = {k: v for k, v in scores_diff.to_dict()['score'].items()}

                    # save model card
                    with open(str(OUTPUT_FILE)+'.json', 'w', encoding='utf-8') as f:
                        json.dump(model_card, f)

                    ###### CREATE MODEL ARCHIVE

                    tmp_path = None
                    if target_name == 'custom' and train_embedding == False:  # repack existing archive
                        BASEMODEL_NAME = BASEMODEL_FILE.split('/')[-1]
                        tmp_path = '.'+BASEMODEL_NAME
                        with ZipFile(OUTPUT_PATH / BASEMODEL_NAME+'.mar', 'r') as zf:
                            zf.extractall(tmp_path)
                        
                        extra_files = [tmp_path+'/'+p.name for p in Path(tmp_path).glob('*.joblib') if p.name != CLF_FILE]
                        extra_files += [tmp_path+'/handler.py', str(BASE_PATH / BASEMODEL_NAME)]

                        command_string = str(ARCHIVER_PATH)+' --model-name '+BASEMODEL_NAME+' --version 1.0 --serialized-file '+BASEMODEL_FILE+'/pytorch_model.bin --handler '+tmp_path+'/run_handler.py'+' --extra-files '+','.join(extra_files)+' --export-path '+str(OUTPUT_PATH)+' --runtime python3 --force'

                    else:  # pack new archive
                        run(['cp', CLF_FILE, 'default.joblib'])
                        command_string = str(ARCHIVER_PATH)+' --model-name '+MODEL_NAME+' --version 1.0 --serialized-file '+str(Path(OUTPUT_FILE / 'pytorch_model.bin'))+' --handler '+str(Path(ASSETS_PATH /'run_handler.py'))+' --extra-files '+str(Path(ASSETS_PATH / 'handler.py'))+',default.joblib,'+str(OUTPUT_FILE)+' --export-path '+str(OUTPUT_PATH)+' --runtime python3 --force'
                    
                    try:
                        run(command_string.split(' '))
                    except:
                        print('failed to create model archive using command:', command_string)

                    ###### MISC

                    # cleanup
                    if tmp_path:
                        Path(tmp_path).unlink()
                    if Path('default.joblib').exists():
                        Path('default.joblib').unlink()

                    if Path(OUTPUT_FILE).exists():
                        Path(OUTPUT_FILE).rename(Path(BASE_PATH / MODEL_NAME))  # move embedding to base path for retraining
                    Path(CLF_FILE).rename(Path(BASE_PATH / CLF_FILE))  # move classifier to base path for retraining
                    run(['cp', str(OUTPUT_FILE)+'.json', str(BASE_PATH / MODEL_NAME)+'.json'])  # copy model card to base path (includes important info for embedding: raw debiasing scores)

                # save to log (= checkpoint for resume function)
                if train_base_model:
                    log.append(lang+'\t'+target_name+'\t'+str(test_acc)+'\t'+str(test_f1))
                    with open(CURRENT_LOG, 'w') as f:
                        f.write('\n'.join(log))
    
    # move config + log to history
    Path(CONFIG_PATH).rename(Path(LOG_PATH / CONFIG_PATH.name))
    if train_base_model and CURRENT_LOG.exists():
        CURRENT_LOG.rename(Path(LOG_PATH / CURRENT_LOG.name))
