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
import numpy as np
import binascii
import torch
import secrets
import random
import json
import sys
import re
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
WORDLIST_PATH = Path('wordlists')  # legitimate offense cues per language (profanity, slurs, insults), one word per line

MASK_CANDIDATES = 50  # shortcut masking: words with the largest attention growth that are checked against the wordlists
MASK_MIN_COUNT = 10  # minimum occurrences of a word in the detection sample
DETECT_SAMPLE = 3000  # training texts used for detection
WORD = re.compile(r"[\w'#@-]+")

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

def fit_embedding(base_model_file, x, y, data_ratio, attention_regularization):
    """contrastive fine-tuning on sentence pairs"""
    examples = []
    for _ in range(5):
        examples = sentence_pairs_generation(np.array(x), np.array(y), examples)
    examples = examples[:int(len(examples)*data_ratio)]  # use only part of the data for embedding fine-tuning
    model = SentenceTransformer(base_model_file)
    # attention entropy regularization (EAR) is off by default: it does not prevent shortcuts and costs pretrained information
    loss = losses.CosineSimilarityLoss(model, regularization=attention_regularization)
    model.fit(train_objectives=[(DataLoader(examples, shuffle=True, batch_size=16), loss)], epochs=EPOCHS, warmup_steps=10,
              show_progress_bar=True)
    return model


@torch.no_grad()
def received_by_word(model, texts, batch_size=32):
    """mean share of the attention that each word (lowercased) receives in a text: mean over layers, heads and query
    tokens, normalized over the non-special tokens; and the number of occurrences"""
    model.eval()
    totals, counts = defaultdict(float), defaultdict(int)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = model.tokenizer(chunk, padding=True, truncation=True, max_length=model.max_seq_length, return_tensors='pt',
                              return_offsets_mapping=True, return_special_tokens_mask=True)
        offsets, special = enc.pop('offset_mapping'), enc.pop('special_tokens_mask')
        features = {k: v.to(model.device) for k, v in enc.items()}
        attentions = torch.stack(model[0](features, output_attentions=True)['attentions'])  # L x B x H x S x S
        mask = features['attention_mask'].float()
        a = attentions.mean((0, 2))  # B x S x S
        received = ((a * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)).cpu()
        received = received * ((mask.cpu() > 0) & (special == 0))
        received = received / received.sum(1, keepdim=True).clamp_min(1e-12)
        for b, text in enumerate(chunk):
            for m in WORD.finditer(text):
                word = m.group().lower().strip("'#@-")
                if word:
                    totals[word] += sum(float(received[b, j]) for j, (start, end) in enumerate(offsets[b].tolist())
                                        if end > start and start < m.end() and end > m.start())
                    counts[word] += 1
    return {w: totals[w] / counts[w] for w in totals}, counts


def load_whitelist(lang):
    """legitimate offense cues: every file in WORDLIST_PATH/<lang> (all languages for 'multi'), one word per line"""
    files = [f for f in (WORDLIST_PATH.glob('*/*') if lang == 'multi' else (WORDLIST_PATH / lang).glob('*')) if f.is_file()]
    words = set()
    for f in files:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            words |= {line.strip().lower() for line in fh if line.strip() and ' ' not in line.strip()}
    return words


def detect_shortcuts(model, base_model, texts, whitelist, identity):
    """detect-then-mask: words whose received attention grew most during fine-tuning are shortcut candidates; those in
    the wordlists (legitimate offense cues) are kept, identity terms never count as legitimate, the rest is masked"""
    texts = random.sample(texts, min(len(texts), DETECT_SAMPLE))
    shares, counts = received_by_word(model, texts)
    shares_base, _ = received_by_word(base_model, texts)
    growth = {w: shares[w] / shares_base[w] for w in shares if counts[w] >= MASK_MIN_COUNT and shares_base.get(w)}
    candidates = sorted(growth, key=growth.get, reverse=True)[:MASK_CANDIDATES]
    listed = lambda w: w in whitelist or (w.endswith('s') and w[:-1] in whitelist) or (w.endswith('es') and w[:-2] in whitelist)
    legitimate = [w for w in candidates if w not in identity and listed(w)]
    return {'masked': [w for w in candidates if w not in legitimate], 'kept (in wordlists)': legitimate,
            'attention growth': {w: round(growth[w], 3) for w in candidates}}


def mask_words(texts, words):
    words = set(words)
    return [' '.join(w for w in text.split(' ') if re.sub(r"[^\w']", '', w.lower()) not in words) for text in texts]


@torch.no_grad()
def received_share(model, text, term):
    """share of the attention that the term's tokens receive in the text: mean over layers, heads and query tokens,
    normalized over the non-special tokens; None if the term does not occur"""
    start = text.lower().find(term.lower())
    if start < 0:
        return None
    end = start + len(term)
    enc = model.tokenizer([text], truncation=True, max_length=model.max_seq_length, return_tensors='pt',
                          return_offsets_mapping=True, return_special_tokens_mask=True)
    offsets, special = enc.pop('offset_mapping')[0], enc.pop('special_tokens_mask')[0]
    features = {k: v.to(model.device) for k, v in enc.items()}
    attentions = torch.stack(model[0](features, output_attentions=True)['attentions'])  # L x 1 x H x S x S
    received = attentions[:, 0].mean((0, 1, 2)).cpu() * (special == 0)  # S, attention received per key token
    in_term = torch.tensor([a < end and b > start and b > a for a, b in offsets.tolist()])
    return float(received[in_term].sum() / received.sum().clamp_min(1e-12))


def term_bias(model, base_model, clf, corpus):
    """bias evaluation on neutral sentences, one per identity / occupation term:
    - flagged: the classifier flags the (neutral) sentence, i.e. a false positive on the term
    - attention growth: the term's received attention share relative to the base model; terms the fine-tuned model
      attends to much more than the base model are candidates for shortcuts"""
    model.eval()
    base_model.eval()
    flags = clf.predict(model.encode([text for _, text in corpus]))
    bias = {'flagged terms': sorted(term for (term, _), flag in zip(corpus, flags) if flag),
            'false positive rate': float(np.mean(flags)), 'attention growth': {}}
    for term, text in corpus:
        share, share_base = received_share(model, text, term), received_share(base_model, text, term)
        if share is not None and share_base:
            bias['attention growth'][term] = share / share_base
    bias['attention growth'] = dict(sorted(bias['attention growth'].items(), key=lambda kv: kv[1], reverse=True))
    return bias

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

            with open(EVAL_CORPUS, 'r', encoding='utf-8') as f:
                eval_corpus = [line.rstrip('\n').split('\t', 1) for line in f if '\t' in line]  # (term, neutral sentence)
            
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
                    attention_regularization = options.get('attention regularization', False)
                    mask_shortcuts = options.get('mask shortcuts', False)
                else:
                    attention_regularization = False
                    mask_shortcuts = False
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

                bias = None
                shortcuts = None
                emissions = None  # no emissions tracked without embedding training

                ###### FINETUNE MODEL

                if train_embedding:
                    # track emissions start
                    tracker = OfflineEmissionsTracker(country_iso_code='AUT', save_to_file=False)
                    tracker.start()

                    model = fit_embedding(BASEMODEL_FILE, x_train, y_train, embedding_data_ratio, attention_regularization)

                    if mask_shortcuts:  # detect-then-mask: remove shortcut words from the training texts and fine-tune again
                        whitelist = load_whitelist(lang)
                        if not whitelist:
                            print('config error: "mask shortcuts" needs wordlists of legitimate offense cues in', WORDLIST_PATH / lang)
                            exit(0)
                        identity = {w.lower() for term, _ in eval_corpus for w in WORD.findall(term)}
                        shortcuts = detect_shortcuts(model, SentenceTransformer(BASEMODEL_FILE), x_train, whitelist, identity)
                        print('masked shortcut words:', shortcuts['masked'])
                        x_train = mask_words(x_train, shortcuts['masked'])
                        model = fit_embedding(BASEMODEL_FILE, x_train, y_train, embedding_data_ratio, attention_regularization)

                    # track emissions end
                    emissions = tracker.stop()  # Emissions as CO₂-equivalents [CO₂eq], in kg
                    del tracker
                else:
                    model = SentenceTransformer(BASEMODEL_FILE)

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

                if not target_name.startswith ('benchmark '):  # for 'all' + custom models: bias against the base model
                    bias = term_bias(model, SentenceTransformer(BASEMODEL_FILE), clf, eval_corpus)

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
                        benchmarks = {}
                        benchmark_configs = [(name, info) for name, info in model_config['targets'].items() if name.startswith ('benchmark ')]
                        benchmark_loglines = [line for line in log if line.startswith(lang) and '\tall\t' not in line]
                        for config, logline in zip (benchmark_configs, benchmark_loglines):
                            benchmarks[config[0]] = {}
                            benchmarks[config[0]]['description'] = config[1]['description']
                            benchmarks[config[0]]['link'] = config[1]['link']
                            benchmarks[config[0]]['scores'] = [float(score) for score in logline.split('\t')[-2:]]
                        
                        model_card['benchmarks'] = benchmarks

                    elif not target_name.startswith ('benchmark '):  # custom models
                        model_card['model info']['base model'] = BASEMODEL_FILE.split('/')[-1]

                if not target_name.startswith ('benchmark '):  # 'all' + custom models

                    if not update_base_model:
                        model_card['model info']['model name'] = MODEL_NAME
                        model_card['model info']['model type'] = model_config['base']['model type']
                        model_card['model info']['pretraining task'] = model_config['base']['pretraining task']
                        model_card['model info']['downstream task'] = model_config['base']['downstream task']
                        model_card['model info']['emissions'] = emissions
                        model_card.setdefault('benchmarks', {})['Self-consistency'] = {'scores': self_consistency}
                        model_card['bias'] = bias
                        if shortcuts:
                            model_card['shortcut masking'] = shortcuts

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
                        run(['rm', '-rf', tmp_path])
                    if Path('default.joblib').exists():
                        Path('default.joblib').unlink()

                    if Path(OUTPUT_FILE).exists():
                        Path(OUTPUT_FILE).rename(Path(BASE_PATH / MODEL_NAME))  # move embedding to base path for retraining
                    Path(CLF_FILE).rename(Path(BASE_PATH / CLF_FILE))  # move classifier to base path for retraining
                    run(['cp', str(OUTPUT_FILE)+'.json', str(BASE_PATH / MODEL_NAME)+'.json'])  # copy model card to base path

                # save to log (= checkpoint for resume function)
                if train_base_model:
                    log.append(lang+'\t'+target_name+'\t'+str(test_acc)+'\t'+str(test_f1))
                    with open(CURRENT_LOG, 'w') as f:
                        f.write('\n'.join(log))
    
    # move config + log to history
    Path(CONFIG_PATH).rename(Path(LOG_PATH / CONFIG_PATH.name))
    if train_base_model and CURRENT_LOG.exists():
        CURRENT_LOG.rename(Path(LOG_PATH / CURRENT_LOG.name))
