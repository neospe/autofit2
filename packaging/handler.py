from sentence_transformers import SentenceTransformer
from zipfile import ZipFile
from html import unescape
from joblib import load
from glob import glob
import re


url_pattern = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)')
user_pattern = re.compile('@\S+')
number_pattern = re.compile('([0-9]+?)')
ip_pattern = re.compile('\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')


def clean_str(string):
    #string_orig = string

    #string = url_pattern.sub('URL', string)
    string = url_pattern.sub('', string)
    #string = string.replace('"', '\\"')  # escape double quotation marks
    string = unescape(string)  # translate html entities to utf8 (e.g. '&amp'->'&')

    # transform emoticons to corresponding emoji
    string = string.replace(' :)', ' 🙂 ').replace(' :-)', ' 🙂 ').replace(' ;)', ' 😉 ').replace(' ;-)', ' 😉 ').replace(' :D', ' 😁 ').replace(' :-D', ' 😁 ').replace(' :(', ' 🙁 ').replace(' :-(', ' 🙁 ').replace(' :/', ' 😕 ').replace(' :-/', ' 😕 ').replace(' 8)', ' 😎 ').replace(' 8-)', ' 😎 ').replace(' X)', ' 😵 ').replace(' X-)', ' 😵 ').replace(' :-P', ' 😋 ').replace(' :P', ' 😋 ').replace(' <3', ' ❤️ ').replace(' :\'(', ' 😢 ').replace(' :\'-(', ' 😢 ')

    #string = user_pattern.sub('@USER', string)
    #string = number_pattern.sub('NUMBER', string)
    string = user_pattern.sub('', string)
    string = number_pattern.sub('', string)
    string = string.replace('|LBR|', '')
    string = string.replace('\n', '')

    # remove bpemb special characters
    string = string.replace('<unk>', '').replace('<s>', '').replace('</s>', '')

    # remove excess whitespace
    string = string.replace('   ', ' ').replace('  ', ' ').strip()

    # debug
    #if string_orig != string:
    #    print(string_orig)
    #    print(string, '\n')

    return string


class SentenceTransformerHandler(object):
    
    def __init__(self):
        super(SentenceTransformerHandler, self).__init__()
        self.initialized = False
        self.embedder = None
        self.clf = None
        self.clf_store = {}
        self.max_clf = 100  # maximum number of classifiers to load
    
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")        
        try:
            with ZipFile(model_dir + '/pytorch_model.bin',  'r') as zip_ref:
                zip_ref.extractall(model_dir)
        except:
            print('tried unzipping again')
        self.embedder = SentenceTransformer(model_dir)

        # load one classifier
        #self.clf = load(model_dir + '/default.joblib')
        
        # load multiple classifiers
        clf_paths = [p for p in glob(model_dir+'/*.joblib')][:self.max_clf]
        for p in clf_paths:
            model_id = p.split('/')[-1].replace('.joblib', '')
            self.clf_store[model_id] = load(p)
        
        self.initialized = True
    
    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        inputs = data[0].get('data')
        if inputs is None:
            inputs = data[0].get('body')
        sentences = inputs['queries']

        # if multiple classifiers: select by model id
        self.clf = self.clf_store[inputs['model id']]

        return sentences
    
    def inference(self, data):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        query_embeddings = self.embedder.encode(data)  # returns np.ndarray
        score = self.clf.predict_proba(query_embeddings.reshape(1, -1))
        return score[0][1]  # return the second probability = the positive class
    
    def postprocess(self, data):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        return [data]