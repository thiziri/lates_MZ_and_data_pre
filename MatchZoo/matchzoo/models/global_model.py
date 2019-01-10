# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import sys
import json
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Embedding
from model import BasicModel
from utils.utility import *
from layers.Match import *
from keras.utils.vis_utils import plot_model

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo

# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed

# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((max_size, feat_size), dtype=np.float32)

    if len(embed_dict) > len(embed):
        raise Exception("vocab_size %d is larger than embed_size %d, change the vocab_size in the config!"
                        % (len(embed_dict), len(embed)))

class GLOBAL(BasicModel):
    def __init__(self, config):
        super(GLOBAL, self).__init__(config)
        self.__name = 'GLOBAL'
        self.check_list = ['text1_maxlen', 'text2_maxlen', 'embed', 'embed_size', 'train_embed',  'vocab_size',
                           "trainable", "hiden_layers", "models"]
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[GLOBAL] parameter check wrong')
        print('[GLOBAL] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.config.update(config)

    def build(self):
        # load the models to be combined:
        models_configs = {}
        for conf in self.config["models"]:
            with open(conf, 'r') as f:
                model_config = json.load(f)

                # collect embedding
                if 'embed_path' in model_config['inputs']['share']:
                    embed_dict = read_embedding(filename=model_config['inputs']['share']['embed_path'])
                    _PAD_ = model_config['inputs']['share']['vocab_size'] - 1
                    embed_dict[_PAD_] = np.zeros((model_config['inputs']['share']['embed_size'],), dtype=np.float32)
                    embed = np.float32(np.random.uniform(-0.02, 0.02, [model_config['inputs']['share']['vocab_size'],
                                                                       model_config['inputs']['share']['embed_size']]))
                    model_config['inputs']['share']['embed'] = convert_embed_2_numpy(embed_dict, embed=embed)
                else:
                    embed = np.float32(
                        np.random.uniform(-0.2, 0.2, [model_config['inputs']['share']['vocab_size'],
                                                      model_config['inputs']['share']['embed_size']]))
                    model_config['inputs']['share']['embed'] = embed
                    
                models_configs[load_model(model_config)] = model_config
        for model in models_configs:
            global_conf = models_configs[load_model(conf)]["global"]
            weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])
            model.load_weights(weights_file)

        # Inputs:
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('query', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('doc', doc)

        # Apply models:
        models_outputs = [model([query, doc]) for model in models_configs]

        # Merge the different outputs:
        merged = Concatenate()(models_outputs)
        show_layer_info('Merged', merged)

        # Additional layers:
        for i in self.config['hiden_layers']:
            merged = Dense(i, activation='tanh')(merged)
            show_layer_info('Dense', merged)

        # Final output:
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(merged)
            show_layer_info('Dense', out_)
        else:
            out_ = Dense(1, activation='softmax')(merged)
            show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        plot_model(model, to_file='../global_model' + str(len(self.config['hiden_layers'])) + ".png", show_shapes=True,
                   show_layer_names=True)

        return model
