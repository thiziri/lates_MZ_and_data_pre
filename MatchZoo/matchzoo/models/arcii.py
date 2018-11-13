# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Dot
from keras.optimizers import Adam
from model import BasicModel

from layers.Match import *
from utils.utility import *
from keras.activations import softmax

class ARCII(BasicModel):
    def __init__(self, config):
        super(ARCII, self).__init__(config)
        self.__name = 'ARCII'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                    'text1_attention', 'text2_attention',
                   '1d_kernel_size', '1d_kernel_count',
                   'num_conv2d_layers','2d_kernel_sizes',
                   '2d_kernel_counts','2d_mpool_sizes', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)  # attention init
        self.setup(config)
        if not self.check():
            raise TypeError('[ARCII] parameter check wrong')
        print('[ARCII] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('1d_kernel_count', 32)
        self.set_default('1d_kernel_size', 3)
        self.set_default('num_conv2d_layers', 2)
        self.set_default('2d_kernel_counts', [32, 32])
        self.set_default('2d_kernel_sizes', [[3, 3], [3, 3]])
        self.set_default('2d_mpool_sizes',  [[3, 3], [3, 3]])
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        def conv2d_work(input_dim):
            seq = Sequential()
            assert self.config['num_conv2d_layers'] > 0
            for i in range(self.config['num_conv2d_layers']):
                seq.add(Conv2D(filters=self.config['2d_kernel_counts'][i], kernel_size=self.config['2d_kernel_sizes'][i], padding='same', activation='relu'))
                seq.add(MaxPooling2D(pool_size=(self.config['2d_mpool_sizes'][i][0], self.config['2d_mpool_sizes'][i][1])))
            return seq
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        # ########## compute attention weights for the query words: better then mvlstm alone
        if self.config["text1_attention"]:
            q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(
                q_embed)  # use_bias=False to simple combination
            show_layer_info('Dense', q_w)
            q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'],), name="q_w")(q_w)
            show_layer_info('Lambda-softmax', q_w)
            # ########## add attention weights for Q_words
            q_w_layer = Lambda(lambda x: K.repeat_elements(q_w, rep=self.config['embed_size'], axis=2))(q_w)
            show_layer_info('repeat', q_w_layer)
            q_embed = Multiply()([q_w_layer, q_embed])
            show_layer_info('Dot-qw', q_embed)
        # ####################### attention text1

        # ########## compute attention weights for the document words:
        if self.config['text2_attention']:
            d_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(d_embed)
            show_layer_info('Dense', d_w)
            d_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text2_maxlen'],), name="d_w")(d_w)
            show_layer_info('Lambda-softmax', d_w)
            # ########## add attention weights for D_words
            d_w_layer = Lambda(lambda x: K.repeat_elements(d_w, rep=self.config['embed_size'], axis=2))(d_w)
            d_embed = Multiply()([d_w_layer, d_embed])
            show_layer_info('Dot-qw', d_embed)
        # ####################### attention text2

        q_conv1 = Conv1D(self.config['1d_kernel_count'], self.config['1d_kernel_size'], padding='same') (q_embed)
        show_layer_info('Conv1D', q_conv1)
        d_conv1 = Conv1D(self.config['1d_kernel_count'], self.config['1d_kernel_size'], padding='same') (d_embed)
        show_layer_info('Conv1D', d_conv1)

        cross = Match(match_type='plus')([q_conv1, d_conv1])
        show_layer_info('Match-plus', cross)

        z = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], -1))(cross)
        show_layer_info('Reshape', z)

        for i in range(self.config['num_conv2d_layers']):
            z = Conv2D(filters=self.config['2d_kernel_counts'][i], kernel_size=self.config['2d_kernel_sizes'][i], padding='same', activation='relu')(z)
            show_layer_info('Conv2D', z)
            z = MaxPooling2D(pool_size=(self.config['2d_mpool_sizes'][i][0], self.config['2d_mpool_sizes'][i][1]))(z)
            show_layer_info('MaxPooling2D', z)

        #dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])([conv2d, dpool_index])
        pool1_flat = Flatten()(z)
        show_layer_info('Flatten', pool1_flat)
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        show_layer_info('Dropout', pool1_flat_drop)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        return model
