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
from layers.DynamicMaxPooling import *
from utils.utility import *
from keras.activations import softmax

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


class MatchPyramid(BasicModel):
    def __init__(self, config):
        super(MatchPyramid, self).__init__(config)
        self.__name = 'MatchPyramid'
        self.check_list = ['text1_maxlen', 'text2_maxlen', 'embed', 'embed_size', 'vocab_size', 'text1_attention',
                           'text2_attention', 'kernel_size', 'kernel_count', 'dpool_size', 'dropout_rate',
                           "context_len", "context_num", "passage_attention"]
        self.embed_trainable = config['train_embed']
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)  # attention init
        self.setup(config)
        if not self.check():
            raise TypeError('[MatchPyramid] parameter check wrong')
        print('[MatchPyramid] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', [3, 3])
        self.set_default('dpool_size', [3, 10])
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3],
                            dtype='int32')
        show_layer_info('Input', dpool_index)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
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
            d_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text2_maxlen'],))(d_w)
            show_layer_info('Lambda-softmax', d_w)
            # ########## add attention weights for D_words
            d_w_layer = Lambda(lambda x: K.repeat_elements(d_w, rep=self.config['embed_size'], axis=2))(d_w)
            d_embed = Multiply()([d_w_layer, d_embed])
            show_layer_info('Dot-qw', d_embed)
        # ####################### attention text2

        cross = Dot(axes=[2, 2], normalize=False)([q_embed, d_embed])
        show_layer_info('Dot', cross)

        if self.config["passage_attention"]:
            # ########################## compute the passages attention weights
            p_cross = Permute((2, 1))(cross)
            starts = [i for i in range(0, self.config['text2_maxlen'], self.config['context_len'])]
            slice_layer = [crop(1, start, start + self.config['context_len']) for start in starts]
            slices = [slice_layer_i(p_cross) for slice_layer_i in slice_layer]
            attention_ws = []
            for slice in slices:
                s_dw = Dense(1, use_bias=False)(slice)
                s_dw = Lambda(lambda x: softmax(x, axis=1))(s_dw)
                attention_ws.append(s_dw)
            d_w = concatenate(attention_ws, 1)
            cross = Multiply()([d_w, p_cross])
            show_layer_info('Multiply', cross)
            cross = Permute((2, 1))(cross)
            # ########################## passages attention
        cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(cross)
        show_layer_info('Reshape', cross_reshape)

        conv2d = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])

        conv1 = conv2d(cross_reshape)
        show_layer_info('Conv2D', conv1)
        pool1 = dpool([conv1, dpool_index])
        show_layer_info('DynamicMaxPooling', pool1)
        pool1_flat = Flatten()(pool1)
        show_layer_info('Flatten', pool1_flat)
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        show_layer_info('Dropout', pool1_flat_drop)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        return model
