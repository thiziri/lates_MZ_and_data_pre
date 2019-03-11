# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax
from keras.layers import Reshape, Embedding, Dot
from model import BasicModel
from utils.utility import *
from layers.Match import *
from keras.utils.vis_utils import plot_model

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


class A_MVLSTM(BasicModel):
    def __init__(self, config):
        super(A_MVLSTM, self).__init__(config)
        self.__name = 'A_MVLSTM'
        self.check_list = ['text1_maxlen', 'text2_maxlen', 'embed', 'embed_size', 'train_embed',  'vocab_size',
                           'position_att_text2', 'hidden_size', 'topk', 'dropout_rate', 'text1_attention',
                           'text2_attention', 'position_att_text1', "context_len", "context_num", "passage_attention"]
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[A_MVLSTM] parameter check wrong')
        print('[A_MVLSTM] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.set_default('text1_attention', True)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding_q', q_embed)

        # ########## compute attention weights for the query words: better then mvlstm alone
        if self.config["text1_attention"]:
            q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)  # use_bias=False to simple combination
            show_layer_info('Dense', q_w)
            q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'],), name="q_w")(q_w)
            show_layer_info('Lambda-softmax', q_w)
            # ########## add attention weights for Q_words
            q_w_layer = Lambda(lambda x: K.repeat_elements(q_w, rep=self.config['embed_size'], axis=2))(q_w)
            show_layer_info('repeat', q_w_layer)
            q_embed = Multiply()([q_w_layer, q_embed])
            show_layer_info('Dot-qw', q_embed)
        # ####################### attention

        d_embed = embedding(doc)
        show_layer_info('Embedding_d', d_embed)

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
        # ####################### attention

        q_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(q_embed)
        show_layer_info('Bidirectional-LSTM_q', q_rep)

        # ################# add attention for query positions:
        if self.config["position_att_text1"]:
            pos_w = Dense(1, activation='tanh')(q_rep)  # TimeDistributed(Dense(1, activation='tanh'))(q_rep)
            pos_w = Flatten()(pos_w)
            pos_w = Activation('softmax')(pos_w)
            pos_w = RepeatVector(self.config['hidden_size']*2)(pos_w)
            pos_w = Permute([2, 1])(pos_w)
            q_rep = Multiply()([q_rep, pos_w])  # merge([q_rep, pos_w], mode='mul')

        d_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(d_embed)
        show_layer_info('Bidirectional-LSTM_d', d_rep)

        # ################# add attention for document positions:
        if self.config["position_att_text2"]:
            # https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
            # timedistributed repeats the net between brackets for every input time step of the generated by the bi-LSTM
            # so, same weights are applied for all the time steps. Without it, different weights are learned
            pos_w = Dense(1, activation='tanh')(d_rep)  # TimeDistributed(Dense(1, activation='tanh'))(d_rep)
            pos_w = Flatten()(pos_w)
            pos_w = Activation('softmax')(pos_w)
            pos_w = RepeatVector(self.config['hidden_size']*2)(pos_w)
            pos_w = Permute([2, 1])(pos_w)
            d_rep = Multiply()([d_rep, pos_w])  # merge([d_rep, pos_w], mode='mul')  #

        cross = Match(match_type='dot')([q_rep, d_rep])
        show_layer_info('Match-dot', cross)

        if self.config["passage_attention"]:
            # ########################## compute the passages attention weights
            p_cross = Reshape((self.config['text2_maxlen'], self.config['text1_maxlen']))(cross)
            show_layer_info('p_cross', p_cross)
            starts = [i for i in range(0, self.config['text2_maxlen'], self.config['context_len'])]
            slice_layer = [crop(1, start, start + self.config['context_len']) for start in starts]
            slices = [slice_layer_i(p_cross) for slice_layer_i in slice_layer]
            attention_ws = []
            for slice in slices:
                s_dw = Dense(1, use_bias=False)(slice)
                s_dw = Lambda(lambda x: softmax(x, axis=1))(s_dw)
                attention_ws.append(s_dw)
            d_w = concatenate(attention_ws, 1)
            show_layer_info('attW', d_w)
            cross = Multiply()([d_w, p_cross])
            show_layer_info('Multiply', cross)
            # ########################## passages attention

        cross_reshape = Reshape((-1,))(cross)
        show_layer_info('Reshape', cross_reshape)

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(cross_reshape)
        show_layer_info('Lambda-topk', mm_k)

        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(mm_k)
        show_layer_info('Dropout', pool1_flat_drop)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        plot_model(model, to_file='../amvlstm.png', show_shapes=True, show_layer_names=True)
        return model
