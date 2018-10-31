# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.layers import Embedding
from model import BasicModel
from utils.utility import *
from layers.Match import *
from keras.utils.vis_utils import plot_model

class A_CCNNM(BasicModel):
    def __init__(self, config):
        super(A_CCNNM, self).__init__(config)
        self.__name = 'A_CCNNM'
        self.check_list = ['text1_maxlen', 'embed', 'embed_size', 'train_embed',  'vocab_size',
                           "kernel_size", "context_len", "context_num"]
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[A_CCNNM] parameter check wrong')
        print('[A_CCNNM] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('filters', 10)
        self.set_default('kernel_size', 3)
        self.config.update(config)

    def build(self):
        # use a convolution layer of size of the context
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['context_num']*self.config['context_len'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable = self.embed_trainable)

        q_embed = embedding(query)
        d_embed = embedding(doc)
        show_layer_info('Embedding_q', q_embed)
        show_layer_info('Embedding_intersect', d_embed)

        cross = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])  # works  #######################
        show_layer_info('Match-dot', cross)

        cross = Permute((2, 1))(cross)  # to apply the convolution on the document axis and get context weights
        show_layer_info('Permute', cross)  # (None, int, int) == (batch, doc_len, query_len)

        contxt = Conv1D(self.config['context_embed'], self.config['context_len'],
                       strides=self.config['context_len'],  # we need to get as output single vector of context weights
                       activation='relu',
                       name="conv",)(cross)
        show_layer_info('Conv1D', contxt)

        # ################################### attention weights
        attention = Dense(1)(contxt)
        attention = Activation('softmax')(attention)

        # make_flat = Lambda(lambda x: K.batch_flatten(x))  # make it flat to compute att weights
        contxt = Multiply()([contxt, attention])
        show_layer_info('Dense', cross)
        if self.config['context_embed'] > 1:  # compute context importance weights
            contxt = Bidirectional(LSTM(self.config['context_num'], return_sequences=False))(contxt)
            show_layer_info('biLSTM', contxt)
        else:
            contxt = Reshape((self.config['context_num'], ))(contxt)
            show_layer_info('reshape', contxt)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(contxt)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(contxt)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        plot_model(model, to_file='../accnnm.png', show_shapes=True, show_layer_names=True)
        return model
