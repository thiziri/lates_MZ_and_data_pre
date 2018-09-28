# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
from keras.models import Model
from keras.layers import *
from keras.activations import softmax
from keras.layers import Embedding
from model import BasicModel
from utils.utility import *
from layers.Match import *
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.contrib import autograph

class A_MVLSTM_CONV(BasicModel):
    def __init__(self, config):
        super(A_MVLSTM_CONV, self).__init__(config)
        self.__name = 'A_MVLSTM'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'train_embed',  'vocab_size', "kernel_size", "filters",
                   'hidden_size', 'topk', 'dropout_rate', 'text1_attention', 'text2_attention']
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
        self.set_default('dropout_lstm', 0)
        self.set_default('kernel_size', 3)
        self.set_default('text1_attention', True)
        self.set_default('text2_attention', False)
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
            q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'],))(q_w)
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
        q_rep = BatchNormalization()(q_rep)
        q_rep = Dropout(self.config["dropout_lstm"])(q_rep)

        d_rep = Bidirectional(LSTM(self.config['hidden_size'], return_sequences=True, dropout=self.config['dropout_rate']))(d_embed)
        show_layer_info('Bidirectional-LSTM_d', d_rep)
        d_rep = BatchNormalization()(d_rep)
        d_rep = Dropout(self.config["dropout_lstm"])(d_rep)

        cross = Match(match_type='dot')([q_rep, d_rep])
        show_layer_info('Match-dot', cross)

        cov1 = Conv2D(self.config['filters'], self.config['kernel_size'], activation='relu', name="conv1",
                      padding='same')(cross)
        cov1 = BatchNormalization()(cov1)
        cov1 = Dropout(self.config["dropout_rate"])(cov1)
        show_layer_info('Conv1', cov1)

        cov1 = MaxPooling2D(pool_size=3, name="maxPool")(cov1)
        show_layer_info('MaxPooling1D-cov2', cov1)

        cov2 = Conv2D(self.config['filters'], self.config['kernel_size'], activation='relu', name="conv2",
                      padding='same')(cov1)
        cov2 = BatchNormalization()(cov2)
        cross = Dropout(self.config["dropout_rate"])(cov2)
        show_layer_info('Conv2', cov2)

        cross_reshape = Reshape((-1, ))(cross)
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
        plot_model(model, to_file='../amvlstm_conv.png', show_shapes=True, show_layer_names=True)
        return model
