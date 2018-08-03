# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
from keras.models import Model
from keras.layers import *
from model import BasicModel
from utils.utility import *
from layers.Match import *
from keras.utils.vis_utils import plot_model

class ConvWeakCollaboration(BasicModel):
    def __init__(self, config):
        super(ConvWeakCollaboration, self).__init__(config)
        self._name = 'WeakCollaboration'
        self.check_list = ['number_q_lstm_units', 'number_d_lstm_units', 'q_lstm_dropout', 'd_lstm_dropout', 'embed',
                           'embed_size', 'vocab_size', 'num_layers', 'hidden_sizes']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[WeakCollaboration] parameter check wrong')
        print('[WeakCollaboration] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('dropout_rate', 0.)
        self.set_default('q_lstm_dropout', 0.)
        self.set_default('d_lstm_dropout', 0.)
        self.set_default('mask_zero', False)
        self.config.update(config)

    def build(self):
        query = Input(name="query", batch_shape=[None, None], dtype='int32')
        show_layer_info('Input', query)
        doc = Input(name="doc", batch_shape=[None, None], dtype='int32')
        show_layer_info('Input', doc)

        input_embed = self.config['vocab_size'] if self.config['mask_zero'] else self.config['vocab_size']
        embedding = Embedding(input_embed, self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable, name="embeddings",
                              mask_zero=self.config['mask_zero'])
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        q_lstm_layer = Bidirectional(LSTM(self.config["number_q_lstm_units"],
                                          dropout=self.config["q_lstm_dropout"],
                                          recurrent_dropout=self.config["q_lstm_dropout"],
                                          return_sequences=True),
                                     name="q_lstm")
        d_lstm_layer = Bidirectional(LSTM(self.config["number_d_lstm_units"],
                                          dropout=self.config["d_lstm_dropout"],
                                          recurrent_dropout=self.config["d_lstm_dropout"],
                                          return_sequences=True),
                                     name="d_lstm")
        q_mat = q_lstm_layer(q_embed)
        show_layer_info('Bibirectional-LSTM', q_mat)
        d_mat = d_lstm_layer(d_embed)
        show_layer_info('Bibirectional-LSTM', d_mat)
        input_mat = Match(normalize=True)([q_mat, d_mat])  # the result is cosine similarity matrix
        show_layer_info('Match', input_mat)
        #input_mat = BatchNormalization()(input_mat)
        #input_mat = Dropout(self.config["dropout_rate"])(input_mat)
        input_mat = Reshape((self.config["text1_maxlen"], self.config["text2_maxlen"]))(input_mat)
        show_layer_info('Match', input_mat)
        merged = Conv1D(self.config['filters'], self.config['kernel_size'],
                        activation=self.config['conv_activation'], name="conv1", padding='same')(input_mat)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.config["conv_dropout"])(merged)
        show_layer_info('Conv1D', merged)
        merged = MaxPooling1D(pool_size=self.config['pool_size'], name="maxPool1")(merged)
        show_layer_info('MaxPooling1D', merged)
        merged = Conv1D(self.config['filters'], self.config['kernel_size'],
                        activation=self.config['conv_activation'], name="conv2", padding='same')(input_mat)
        show_layer_info('Conv1D', merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.config["conv_dropout"])(merged)
        merged = MaxPooling1D(pool_size=self.config['pool_size'], name="maxPool2")(merged)
        show_layer_info('MaxPooling1D', merged)
        merged = Conv1D(self.config['filters'], self.config['kernel_size'],
                        activation=self.config['conv_activation'], name="conv3", padding='same')(input_mat)
        show_layer_info('Conv1D', merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.config["conv_dropout"])(merged)
        merged = MaxPooling1D(pool_size=self.config['pool_size'], name="maxPool3")(merged)
        show_layer_info('MaxPooling1D', merged)
        merged = Flatten()(merged)

        dense = Dense(self.config["hidden_sizes"][0], activation=self.config['hidden_activation'],
                      name="MLP_combine_0")(merged)
        show_layer_info('Dense', dense)
        for i in range(self.config["num_layers"] - 1):
            dense = BatchNormalization()(dense)
            dense = Dropout(self.config["dropout_rate"])(dense)
            dense = Dense(self.config["hidden_sizes"][i + 1], activation=self.config['hidden_activation'],
                          name="MLP_combine_" + str(i + 1))(dense)
            show_layer_info('Dense', dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(self.config["dropout_rate"])(dense)
        out_ = Dense(1, activation=self.config['output_activation'], name="MLP_out")(dense)
        show_layer_info('Output', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        plot_model(model, to_file='../conv_wc_model_plot.png', show_shapes=True, show_layer_names=True)
        return model