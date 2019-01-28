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


class A_CCNNM(BasicModel):
    def __init__(self, config):
        super(A_CCNNM, self).__init__(config)
        self.__name = 'A_CCNNM'
        self.check_list = ['text1_maxlen', 'text2_maxlen', 'embed', 'embed_size', 'train_embed',  'vocab_size',
                           "context_embed", "context_len", "context_num", "conv_dropout_rate", "pool_size",
                           "text1_attention", "text2_attention", "merge_levels"]
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[A_CCNNM] parameter check wrong')
        print('[A_CCNNM] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.config.update(config)

    def build(self):
        # use a convolution layer of size of the context
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('query', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('doc', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)

        q_embed = embedding(query)
        d_embed = embedding(doc)
        show_layer_info('Embedding_q', q_embed)
        show_layer_info('Embedding_intersect', d_embed)

        # ########## compute attention weights for the query words
        if self.config["text1_attention"]:
            q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
            show_layer_info('Dense', q_w)
            q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'],))(q_w)
            show_layer_info('Lambda-softmax', q_w)
            # ########## add attention weights for Q_words
            q_w_layer = Lambda(lambda x: K.repeat_elements(q_w, rep=self.config['embed_size'], axis=2))(q_w)
            show_layer_info('repeat', q_w_layer)
            q_embed = Multiply()([q_w_layer, q_embed])
            show_layer_info('Dot-qw', q_embed)
        # ####################### attention text1

        # ########## compute attention weights for the document words by contexts:
        if self.config['text2_attention']:
            if self.config['per_context']:
                # compute attention weights per context
                starts = [i for i in range(0, self.config['text2_maxlen'], self.config['context_len'])]  # slices starts
                slice_layer = [crop(1, start, start + self.config['context_len']) for start in starts]  # slicing layers
                slices = [slice_layer_i(d_embed) for slice_layer_i in slice_layer]  # create slices (contextes)
                attention_ws = []
                for slice in slices:
                    s_dw = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(slice)
                    s_dw = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['context_len'],))(s_dw)
                    attention_ws.append(s_dw)  # get attention weights per contexte
                # concat all the attention weights
                d_w = concatenate(attention_ws, 1)
            else:
                # compute attention weights over all the document
                d_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(d_embed)
                show_layer_info('Dense', d_w)
                d_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text2_maxlen'],))(d_w)
                show_layer_info('Lambda-softmax', d_w)

            # ########## add attention weights for D_words
            d_w_layer = Lambda(lambda x: K.repeat_elements(d_w, rep=self.config['embed_size'], axis=2))(d_w)
            d_embed = Multiply()([d_w_layer, d_embed])
            show_layer_info('Dot-dw', d_embed)
        # ####################### attention text2

        cross = Dot(axes=[2, 2], normalize=False)([q_embed, d_embed])  # dot_product
        show_layer_info('Match-dot', cross)

        cross = Permute((2, 1))(cross)  # to apply the convolution on the document axis and get context weights
        show_layer_info('Permute', cross)  # (None, int, int) == (batch, doc_len, query_len)

        contxt = Conv1D(self.config['context_embed'], self.config['context_len'],  # filters = context_embed
                       strides=self.config['context_len'],  # we need to get as output single vector of context weights
                       activation='relu',
                       name="conv")(cross)
        contxt = BatchNormalization()(contxt)
        contxt = Dropout(self.config['conv_dropout_rate'])(contxt)
        show_layer_info('Conv1D', contxt)

        # ################################### attention weights on passage level:
        if self.config['context_attention']:
            attention = Dense(1, use_bias=False)(contxt)
            attention = Activation('softmax')(attention)
            show_layer_info('Attention', attention)

            # make_flat = Lambda(lambda x: K.batch_flatten(x))  # make it flat to compute att weights
            contxt = Multiply()([contxt, attention])
            show_layer_info('Dot_contxt_w', contxt)

        # select most important:
        important_context = MaxPooling1D(pool_size=self.config["pool_size"], strides=self.config["pool_size"])
        contxt = important_context(contxt)
        show_layer_info('Max_pool', contxt)

        # add the word-level features:
        if self.config['merge_levels']:
            # zero padding:
            word_level = Permute((2, 1))(cross)
            word_level_padd = Lambda(lambda x: K.reshape(ZeroPadding1D((0, contxt.shape[2] -x.shape[2]))(K.reshape(x,
                                                                                                                   (-1,
                                                                                                                    x.shape[2],
                                                                                                                    x.shape[1]
                                                                                                                    ))
                                                                                                         ),
                                                         (-1, x.shape[1],
                                                          contxt.shape[2])) if x.shape[-1] < contxt.shape[-1] else x)(word_level)
            show_layer_info('word_level_padd', word_level_padd)

            contxt_padded = Lambda(lambda x: K.reshape(ZeroPadding1D((0, word_level.shape[2] -x.shape[2]))(K.reshape(x,
                                                                                                                     (-1,
                                                                                                                      x.shape[2],
                                                                                                                      x.shape[1]
                                                                                                                      ))
                                                                                                           ),
                                                       (-1, x.shape[1],
                                                        word_level.shape[2])) if x.shape[-1] < word_level.shape[-1] else x)(contxt)
            show_layer_info('contxt_padded', contxt_padded)

            contxt = Concatenate(axis=1, name="merge_levels")([word_level_padd, contxt_padded])
            show_layer_info('merge_levels', contxt)

        if self.config['context_embed'] > 1:  # compute context importance weights
            lstm_units = int(contxt.shape[1]) if self.config['merge_levels'] else self.config['context_num']
            contxt = Permute((2, 1))(contxt) if self.config['merge_levels'] else contxt
            contxt = Bidirectional(LSTM(lstm_units, return_sequences=False))(contxt)
            contxt = BatchNormalization()(contxt)
            contxt = Dropout(self.config['lstm_dropout_rate'])(contxt)
            show_layer_info('biLSTM', contxt)
        else:
            contxt = Reshape((int(contxt.shape[1]),))(contxt)
            show_layer_info('reshape', contxt)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(contxt)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(contxt)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        plot_model(model, to_file='images/accnnm_ec_'+str(self.config['context_embed'])+".png", show_shapes=True,
                   show_layer_names=True)
        return model
