# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from model import BasicModel
import tensorflow as tf
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


class DUET(BasicModel):
    def __init__(self, config):
        super(DUET, self).__init__(config)
        self.__name = 'DUET'
        self.check_list = ['text1_maxlen', 'text2_maxlen', 'embed_size', 'lm_kernel_count', 'lm_hidden_sizes',
                           'dm_kernel_count', 'dm_kernel_size', 'dm_q_hidden_size',  'dm_hidden_sizes',
                           'lm_dropout_rate', 'dm_dropout_rate', 'text1_attention', 'text2_attention', "context_len",
                           "context_num", "passage_attention_local", "passage_attention_distributed"]
        self.embed_trainable = config['train_embed']
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)  # attention init
        self.setup(config)
        if not self.check():
            raise TypeError('[DUET] parameter check wrong')
        print('[DUET] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('lm_kernel_count', 32)
        self.set_default('lm_hidden_sizes', [100])
        self.set_default('dm_kernel_count', 32)
        self.set_default('dm_kernel_size', 3)
        self.set_default('dm_q_hidden_size', 100)
        self.set_default('dm_hidden_sizes', [100, 100])
        self.set_default('lm_dropout_rate', 0.8)
        self.set_default('dm_dropout_rate', 0.8)
        self.config.update(config)

    def build(self):
        def xor_match(x):
            t1 = x[0]
            t2 = x[1]
            t1_shape = t1.get_shape()
            t2_shape = t2.get_shape()
            t1_expand = K.tf.stack([t1] * t2_shape[1], 2)
            t2_expand = K.tf.stack([t2] * t1_shape[1], 1)
            out_bool = K.tf.equal(t1_expand, t2_expand)
            out = K.tf.cast(out_bool, K.tf.float32)
            return out
        def hadamard_dot(x):
            x1 = x[0]
            x2 = x[1]
            out = x1 * x2
            #out = tf.matmul(x1, x2)
            #out = K.tf.einsum('ij, ijk -> jk', x1, x2)
            return out
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
            q_embed = Multiply(name="dot")([q_w_layer, q_embed])
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

        lm_xor = Lambda(xor_match)([query, doc])
        show_layer_info('XOR', lm_xor)
        # passages attention weights in the local model
        if self.config["passage_attention_local"]:
            # ########################## compute the passages attention weights
            p_cross = Permute((2, 1))(lm_xor)
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
            lm_xor = Multiply()([d_w, p_cross])
            show_layer_info('Multiply', lm_xor)
            lm_xor = Permute((2, 1))(lm_xor)
            show_layer_info('lm_xor', lm_xor)
            # ########################## passages attention

        #lm_xor_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(lm_xor)
        #show_layer_info('Reshape', lm_xor_reshape)
        lm_conv = Conv1D(self.config['lm_kernel_count'],self.config['text2_maxlen'], padding='same', activation='tanh')(lm_xor)
        show_layer_info('Conv1D', lm_conv)
        lm_conv = Dropout(self.config['lm_dropout_rate'])(lm_conv)
        show_layer_info('Dropout', lm_conv)
        lm_feat = Reshape((-1,))(lm_conv)
        show_layer_info('Reshape', lm_feat)
        for hidden_size in self.config['lm_hidden_sizes']:
            lm_feat = Dense(hidden_size, activation='tanh')(lm_feat)
            show_layer_info('Dense', lm_feat)
        lm_drop = Dropout(self.config['lm_dropout_rate'])(lm_feat)
        show_layer_info('Dropout', lm_drop)
        lm_score = Dense(1)(lm_drop)
        show_layer_info('Dense', lm_score)

        dm_q_conv = Conv1D(self.config['dm_kernel_count'], self.config['dm_kernel_size'], padding='same', activation='tanh')(q_embed)
        show_layer_info('Conv1D', dm_q_conv)
        dm_q_conv = Dropout(self.config['dm_dropout_rate'])(dm_q_conv)
        show_layer_info('Dropout', dm_q_conv)
        dm_q_mp = MaxPooling1D(pool_size = self.config['text1_maxlen'])(dm_q_conv)
        show_layer_info('MaxPooling1D', dm_q_mp)
        dm_q_rep = Reshape((-1,))(dm_q_mp)
        show_layer_info('Reshape', dm_q_rep)
        dm_q_rep = Dense(self.config['dm_q_hidden_size'])(dm_q_rep)
        show_layer_info('Dense', dm_q_rep)
        dm_q_rep = Lambda(lambda x: tf.expand_dims(x, 1))(dm_q_rep)

        dm_d_conv1 = Conv1D(self.config['dm_kernel_count'], self.config['dm_kernel_size'], padding='same', activation='tanh')(d_embed)
        show_layer_info('Conv1D', dm_d_conv1)
        dm_d_conv1 = Dropout(self.config['dm_dropout_rate'])(dm_d_conv1)
        show_layer_info('Dropout', dm_d_conv1)
        dm_d_mp = MaxPooling1D(pool_size = self.config['dm_d_mpool'])(dm_d_conv1)
        show_layer_info('MaxPooling1D', dm_d_mp)
        dm_d_conv2 = Conv1D(self.config['dm_kernel_count'], 1, padding='same', activation='tanh')(dm_d_mp)
        show_layer_info('Conv1D', dm_d_conv2)
        dm_d_conv2 = Dropout(self.config['dm_dropout_rate'])(dm_d_conv2)
        show_layer_info('Dropout', dm_d_conv2)

        h_dot = Lambda(hadamard_dot)([dm_q_rep, dm_d_conv2])
        show_layer_info('HadamarDot', h_dot)
        # compute passages attention in the distributed model
        if self.config["passage_attention_distributed"]:
            # ########################## compute the passages attention weights
            p_cross = h_dot  # Permute((2, 1))(h_dot)
            show_layer_info('p_cross', p_cross)
            starts = [i for i in range(0, int(h_dot.shape[1]), self.config['context_len'])]
            slice_layer = [crop(1, start, start + self.config['context_len']) for start in starts]
            slices = [slice_layer_i(p_cross) for slice_layer_i in slice_layer]
            attention_ws = []
            for slice in slices:
                s_dw = Dense(1, use_bias=False)(slice)
                s_dw = Lambda(lambda x: softmax(x, axis=1))(s_dw)
                attention_ws.append(s_dw)
            d_w = concatenate(attention_ws, 1)
            show_layer_info('attW', d_w)
            lm_xor = Multiply()([d_w, p_cross])
            show_layer_info('Multiply', lm_xor)
            lm_xor = Permute((2, 1))(lm_xor)
            show_layer_info('lm_xor', lm_xor)
            # ########################## passages attention
        dm_feat = Reshape((-1,))(h_dot)
        show_layer_info('Reshape', dm_feat)
        for hidden_size in self.config['dm_hidden_sizes']:
            dm_feat = Dense(hidden_size)(dm_feat)
            show_layer_info('Dense', dm_feat)
        dm_feat_drop = Dropout(self.config['dm_dropout_rate'])(dm_feat)
        show_layer_info('Dropout', dm_feat_drop)
        dm_score = Dense(1)(dm_feat_drop)
        show_layer_info('Dense', dm_score)
        out_ = Add()([lm_score, dm_score])
        show_layer_info('Add', out_)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
            show_layer_info('Dense', out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
