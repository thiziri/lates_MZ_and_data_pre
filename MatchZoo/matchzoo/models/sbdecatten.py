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
from layers.BiLSTM import BiLSTM
from layers.MultiPerspectiveMatch import MultiPerspectiveMatch
#from layers.Attention import MultiPerspectiveAttention
from layers.SequenceMask import SequenceMask
from utils.utility import *
from keras.activations import softmax


class SBDecAtten(BasicModel):
    """implementation of a siamese decomposeable attention
        
    """
    def __init__(self, config):
        super(SBDecAtten, self).__init__(config)
        self.__name = 'SBDecAtten'
        self.check_list = ['text1_maxlen', 'text2_maxlen',
                           'embed', 'embed_size', 'vocab_size',
                           'text1_attention', 'text2_attention',
                            'dropout_rate']
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)  # attention init
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[SBDecAtten] parameter check wrong')
        print('[SBDecAtten] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.set_default('dropout_rate', 0)
        self.set_default('text1_attention', False)
        self.set_default('text2_attention', False)
        self.config.update(config)

    def build(self):
        self.projection_dim=300
        self.compare_dim=300
        self.compare_dropout=0.2
        self.projection_hidden = 0
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        leaks_input = Input(name='leaks_input', shape=(1,))
        show_layer_info('Input', leaks_input)
        leaks_dense = Dense(int(self.config['number_dense_units']/2), activation='relu')(leaks_input)
        show_layer_info('Dense', leaks_dense)
        
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        # ########## compute attention weights for the query words: better then mvlstm alone
        if self.config["text1_attention"]:
            q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(
                q_embed)  # use_bias=False to simple combination
            show_layer_info('Dense', q_w)
            q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'],))(q_w)
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

        # Projection
        projection_layers = []
        if self.projection_hidden > 0:
            projection_layers.extend([
                    Dense(self.projection_hidden, activation='elu'),
                    Dropout(rate=self.config['rate_drop_dense']),
                ])
        projection_layers.extend([
                Dense(self.projection_dim, activation=None),
                Dropout(rate=self.config['rate_drop_dense']),
            ])
        q1_encoded = self.time_distributed(q_embed, projection_layers)
        q2_encoded = self.time_distributed(d_embed, projection_layers)

        # Attention
        q1_aligned, q2_aligned = self.soft_attention_alignment(q1_encoded, q2_encoded)    

        # Compare
        q1_combined = Concatenate()([q1_encoded, q2_aligned, self.submult(q1_encoded, q2_aligned)])
        q2_combined = Concatenate()([q2_encoded, q1_aligned, self.submult(q2_encoded, q1_aligned)]) 
        compare_layers = [
            Dense(self.compare_dim, activation='elu'),
            Dropout(self.compare_dropout),
            Dense(self.compare_dim, activation='elu'),
            Dropout(self.compare_dropout),
        ]
        q1_compare = self.time_distributed(q1_combined, compare_layers)
        q2_compare = self.time_distributed(q2_combined, compare_layers)

        # Aggregate
        q1_rep = self.apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
        q2_rep = self.apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])
        dense = BatchNormalization()(merged)
        dense = Dense(self.config['number_dense_units'], activation='elu')(dense)
        dense = Dropout(self.config['rate_drop_dense'])(dense)
        dense = BatchNormalization()(dense)
        dense = Dense(self.config['number_dense_units'], activation='elu')(dense)
        dense = Dropout(self.config['rate_drop_dense'])(dense)



        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(dense)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1, activation='sigmoid')(dense)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        return model

    def unchanged_shape(self, input_shape):
        "Function for Lambda layer"
        return input_shape


    def substract(self, input_1, input_2):
        "Substract element-wise"
        neg_input_2 = Lambda(lambda x: -x, output_shape=self.unchanged_shape)(input_2)
        out_ = Add()([input_1, neg_input_2])
        return out_


    def submult(self, input_1, input_2):
        "Get multiplication and subtraction then concatenate results"
        mult = Multiply()([input_1, input_2])
        sub = self.substract(input_1, input_2)
        out_= Concatenate()([sub, mult])
        return out_


    def apply_multiple(self, input_, layers):
        "Apply layers to input then concatenate result"
        if not len(layers) > 1:
            raise ValueError('Layers list should contain more than 1 layer')
        else:
            agg_ = []
            for layer in layers:
                agg_.append(layer(input_))
            out_ = Concatenate()(agg_)
        return out_


    def time_distributed(self, input_, layers):
        "Apply a list of layers in TimeDistributed mode"
        out_ = []
        node_ = input_
        for layer_ in layers:
            node_ = TimeDistributed(layer_)(node_)
        out_ = node_
        return out_


    def soft_attention_alignment(self, input_1, input_2):
        "Align text representation with neural soft attention"
        attention = Dot(axes=-1)([input_1, input_2])
        w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                         output_shape=self.unchanged_shape)(attention)
        w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                                 output_shape=self.unchanged_shape)(attention))
        in1_aligned = Dot(axes=1)([w_att_1, input_1])
        in2_aligned = Dot(axes=1)([w_att_2, input_2])
        return in1_aligned, in2_aligned
