# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
from keras.models import Model
from keras.layers import *
from keras.layers import Embedding
from model import BasicModel
from utils.utility import *
from layers.Match import *
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow.contrib import autograph

class CANN(BasicModel):
    def __init__(self, config):
        super(CANN, self).__init__(config)
        self.__name = 'CANN'
        self.check_list = ['text1_maxlen', 'text2_maxlen', 'embed', 'embed_size', 'train_embed',  'vocab_size',
                           "kernel_size", "filters", 'context']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        if not self.check():
            raise TypeError('[CANN] parameter check wrong')
        print('[CANN] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('context', 3)
        self.set_default('kernel_size', 3)
        self.config.update(config)

    def build(self):
        def get_model(A, t):
            """
            Establish the context extractions from the text A of word elements in t
            :param A: large text
            :param t: short text
            :return: tensor
            """
            expanded_a = tf.expand_dims(A, axis=1)  # expand A in axis 1 to compare elements in A and t with broadcast
            equal = tf.equal(expanded_a, t)  # find where A and t are equal with each other
            reduce_all = tf.reduce_all(equal, axis=2)
            where = tf.where(reduce_all)  # find the indices
            where = tf.cast(where, dtype=tf.int32)
            # find the indices to do tf.gather, if a match is found in the start or
            # end of A, then pick up the two elements after or before it, otherwise the left one and the right one
            # along with itself are used
            @autograph.convert()
            def _map_fn(x):
                if x[0] == 0:
                    return tf.range(x[0], x[0] + 3)
                elif x[0] == tf.shape(A)[0] - 1:
                    return tf.range(x[0] - 2, x[0] + 1)
                else:
                    return tf.range(x[0] - 1, x[0] + 2)
            indices = tf.map_fn(_map_fn, where, dtype=tf.int32)
            # reshape the found indices to a vector
            reshape = tf.reshape(indices, [-1])
            # gather output with found indices
            output = tf.gather(A, reshape)
            return output

        query = Input(name='query', shape=(None,))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(None,))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable = self.embed_trainable)

        intersect = Lambda(lambda x: get_model(x, query))(doc)  # computes the contextual layer  output_shape=(None,)
        show_layer_info("Intersection_context", intersect)

        q_embed = embedding(query)
        in_embed = embedding(intersect)
        show_layer_info('Embedding_q', q_embed)
        show_layer_info('Embedding_intersect', in_embed)

        # cross = Match(match_type='dot')([q_embed, in_embed])  # not working !!!!!!!!!!!!!!!!
        cross = Dot(axes=[2, 2], normalize=True)([q_embed, in_embed])  # works  #######################
        show_layer_info('Match-dot', cross)

        cross = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen']))(cross)
        show_layer_info('Reshape', cross)
        cross = Permute((2, 1))(cross)
        show_layer_info('Permute', cross)

        cross = Conv1D(self.config['filters'], self.config['kernel_size'], activation='relu', name="conv",)(cross)
        show_layer_info('Conv1D', cross)
        cross = Flatten()(cross)
        show_layer_info('Flattened', cross)

        # ################################### attention weights
        attention = Dense(1)(cross)
        attention = Activation('softmax')(attention)

        cross = Multiply()([cross, attention])
        show_layer_info('Dense', cross)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(cross)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(cross)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=out_)
        plot_model(model, to_file='../cann.png', show_shapes=True, show_layer_names=True)
        return model
