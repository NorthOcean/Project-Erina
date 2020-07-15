'''
@Author: Conghao Wong
@Date: 1970-01-01 08:00:00
@LastEditors: Conghao Wong
@LastEditTime: 2020-07-15 15:24:25
@Description: file content
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import Base_Model
from models import calculate_ADE, calculate_FDE

class FC_cycle(Base_Model):
    def __init__(self, train_info, args):
        super().__init__(train_info, args)

    def create_model(self):
        embadding = keras.layers.Dense(64)
        LSTM = keras.layers.LSTM(64)
        MLP = keras.layers.Dense(self.pred_frames * 2*self.obs_frames)

        inputs = keras.layers.Input(shape=[self.obs_frames, 2])
        output1 = embadding(inputs)
        output2 = LSTM(output1)
        output3 = MLP(output2)
        output4 = keras.layers.Reshape([self.pred_frames, 2*self.obs_frames])(output3)
        output5 = keras.layers.Dense(2)(output4)    #shape=[batch, 12, 2]

        output5_reverse = tf.reverse(output5, axis=[1])
        output6 = embadding(output5_reverse)
        output7 = LSTM(output6)  # shape=[12, 64]
        output8 = MLP(output7)
        output9 = keras.layers.Reshape([self.obs_frames, 2*self.pred_frames])(output8)
        rebuild = keras.layers.Dense(2)(output9)
        rebuild_reverse = tf.reverse(rebuild, [1])
        
        lstm = keras.Model(inputs=inputs, outputs=[output5, rebuild_reverse])

        lstm.build(input_shape=[None, self.obs_frames, 2])
        lstm_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        
        return lstm, lstm_optimizer

    def loss(self, model_output, gt, obs='null'):
        self.loss_namelist = ['ADE_t', 'rebuild_t']
        predict = model_output[0]
        rebuild = model_output[1]
        loss_ADE = calculate_ADE(predict, gt)
        loss_rebuild = calculate_ADE(rebuild, obs)
        loss_list = tf.stack([loss_ADE, loss_rebuild])
        return 1.0 * loss_ADE + 0.4 * loss_rebuild, loss_list

    def loss_eval(self, model_output, gt, obs='null'):
        self.loss_eval_namelist = ['ADE', 'FDE', 'L2_rebuild']
        predict = model_output[0]
        rebuild = model_output[1]
        loss_ADE = calculate_ADE(predict, gt).numpy()
        loss_FDE = calculate_FDE(predict, gt).numpy()
        loss_rebuild = calculate_ADE(rebuild, obs).numpy()

        return loss_ADE, loss_FDE, loss_rebuild


class SS_cycle(Base_Model):
    def __init__(self, train_info, args):
        super().__init__(train_info, args)

    def create_model(self):
        embadding = keras.layers.Dense(64)
        LSTM = keras.layers.LSTM(64, return_sequences=True)
        MLP = keras.layers.Dense(self.pred_frames * 2*self.obs_frames)

        inputs = keras.layers.Input(shape=[self.obs_frames, 2])     
        positions_embadding = embadding(inputs)
        traj_feature = LSTM(positions_embadding)
        concat_feature = tf.concat([traj_feature, positions_embadding], axis=-1)
        feature_flatten = tf.reshape(concat_feature, [-1, self.obs_frames * 64 * 2])
        feature_fc = keras.layers.Dense(self.pred_frames * 64)(feature_flatten)
        feature_reshape = tf.reshape(feature_fc, [-1, self.pred_frames, 64])
        output5 = keras.layers.Dense(2)(feature_reshape)

        output5_reverse = tf.reverse(output5, axis=[1])
        output6 = embadding(output5_reverse)
        output7 = LSTM(output6)  # shape=[12, 64]
        output8 = tf.concat([output6, output7], axis=-1)
        output9 = tf.reshape(output8, [-1, self.pred_frames * 64 * 2])
        output10 = keras.layers.Dense(self.obs_frames * 64)(output9)
        output11 = tf.reshape(output10, [-1, self.obs_frames, 64])
        rebuild = keras.layers.Dense(2)(output11)
        rebuild_reverse = tf.reverse(rebuild, [1])
        
        lstm = keras.Model(inputs=inputs, outputs=[output5, rebuild_reverse])

        lstm.build(input_shape=[None, self.obs_frames, 2])
        lstm_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        
        return lstm, lstm_optimizer

    def loss(self, model_output, gt, obs='null'):
        self.loss_namelist = ['ADE_t', 'rebuild_t']
        predict = model_output[0]
        rebuild = model_output[1]
        loss_ADE = calculate_ADE(predict, gt)
        loss_rebuild = calculate_ADE(rebuild, obs)
        loss_list = tf.stack([loss_ADE, loss_rebuild])
        return 1.0 * loss_ADE + 0.4 * loss_rebuild, loss_list

    def loss_eval(self, model_output, gt, obs='null'):
        self.loss_eval_namelist = ['ADE', 'FDE', 'L2_rebuild']
        predict = model_output[0]
        rebuild = model_output[1]
        loss_ADE = calculate_ADE(predict, gt).numpy()
        loss_FDE = calculate_FDE(predict, gt).numpy()
        loss_rebuild = calculate_ADE(rebuild, obs).numpy()

        return loss_ADE, loss_FDE, loss_rebuild


class SS_LSTM_beta(Base_Model):
    """
    `S`tate and `S`equence `LSTM`
    """
    def __init__(self, train_info, args):
        super().__init__(train_info, args)

    def create_model(self):
        positions = keras.layers.Input(shape=[self.obs_frames, 2])
        positions_embadding = keras.layers.Dense(64)(positions)
        traj_feature = keras.layers.LSTM(64, return_sequences=True)(positions_embadding)

        concat_feature = tf.concat([traj_feature, positions_embadding], axis=-1)
        feature_flatten = tf.reshape(concat_feature, [-1, self.obs_frames * 64 * 2])
        feature_fc = keras.layers.Dense(self.pred_frames * 64)(feature_flatten)
        feature_reshape = tf.reshape(feature_fc, [-1, self.pred_frames, 64])
        output5 = keras.layers.Dense(2)(feature_reshape)
        lstm = keras.Model(inputs=positions, outputs=[output5, concat_feature])

        lstm.build(input_shape=[None, self.obs_frames, 2])
        lstm_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        
        return lstm, lstm_optimizer