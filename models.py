'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:34
@LastEditors  : ConghaoWong
@LastEditTime : 2019-12-30 11:27:09
@Description: file content
'''
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt

from helpmethods import(
    list2array,
    dir_check,
    draw_test_results,
)

class __Base_Model():
    """
    Base model for prediction.

    Following items should be given when using this model:
    ```
    self.create_model(self), # create prediction model
    self.loss(self, pred, gt),  # loss function when training model
    self.loss_eval(self, pred, gt), # loss function when test model
    self.forward_train(self, inputs, agents_train='null'), # model result in training steps
    self.forward_test(self, inputs, gt='null', agents_test='null'). # model result in test steps
    ```
    """
    def __init__(self, agents, args):
        self.args = args
        self.agents = agents
        
    def run_commands(self):
        self.initial_dataset()

        if self.args.load == 'null':
            self.model, self.optimizer = self.create_model()
            self.train(self.agents_train, self.agents_test)
        else:
            self.load_data_and_model()
            if self.args.test:
                self.agents_test = self.test(self.agents_test)
                self.draw_pred_results(self.agents_test)

    def initial_dataset(self):
        self.obs_frames = self.args.obs_frames
        self.pred_frames = self.args.pred_frames
        self.total_frames = self.obs_frames + self.pred_frames
        self.log_dir = dir_check(self.args.log_dir)

        if not self.args.load == 'null':
            return

        self.sample_number = len(self.agents)
        self.sample_time = (1 + self.args.reverse) * (1 + self.args.add_noise)
        self.sample_number_original = int(self.sample_number/self.sample_time) 

        index = set([i for i in range(self.sample_number_original)])
        train_index = random.sample(index, int(self.sample_number_original * self.args.train_percent))
        test_index = list(index - set(train_index))
        
        self.agents_train = [self.agents[(more_sample + 1) * index] for more_sample in range(self.sample_time) for index in train_index]
        self.train_index = [(more_sample + 1) * index for more_sample in range(self.sample_time) for index in train_index]
        
        self.agents_test = [self.agents[index] for index in test_index]
        self.test_index = test_index
    
    def load_data_and_model(self):
        base_path = self.args.load + '{}'
        self.model = keras.models.load_model(base_path.format('.h5'))
        self.agents_test = np.load(base_path.format('test.npy'), allow_pickle=True)
        self.test_index = np.load(base_path.format('index.npy'), allow_pickle=True)
    
    def create_model(self):
        raise 'MODEL is not defined!'
        return model, optimizer

    def loss(self, model_output, gt, obs='null'):
        self.loss_namelist = ['ADE']
        loss_ADE = calculate_ADE(model_output[0], gt)
        loss_list = tf.stack([loss_ADE])
        return loss_ADE, loss_list

    def loss_eval(self, model_output, gt, obs='null'):
        self.loss_eval_namelist = ['ADE', 'FDE']
        return calculate_ADE(model_output[0], gt).numpy(), calculate_FDE(model_output[0], gt).numpy()

    def create_loss_dict(self, loss, name_list):
        dic = {}
        for loss, name in zip(loss, name_list):
            dic[name] = loss
        return dic

    def forward(self, input_agents):
        """This method is a direct IO"""
        model_inputs = tf.cast(tf.stack([agent.traj_train for agent in input_agents]), tf.float32)
        outputs = self.model(model_inputs)
        if not type(outputs) == list:
            outputs = [outputs]
        
        pred_traj = outputs[0].numpy()
        for i in range(len(input_agents)):
            input_agents[i].pred = pred_traj[i]
            input_agents[i].pred_fix()
        return input_agents

    def forward_train(self, inputs, agents_train='null'):
        output = self.model(inputs)
        if not type(output) == list:
            output = [output]
        return output

    def __forward_train(self, input_agents):
        input_trajs = tf.cast(tf.stack([agent.traj_train for agent in input_agents]), tf.float32)
        gt = tf.cast(tf.stack([agent.traj_gt for agent in input_agents]), tf.float32)
        return self.forward_train(input_trajs, input_agents), gt, input_trajs

    def forward_test(self, inputs, gt='null', agents_test='null'):
        output = self.model(inputs)
        if not type(output) == list:
            output = [output]
        return output

    def __forward_test(self, input_agents):
        input_trajs = tf.cast(tf.stack([agent.traj_train for agent in input_agents]), tf.float32)
        gt = tf.cast(tf.stack([agent.traj_gt for agent in input_agents]), tf.float32)
        return self.forward_test(input_trajs, gt, input_agents), gt, input_trajs

    def test_step(self, input_agents):
        model_output, gt, obs = self.__forward_test(input_agents)
        loss_eval = self.loss_eval(model_output, gt, obs=obs)
        for i, output_curr in enumerate(model_output[0]):
            input_agents[i].pred = output_curr.numpy()

        return model_output, loss_eval, gt, input_agents
    
    def train(self, agents_train, agents_test):
        train_agents_number = len(agents_train)
        test_agents_number = len(agents_test)

        batch_number = int(np.ceil(train_agents_number / self.args.batch_size))
        summary_writer = tf.summary.create_file_writer(self.args.log_dir)

        if self.args.reverse:
            print('Using reverse data to train. (2x)')
        if self.args.add_noise:
            print('Using noise data to train. ({}x)'.format(self.args.add_noise))
        
        print('original_sample_number = {}, total {}x train samples.'.format(self.sample_number_original, self.sample_time))
        print('train_sample_number = {}'.format(len(self.train_index)))
        print('test_sample_number = {}'.format(len(self.test_index)))
        print('batch_number = {}\nbatch_size = {}'.format(batch_number, self.args.batch_size))
        
        print('Start training:')
        test_results = []
        for epoch in range(self.args.epochs):
            ADE = 0
            ADE_move_average = tf.cast(0.0, dtype=tf.float32)    # 计算移动平均
            loss_list = []
            for batch in range(batch_number):
                batch_start = batch * self.args.batch_size
                batch_end = tf.minimum((batch + 1) * self.args.batch_size, train_agents_number)
                agents_current = agents_train[batch_start : batch_end]
                index_current = self.train_index[batch_start : batch_end]

                with tf.GradientTape() as tape:
                    model_output_current, gt_current, obs_current = self.__forward_train(agents_current)
                    loss_ADE, loss_list_current = self.loss(model_output_current, gt_current, obs=obs_current)
                    ADE_move_average = 0.7 * loss_ADE + 0.3 * ADE_move_average

                ADE += loss_ADE
                grads = tape.gradient(ADE_move_average, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                loss_list.append(loss_list_current)
            
            loss_list = tf.reduce_mean(tf.stack(loss_list), axis=0).numpy()

            if (epoch >= self.args.start_test_percent * self.args.epochs) and (epoch % self.args.test_step == 0):
                model_output, loss_eval, _, _ = self.test_step(agents_test)
                test_results.append(loss_eval)

                with summary_writer.as_default():
                    for (loss, name) in zip(loss_eval, self.loss_eval_namelist):
                        tf.summary.scalar(name, loss, step=epoch)

                print('epoch {}/{}, train_loss_total={:.5f}, train_loss={}, test_loss={}'.format(
                    epoch + 1, 
                    self.args.epochs, 
                    ADE/batch_number, 
                    self.create_loss_dict(loss_list, self.loss_namelist),
                    self.create_loss_dict(loss_eval, self.loss_eval_namelist)
                ), flush=True, end='\r')
            
            else:
                print('epoch {}/{}, train_loss_total={:.5f}, train_loss={}'.format(
                    epoch + 1, 
                    self.args.epochs, 
                    ADE/batch_number,
                    self.create_loss_dict(loss_list, self.loss_namelist)
                ), flush=True, end='\r')

            if epoch == self.args.epochs - 1 and self.args.draw_results == True:
                self.test(agents_test)

        print('Training done.')
        latest_epochs = 10
        test_results = list2array(test_results)
        latest_results = np.mean(test_results[-latest_epochs-1:-1, :], axis=0)
        print('In latest {} test epochs, average test loss = {}'.format(
            latest_epochs,
            latest_results
        ))
        np.savetxt(os.path.join(self.args.log_dir, 'train_log.txt'), list2array(test_results))

        if self.args.save_model:
            self.model_save_path = os.path.join(self.args.log_dir, '{}.h5'.format(self.args.model_name))
            self.test_data_save_path = os.path.join(self.args.log_dir, '{}.npy'.format(self.args.model_name + '{}'))

            test_data = [
                agents_test,
                self.test_index,
            ]

            self.model.save(self.model_save_path)
            np.save(self.test_data_save_path.format('test'), test_data[0])   
            np.save(self.test_data_save_path.format('index'), test_data[1])
            print('Trained model is saved at "{}"'.format(self.model_save_path))
    
    def test(self, agents_test):
        test_agents_number = len(agents_test)
        print('Start test:')    
        model_output, loss_eval, gt_test, agents_test = self.test_step(agents_test)
        pred_traj = model_output[0].numpy()
        for i in range(len(agents_test)):
            agents_test[i].pred = pred_traj[i]

        print('test_loss={}'.format(self.create_loss_dict(loss_eval, self.loss_eval_namelist)))
        for loss in loss_eval:
            print(loss, end='\t')
        print('\nTest done.')
        return agents_test

    def draw_pred_results(self, agents):
        draw_test_results(
            agents, 
            self.log_dir, 
            loss_function=calculate_ADE_single, 
            save=self.args.draw_results
        )


class FullAttention_LSTM(__Base_Model):
    """
    LSTM based model with full attention layer.
    """
    def __init__(self, agents, args):
        super().__init__(agents, args)

    def create_model(self):
        inputs = keras.layers.Input(shape=[self.args.obs_frames, 2])
        output1 = keras.layers.Dense(64)(inputs)
        output2 = keras.layers.LSTM(64)(output1)
        output3 = keras.layers.Dense(self.args.pred_frames * 16)(output2)
        output4 = keras.layers.Reshape([self.args.pred_frames, 16])(output3)
        output5 = keras.layers.Dense(2)(output4)
        lstm = keras.Model(inputs=inputs, outputs=[output5])

        # lstm = keras.Sequential([
        #     keras.layers.Dense(64),     
        #     keras.layers.LSTM(64),  
        #     keras.layers.Dense(self.args.pred_frames * 16), 
        #     keras.layers.Reshape([self.args.pred_frames, 16]),
        #     keras.layers.Dense(2), 
        # ])

        lstm.build(input_shape=[None, self.args.obs_frames, 2])
        lstm_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        print(lstm.summary())
        return lstm, lstm_optimizer


class FC_cycle(__Base_Model):
    def __init__(self, agents, args):
        super().__init__(agents, args)

    def create_model(self):
        embadding = keras.layers.Dense(64)
        LSTM = keras.layers.LSTM(64)
        MLP = keras.layers.Dense(self.args.pred_frames * 2*self.args.obs_frames)

        inputs = keras.layers.Input(shape=[self.args.obs_frames, 2])
        output1 = embadding(inputs)
        output2 = LSTM(output1)
        output3 = MLP(output2)
        output4 = keras.layers.Reshape([self.args.pred_frames, 2*self.args.obs_frames])(output3)
        output5 = keras.layers.Dense(2)(output4)    #shape=[batch, 12, 2]

        output5_reverse = tf.reverse(output5, axis=[1])
        output6 = embadding(output5_reverse)
        output7 = LSTM(output6)  # shape=[12, 64]
        output8 = MLP(output7)
        output9 = keras.layers.Reshape([self.args.obs_frames, 2*self.args.pred_frames])(output8)
        rebuild = keras.layers.Dense(2)(output9)
        rebuild_reverse = tf.reverse(rebuild, [1])
        
        lstm = keras.Model(inputs=inputs, outputs=[output5, rebuild_reverse])

        lstm.build(input_shape=[None, self.args.obs_frames, 2])
        lstm_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        print(lstm.summary())
        return lstm, lstm_optimizer

    def loss(self, model_output, gt, obs='null'):
        self.loss_namelist = ['ADE', 'rebuild']
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



class LSTM_Social(__Base_Model):
    def __init__(self, agents, args):
        super().__init__(agents, args)

    def create_model(self):
        inputs = keras.Input(shape=[self.obs_frames, 2])
        inputs_embadding = keras.layers.Dense(64, name='embadding')(inputs)
        lstm_features = keras.layers.LSTM(64, name='lstm')(inputs_embadding)

        social_features = keras.Input(shape=[64])
        concat_features = tf.concat([lstm_features, social_features], axis=1)

        fc_features = keras.layers.Dense(self.args.pred_frames * 16)(concat_features)
        fc_features_reshape = keras.layers.Reshape([self.args.pred_frames, 16])(fc_features)

        outputs = keras.layers.Dense(2)(fc_features_reshape)
        lstm = keras.Model(inputs=[inputs, social_features], outputs=[outputs])
        lstm.build(input_shape=[[None, self.obs_frames, 2], [None, 64]])
        optimizer = keras.optimizers.Adam(lr=self.args.lr)
        print(lstm.summary())
        return lstm, optimizer

    def feature_extractor(self, inputs):
        layer_name_list = ['embadding', 'lstm']
        for layer in layer_name_list:
            layer_current = self.model.get_layer(layer)
            output = layer_current(inputs)
            inputs = output
        
        return output
    
    def forward_train(self, inputs, agents):
        batch = len(inputs)
        social_features = []

        for agents_curr in agents:
            neighbor_list_curr = agents_curr.neighbor_list_current
            social_trajs = []
            for neighbor_curr in agents_curr.neighbor_agent:
                if self.args.future_interaction:
                    traj_curr = neighbor_curr.traj_pred
                else:
                    traj_curr = neighbor_curr.traj_train
                social_trajs.append(traj_curr)

            social_features_curr = self.feature_extractor(tf.stack(social_trajs))
            social_features.append(tf.reduce_max(social_features_curr, axis=0))
        
        social_features = tf.stack(social_features)
        positions = self.model([inputs, social_features])
        return positions

    def forward_test(self, inputs, gt, agents):
        return self.forward_train(inputs, agents)


class LSTM_ED(__Base_Model):
    def __init__(self, agents, args):
        super().__init__(agents, args)
        self.fc_size = 16
        self.feature_layer = 3 + 1  # Input is also a layer
        self.output_layer = 6 + 1

    def create_model(self):
        inputs = keras.Input(shape=[self.obs_frames, 2])
        output0 = keras.layers.Dense(64, activation=tf.nn.relu)(inputs)
        output1 = keras.layers.Dropout(self.args.dropout)(output0)
        output2 = keras.layers.LSTM(128)(output1)                        # 2
        output3 = keras.layers.Dense(self.pred_frames * self.fc_size, activation=tf.nn.relu)(output2)      # 3
        output4 = keras.layers.Reshape([self.pred_frames, self.fc_size])(output3)   # 4
        output5 = keras.layers.LSTM(64, return_sequences=True)(output4)   # 5
        output6 = keras.layers.Dense(2)(output5)

        position_output = output6
        feature_output = output3

        ae = keras.Model(inputs=[inputs], outputs=[position_output, feature_output])
        ae.build(input_shape=[None, self.obs_frames, 2])
        ae_optimizer = keras.optimizers.Adam(lr=self.args.lr)
        print(ae.summary())
        return ae, ae_optimizer
    
    def forward_test(self, inputs_test, groundtruth_test, agents_test='null'):
        batch = inputs_test.shape[0]
        self.noise_mean = 0.0
        self.noise_sigma = 0.1

        test_results = []
        for re in range(self.args.k):
            print('Repeat step {}/{}...\t'.format(re, self.args.k), end='\r')
            noise = np.random.normal(self.noise_mean, self.noise_sigma, size=[batch, self.pred_frames * self.fc_size])
            features = get_model_outputs(self.model, inputs_test, input_layer=0, output_layer=self.feature_layer)
            position_output = get_model_outputs(self.model, features + noise, input_layer=self.feature_layer+1, output_layer=self.output_layer)
            test_results.append(position_output.numpy())
        print('Generate done.')

        self.loss_eval_namelist = ['ADE', 'FDE', 'mADE', 'mFDE', 'sA', 'sF', 'GP_ADE', 'GP_FDE']
        test_results = np.transpose(list2array(test_results), axes=[1, 0, 2, 3])
        return [tf.cast(test_results, tf.float32)]
    
    def loss(self, model_output, gt, obs='null'):
        self.loss_namelist = ['ADE', 'smooth']
        loss_ADE = calculate_ADE(model_output[0], gt)
        loss_smoothness = smooth_loss(obs, model_output[0], step=1)
        loss_list = tf.stack([loss_ADE, loss_smoothness])
        return 1.0 * loss_ADE + 0.0 * loss_smoothness, loss_list
    
    def loss_eval(self, model_output, gt, obs='null'):
        self.loss_eval_namelist = ['ADE', 'FDE', 'mADE', 'mFDE', 'sA', 'sF', 'GP_ADE', 'GP_FDE', 'smooth']
        return self.choose_best_path(model_output[0], gt)[1]
    
    def choose_best_path(self, positions, groundtruth, choose='best'):
        positions = tf.cast(positions, tf.float32)
        groundtruth = tf.cast(groundtruth, tf.float32)
        positions = tf.transpose(positions, [1, 0, 2, 3])

        all_ADE = tf.reduce_mean(tf.linalg.norm(positions - groundtruth, ord=2, axis=3), axis=2)
        all_FDE = tf.linalg.norm(positions[:, :, -1, :] - groundtruth[:, -1, :], ord=2, axis=2)

        mean_traj = tf.reduce_mean(positions, axis=0)
        GP_ADE = tf.reduce_mean(tf.linalg.norm(mean_traj - groundtruth, ord=2, axis=2), axis=1)
        GP_FDE = tf.linalg.norm(mean_traj[:, -1, :] - groundtruth[:, -1, :], ord=2, axis=1)

        mean_ADE = tf.reduce_mean(all_ADE, axis=0)
        mean_FDE = tf.reduce_mean(all_FDE, axis=0)

        s_ADE = tf.math.reduce_std(all_ADE, axis=0)
        s_FDE = tf.math.reduce_std(all_FDE, axis=0)

        minADE_index = tf.cast(tf.argmin(all_ADE, axis=0), tf.int32)
        minADE_index = tf.transpose(tf.stack([
            tf.stack([i for i in range(minADE_index.shape[0])]),
            minADE_index,
        ]))
        positions = tf.transpose(positions, [1, 0, 2, 3])
        traj = tf.gather_nd(positions, minADE_index)
        ADE = tf.gather_nd(tf.transpose(all_ADE), minADE_index)
        FDE = tf.gather_nd(tf.transpose(all_FDE), minADE_index)

        result = traj
        ADE_FDE = tf.stack([
            ADE, FDE, mean_ADE, mean_FDE, s_ADE, s_FDE, GP_ADE, GP_FDE
        ])
        ADE_FDE = tf.reduce_mean(ADE_FDE, axis=1)

        if self.args.save_k_results:
            save_format = os.path.join(self.args.log_dir, '{}.npy')
            np.save(save_format.format('best'), result)
            np.save(save_format.format('all'), positions)
            np.save(save_format.format('gt'), groundtruth)
            np.save(save_format.format('mean'), mean_traj)

        return result, ADE_FDE.numpy()


class Linear(__Base_Model):
    def __init__(self, agents, args):
        super().__init__(agents, args)
        self.args.batch_size = 1
        self.args.epochs = 1
        self.args.draw_results = False
        self.args.train_percent = 0.0
    
    def run_commands(self):
        self.initial_dataset()
        self.model, self.optimizer = self.create_model()
        self.test(self.agents_test)

    def predict_linear(self, x, y, x_p, diff_weights=0):
        if diff_weights == 0:
            P = np.diag(np.ones(shape=[x.shape[0]]))
        else:
            P = np.diag(softmax([(i+1)**diff_weights for i in range(x.shape[0])]))

        A = tf.transpose(tf.stack([np.ones_like(x), x]))
        A_p = tf.transpose(tf.stack([np.ones_like(x_p), x_p]))
        Y = tf.transpose(y)

        P = tf.cast(P, tf.float32)
        A = tf.cast(A, tf.float32)
        A_p = tf.cast(A_p, tf.float32)
        Y = tf.reshape(tf.cast(Y, tf.float32), [-1, 1])
        
        B = tf.matmul(tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.matmul(tf.transpose(A), P), A)), tf.transpose(A)), P), Y)
        Y_p = np.matmul(A_p, B)
        return Y_p, B

    def predict_linear_for_person(self, positions, diff_weights):
        t = np.array([t for t in range(self.obs_frames)])
        t_p = np.array([t + self.obs_frames for t in range(self.pred_frames)])
        x = tf.transpose(positions)[0]
        y = tf.transpose(positions)[1]

        x_p, _ = self.predict_linear(t, x, t_p, diff_weights=diff_weights)
        y_p, _ = self.predict_linear(t, y, t_p, diff_weights=diff_weights)

        return tf.transpose(tf.reshape(tf.stack([x_p, y_p]), [2, self.pred_frames]))
    
    def create_model(self):
        return self.predict_linear_for_person, 0

    def forward_train(self, inputs, agents_train='null'):
        results = []
        for inputs_current in inputs:
            results.append(self.model(inputs_current, diff_weights=self.args.diff_weights))
        
        return [tf.stack(results)]

    def forward_test(self, inputs, gt='null', agents_train='null'):
        results = []
        for inputs_current in inputs:
            results.append(self.model(inputs_current, diff_weights=self.args.diff_weights))
        
        return [tf.stack(results)]

    

"""
helpmethods
"""

def draw_one_traj(traj, GT, save_path):
    plt.figure()

    plt.plot(traj.T[0], traj.T[1], '-*')
    plt.plot(GT.T[0], GT.T[1], '-o')
    plt.axis('scaled')

    plt.savefig(save_path)
    plt.close()


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def calculate_ADE_single(pred, GT):
    """input_shape = [pred_frames, 2]"""
    if not len(pred.shape) == 3:
        pred = tf.reshape(pred, [1, pred.shape[0], pred.shape[1]])
    
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)
    return tf.reduce_mean(tf.linalg.norm(pred - GT, ord=2, axis=2))


def calculate_ADE(pred, GT):
    """input_shape = [batch, pred_frames, 2]"""
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)
    return tf.reduce_mean(tf.linalg.norm(pred - GT, ord=2, axis=2))
    

def calculate_FDE(pred, GT):
    pred = tf.cast(pred, tf.float32)
    GT = tf.cast(GT, tf.float32)
    return tf.reduce_mean(tf.linalg.norm(pred[:, -1, :] - GT[:, -1, :], ord=2, axis=1))


def get_model_outputs(model, inputs, input_layer=0, output_layer=1):
    inn = inputs
    for i in range(input_layer, output_layer+1):
        layer = model.get_layer(index=i)
        output = layer(inn)
        inn = output
    return output


def smooth_loss(obs, pred, step=2, return_min=True, as_loss=True):
    """
    shape for inputs:
    
    `obs`: [batch, obs_frames, 2]
    `pred`: [batch, pred_frames, 2]
    """
    start_point = obs[:, -step:, :]
    pred_frames = pred.shape[1]

    score_list = []
    delta = []
    for frame in range(step):
        delta.append(pred[:, frame] - start_point[:, frame])

    for frame in range(step, pred_frames):
        delta.append(pred[:, frame] - pred[:, frame-step])
    
    delta = tf.transpose(tf.stack(delta), [1, 0, 2])
    cosine_list = tf.stack([calculate_cosine(delta[:, frame-1, :], delta[:, frame, :]) for frame in range(1, pred_frames)])

    if return_min:
        cosine_list = tf.reduce_min(cosine_list, axis=0)
    
    if as_loss:
        cosine_list = tf.reduce_mean(1.0 - cosine_list)
    
    return cosine_list


def calculate_cosine(p1, p2, absolute=True):
    """
    shape for inputs:
    
    `p1`: [batch, 2]
    `p2`: [batch, 2]
    """
    l_p1 = tf.linalg.norm(p1, axis=1)
    l_p2 = tf.linalg.norm(p2, axis=1)
    dot = tf.reduce_sum(p1 * p2, axis=1)
    result = dot/(l_p1 * l_p2)
    if absolute:
        result = tf.abs(result)
    return result
    

