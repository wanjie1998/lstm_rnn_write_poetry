# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanjie time:2018/5/27 0027

import tensorflow as tf
import datetime
from poetry import Poetry


class LstmRnnModel(object):
    def __init__(self):
        # 诗歌生成
        self.poetry = Poetry('data/poetry.txt')
        # 每批次训练多少首诗
        self.batch_size = self.poetry.batch_size
        # 所有出现的字符的数量
        self.word_len = self.poetry.words_len
        # lstm神经网络中间层神经元的个数
        self.lstm_size = 128

    # 将输入的字符索引变成向量
    def embedding_variable(self, inputs):
        # embedding = tf.Variable(tf.truncated_normal(shape=[self.word_len, self.lstm_cell], stddev=0.1))
        embedding = tf.get_variable(name='embedding', shape=[self.word_len, self.lstm_size])
        lstm_inputs = tf.nn.embedding_lookup(embedding, inputs)
        return lstm_inputs

    # 定义lstm网络中的权重矩阵和偏移矩阵
    def _weight_baise_variable(self):
        w = tf.get_variable(name='w', shape=[self.lstm_size, self.word_len])
        b = tf.get_variable(name='b', shape=[1, self.word_len])
        return w, b

    # 定义lstm RNN模型
    def rnn_model(self, lstm_inputs, batch_size, keep_prob):
        # lstm 基本的cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_size, forget_bias=1.0, state_is_tuple=True)
        print('lstm_cell:', lstm_cell.state_size)
        # 防止过拟合
        drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=keep_prob)
        # 多层cell
        cell = tf.nn.rnn_cell.MultiRNNCell([drop] * 2)
        print('cell:', cell.state_size)
        # 初始化记忆
        initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        print('initial_state:', initial_state)
        # 使用lstm网络
        # print(lstm_inputs)
        # print(cell.state_size)
        # print(initial_state)
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_inputs, initial_state=initial_state)
        print('lstm_outputs:', lstm_outputs)
        print('final_state:', final_state)
        # print('final_state:', final_state)
        # 赋值
        # seq_output = tf.concat(lstm_outputs, 1)
        seq_output = lstm_outputs
        x = tf.reshape(seq_output, shape=[-1, self.lstm_size])
        w, b = self._weight_baise_variable()
        logits = tf.matmul(x, w) + b
        prediction = tf.nn.softmax(logits, name='prediction')
        return logits, prediction, initial_state, final_state

    # 定义损失函数
    @staticmethod
    def _loss_model(input_label, logits):
        # 这种方法尝试过不太好
        # # 将input_label 先转成 one_hot 编码
        # input_label_one_hot = tf.one_hot(input_label, depth=self.word_len)
        # input_label_one_hot_reshape = tf.reshape(input_label_one_hot, shape=[-1, self.word_len])
        # loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label_one_hot_reshape, logits=logits)
        # )
        # return loss
        input_label_reshape = tf.reshape(input_label, shape=[-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], [input_label_reshape], [tf.ones_like(input_label_reshape, dtype=tf.float32)],
            # average_across_timesteps=True,
            # softmax_loss_function=self.word_len,
            # name='loss'
        )
        loss = tf.reduce_mean(loss)
        return loss

    # 定义优化器
    # RNN会遇到梯度爆炸（gradients exploding）和梯度弥散（gradients disappearing)的问题。
    # LSTM解决了梯度弥散的问题，但是gradients仍然可能会爆炸，因此我们采用gradient clippling的方式来防止梯度爆炸。
    # 即通过设置一个阈值，当gradients超过这个阈值时，就将它重置为阈值大小，这就保证了梯度不会变得很大。
    @staticmethod
    def _optimize_model(loss, learning_rate):
        grad_clip = 5
        # 使用clipping gradients 计算梯度，并防止梯度爆炸
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_variables), grad_clip)
        # 创建优化器，进行反向传播
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, trainable_variables))
        return optimizer

    # 训练停止接着训练
    @staticmethod
    def _load_model(sess, saver, ckpt_path):
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if latest_ckpt:
            print('resume from', latest_ckpt)
            saver.restore(sess, latest_ckpt)
            return int(latest_ckpt[latest_ckpt.rindex('-') + 1:])
        else:
            print('building model from scratch')
            sess.run(tf.global_variables_initializer())
            return -1

    def train(self, epoch):
        # 定义占位符
        input_data = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name='input_data')
        input_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None], name='input_label')
        # 防止过拟合
        keep_prob = tf.placeholder(dtype=tf.float32)

        # 将输入的字符索引转化成变量
        lstm_inputs = self.embedding_variable(input_data)
        print('lstm_inputs:', lstm_inputs)
        # lstm_rnn模型
        logits, prediction, initial_state, final_state = self.rnn_model(
            lstm_inputs, self.batch_size, keep_prob)
        print('logits:', logits)
        print('prediction:', prediction)
        # 定义损失函数
        loss = self._loss_model(input_label, logits)
        # 定义学习率
        learning_rate = tf.Variable(0.0, trainable=False)
        # 定义优化器 Optimizer
        optimizer = self._optimize_model(loss, learning_rate)

        # 开始训练
        init = tf.global_variables_initializer()
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            # 初始化saver
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables())
            last_epoch = self._load_model(sess, saver, 'model/poetry/')

            step = 0
            # all_loss = 0
            new_state = sess.run(initial_state)
            for i in range(last_epoch+1, epoch):
                # 数据生成器
                batches = self.poetry.batch()
                # 随着模型的训练降低学习率
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** i)))
                for train_data, train_label in batches:
                    feed = {input_data: train_data, input_label: train_label, initial_state: new_state, keep_prob: 0.5}
                    optimizer_test, loss_test, new_state = sess.run([optimizer, loss, final_state], feed_dict=feed)
                    print(datetime.datetime.now().strftime('%c'), '训练轮数：', i, '训练次数：',
                          step, '损失值：', loss_test, '学习效率：', sess.run(learning_rate))
                    step += 1
                    # all_loss += loss_test
                    # if step % 20 == 0:
                    #     avg_loss = all_loss/20
                    #     print('经过 %d 次训练 平均损失值为 %f' % (step, avg_loss))
                    #     all_loss = 0
                saver.save(sess, 'model/poetry/poetry.module', global_step=i)


# def main():
#     poetry_model = PoetryModel()
#     poetry_model.train(epoch=20)
#
#
# if __name__ == '__main__':
#     main()
