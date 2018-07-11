# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanjie time:2018/6/9 0009

import numpy as np
import tensorflow as tf
from lstm_rnn_model import LstmRnnModel


class GetPoetry(object):
    def __init__(self):
        self.poetry_model = LstmRnnModel()
        self.int_to_word = self.poetry_model.poetry.int_to_word
        self.word_to_int = self.poetry_model.poetry.word_to_int

    def _to_word(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return self.int_to_word[sample]

    # 获取藏头诗
    def get_acrostic_poetry(self, number, kind):
        # 定义占位符
        input_data = tf.placeholder(dtype=tf.int32, shape=[1, None], name='input_data')
        # input_label = tf.placeholder(dtype=tf.int32, shape=[self.poetry_model.batch_size, None], name='input_label')
        # 防止过拟合
        keep_prob = tf.placeholder(dtype=tf.float32)
        # 将输入的字符索引转化成变量
        lstm_inputs = self.poetry_model.embedding_variable(input_data)
        # RNN模型
        logits, prediction, initial_state, final_state = self.poetry_model.rnn_model(lstm_inputs, 1, keep_prob)

        init = tf.global_variables_initializer()
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state('model/poetry/')
            saver.restore(sess, ckpt.model_checkpoint_path)

            # 生成诗
            poem = ''
            for i in range(number):
                flag = True
                while flag:
                    sentence = ''
                    new_state = sess.run(initial_state)
                    x = np.zeros(shape=[1, 1])
                    x[0, 0] = self.word_to_int[' ']
                    prediction_test, new_state = sess.run([prediction, final_state], feed_dict={
                        input_data: x, initial_state: new_state, keep_prob: 1
                    })
                    x = np.zeros(shape=[1, 1])
                    x[0, 0] = self.word_to_int[' ']
                    prediction_test, new_state = sess.run([prediction, final_state], feed_dict={
                        input_data: x, initial_state: new_state, keep_prob: 1
                    })
                    word = self._to_word(prediction_test)
                    # print(word)
                    sentence += word
                    while word != '。':
                        x = np.zeros(shape=[1, 1])
                        x[0, 0] = self.word_to_int[word]
                        prediction_test, new_state = sess.run([prediction, final_state], feed_dict={
                            input_data: x, initial_state: new_state, keep_prob: 1
                        })
                        word = self._to_word(prediction_test)
                        sentence += word
                        # print(sentence)
                    if len(sentence) == 2 * kind + 2:
                        sentence += '\n'
                        poem += sentence
                        flag = False
        return poem


def main():
    get_poetry = GetPoetry()
    # head = sys.argv[1]
    # size = int(sys.argv[2])
    poem = get_poetry.get_acrostic_poetry(number=20, kind=7)
    print(poem)


if __name__ == '__main__':
    main()
