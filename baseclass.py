# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanjie time:2018/6/9 0009

import collections
import numpy as np


class BaseClass(object):
    def __init__(self, file_path):
        # 每批次训练 64
        self.batch_size = 64
        self.file_path = file_path
        self.contents = self._get_content()
        self.contents_len = len(self.contents)
        self.word_to_int, self.int_to_word, self.words_len = self._count_every_word()
        self.content_to_vector = self._content_to_vector()
        self.batch_num = self.contents_len // self.batch_size

    # 获取文件内容
    def _get_content(self):
        content = []
        return content

    # 按获取的内容的字数排序
    @staticmethod
    def _sort_content_by_len(contents):
        contents = sorted(contents, key=lambda line: len(line))
        return contents

    # 统计每个字出现的次数，出现次数过的排在前面，列如如下字典
    # word_to_int：{'，': 0, '。': 1, '不': 2, '人': 3, '山': 4, '风': 5, '日': 6......} 生成这样的字典
    # int_to_word：{0: '，', 1: '。', 2: '不', 3: '人', 4: '山', 5: '风', 6: '日'.....}
    def _count_every_word(self):
        all_words = ''.join(self.contents) + ' '
        words_len = len(set(all_words))
        counter = collections.Counter(all_words)
        counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words, num = zip(*counter)
        word_to_int = dict(zip(words, range(len(words))))
        int_to_word = dict(zip(range(len(words)), words))
        return word_to_int, int_to_word, words_len

    # 将所有的诗转化成向量
    def _content_to_vector(self):
        to_vector = lambda word: self.word_to_int[word]
        content_to_vector = [list(map(to_vector, content)) for content in self.contents]
        return content_to_vector

    # 生成器
    def batch(self):
        start_index = 0
        end_index = self.batch_size
        for i in range(self.batch_num):
            batches = self.content_to_vector[start_index:end_index]
            # 输入数据，按每块最大长度初始化数组，填充值用 self.word_to_int[' ']代替
            input_batch = np.full(
                shape=[self.batch_size, max(map(len, batches))], fill_value=self.word_to_int[' '], dtype=np.int32
            )
            for row in range(self.batch_size):
                input_batch[row, :len(batches[row])] = batches[row]
            output_batch = np.copy(input_batch)
            output_batch[:, :-1], output_batch[:, -1] = input_batch[:, 1:], input_batch[:, 0]
            yield input_batch, output_batch
            start_index = start_index + self.batch_size
            end_index = end_index + self.batch_size
