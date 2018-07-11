# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanjie time:2018/5/28 0028

from lstm_rnn_model import LstmRnnModel


def main():
    poetry_model = LstmRnnModel()
    poetry_model.train(epoch=100)


if __name__ == '__main__':
    main()
