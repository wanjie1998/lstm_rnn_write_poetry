# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanjie time:2018/6/9 0009

from baseclass import BaseClass


class English(BaseClass):
    # 继承父类的所有属性
    def __init__(self, file_path):
        BaseClass.__init__(self, file_path)

    # 覆写父类方法
    def _get_content(self):
        # english = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # for line in f:
            #     english.append(line)
            # 列表生成式
            english = [line for line in f]
        return english


# def main():
#     english = English('data/shakespeare.txt')
#     print(english.contents)
#     # for input_batch, output_batch in english.batch():
#     #     print(input_batch)
#     #     print(output_batch)
#
#
# if __name__ == '__main__':
#     main()
