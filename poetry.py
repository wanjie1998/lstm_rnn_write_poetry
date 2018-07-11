# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wanjie time:2018/6/9 0009

from baseclass import BaseClass


class Poetry(BaseClass):
    # 继承父类的所有属性
    def __init__(self, file_path):
        BaseClass.__init__(self, file_path)

    # 覆写父类方法
    def _get_content(self):
        poetrys = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    title, content = line.split(':')
                    content = content.replace(' ', '')
                    if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # content = '[' + content + ']'
                    poetrys.append(content)
                except ValueError:
                    pass
        poetrys = self._sort_content_by_len(poetrys)
        return poetrys


# def main():
#     poetry = Poetry('data/poetry.txt')
#     # for input_batch, output_batch in poem.batch():
#     #     print(input_batch, output_batch)
#     print(poetry.word_to_int)
#     print(poetry.int_to_word)
#     print(poetry.words_len)
#     # print(len(poetry.content_to_vector))
#     # print('0:', poetry.content_to_vector[0])
#
#
# if __name__ == '__main__':
#     main()
