#
# -*- coding: utf-8 -*-

import random

#  打乱文件
#  db_onto_small_mini
#  db_insnet
#  db_InsType_mini



# 把文件里的顺序打乱,保存一个新的 type Graph
# ftypetemp = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_onto_small_mini_new.txt', 'w')
ftypetemp = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_ontonet_new.txt', 'w')
list_triple = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia/db_onto_small_mini.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago/yago_ontonet.txt') as fin:
    for line in fin:
        list_triple.append(line)
random.shuffle(list_triple)
i = 0
for triple in list_triple:
    ftypetemp.write(triple)
    i = i + 1
print("~~", i)
ftypetemp.close()


# 把文件里的顺序打乱,保存一个新的  entity Graph
# fentitytemp = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_insnet_new.txt', 'w')
fentitytemp = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_insnet_mini_new.txt', 'w')
list_triple = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia/db_insnet.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago/yago_insnet_mini.txt') as fin:
    for line in fin:
        list_triple.append(line)
random.shuffle(list_triple)
i = 0
for triple in list_triple:
    fentitytemp.write(triple)
    i = i + 1
print("~~", i)
fentitytemp.close()



# 把文件里的顺序打乱,保存一个新的  entity_type Graph
# fen_tytemp = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_InsType_mini_new.txt', 'w')
fen_tytemp = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_InsType_mini_new.txt', 'w')
list_triple = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia/db_InsType_mini.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago/yago_InsType_mini.txt') as fin:
    for line in fin:
        list_triple.append(line)
random.shuffle(list_triple)
i = 0
for triple in list_triple:
    fen_tytemp.write(triple)
    i = i + 1
print("~~", i)
fen_tytemp.close()


