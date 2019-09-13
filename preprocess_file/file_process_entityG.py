# -*- coding: utf-8 -*-
import os
import argparse
import random

# index 乱序， 有triple658505个，生成0-658505的乱序list
# list_triple_index = range(658505)
list_triple_index = range(390738)
list_triple_index = list(list_triple_index)
random.shuffle(list_triple_index)


# fentity = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_entity.txt', 'w')  # 保存所有的entity
# fentity_re = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_entity_re.txt', 'w')  # 保存所有entity之间的关系
# ftrain_triple = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_entity_Graph.txt', 'w')  # 训练triple的集合
# fval_triple = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/val_entity_Graph.txt', 'w')  # 保存所有的entity
# ftest_triple = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/test_entity_Graph.txt', 'w')


fentity = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_entity.txt', 'w')  # 保存所有的entity
fentity_re = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_entity_re.txt', 'w')  # 保存所有entity之间的关系
ftrain_triple = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/train_entity_Graph.txt', 'w')  # 训练triple的集合
fval_triple = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/val_entity_Graph.txt', 'w')  # 保存所有的entity
ftest_triple = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/test_entity_Graph.txt', 'w')


# 获取所有entity以及entity之间的关系
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_insnet_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_insnet_mini_new.txt') as fin:
    all_triple_copy = dict() # 从这里边 pop 已经取的triple,把已经取的triple pop掉，防止之后增加 重复的
    entity2id = dict()
    entity_id = 0
    entity_relation2id = dict()
    ty_re_id = 0
    line_id = 0
    for line in fin:
        all_triple_copy[list_triple_index.pop(0)] = line
        # all_triple_copy[line_id] = line
        # print(line_id)
        entity1, entity_re, entity2 = line.strip().split('\t')
        if entity1 not in entity2id.keys():
            entity2id[entity1] = int(entity_id)
            fentity.write(entity1+ '\n')
            entity_id = entity_id+1
        if entity2 not in entity2id.keys():
            entity2id[entity2] = int(entity_id)
            fentity.write(entity2 + '\n')
            entity_id = entity_id+1
        if entity_re not in entity_relation2id.keys():
            entity_relation2id[entity_re] = int(ty_re_id)
            fentity_re.write(entity_re + '\n')
            ty_re_id = ty_re_id+1
        line_id = line_id+1
fentity.close()
fentity_re.close()

# 打印entity 的 triple总共有多少行
print("entity re entity ,triple numbers:",line_id)
aim_triple_num = int(line_id*0.85)
print("85% of triple numbers:",aim_triple_num)

aim_triple_num_val = int(line_id*0.1)
print("15% of triple number:", aim_triple_num_val)

aim_triple_num_test = line_id - aim_triple_num - aim_triple_num_val
print("test of triple number:", aim_triple_num_test)


print("all of entity numbers:",len(entity2id))
print("all of relation numbers:",len(entity_relation2id))




# 保证从entity的图里边选出来 一个能涵盖所有entity 以及 relation的最小子集
# 只要entity2id_copy 和 entity_relation2id_copy 还没有pop干净，就从entity re entity 列表里按行读取triple，
# 读取的这一行里的 entity1 entity2 entity_re 里只要有一个元素 属于entity2id_copy ，或者entity_relation2id_copy
# 就保存这一行 triple, 并从entity2id_copy 和 entity_relation2id_copy里pop元素
#
# 直到entity2id_copy 和 entity_relation2id_copy 都为空，看一下取了多少行triple,
# 正常应该是没取够85% 然后看还差多少行才够85%，差的那些行，随机取
# 要保证，训练集里，entity和relation都出现了
entity2id_copy = dict()
entity2id_copy = entity2id # 从entity2id_copy里pop 每一个entity
entity_relation2id_copy = dict()
entity_relation2id_copy = entity_relation2id # 从entity2id_copy里pop 每一个entity_relation

triple_save_num = 0
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_insnet_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_insnet_mini_new.txt') as fin:
        for line in fin:
            if len(entity2id_copy.keys())>0 or len(entity_relation2id_copy.keys())>0:
                entity1, entity_re, entity2 = line.strip().split('\t')
                if entity1 in entity2id_copy.keys() or entity2 in entity2id_copy.keys() or entity_re in entity_relation2id_copy.keys():
                    ftrain_triple.write(line)  # entity1 entity2 entity_re 只要又一个还没有被取到，就保存这条triple
                    triple_save_num = triple_save_num + 1

                    id_temp = list(all_triple_copy.keys())[list(all_triple_copy.values()).index(line)]
                    # print(id_temp)
                    all_triple_copy.pop(id_temp)
                    # 保存triple之后，从entity2id_copy和entity_relation2id_copy里 pop
                    if entity1 in entity2id_copy.keys():
                        entity2id_copy.pop(entity1)
                    if entity2 in entity2id_copy.keys():
                        entity2id_copy.pop(entity2)
                    if entity_re in entity_relation2id_copy.keys():
                        entity_relation2id_copy.pop(entity_re)
            else:
                break
        # 所有的entity和relation都已经在train训练集里面出现了一次了
        print("sub_set of train set, triple numbers:", triple_save_num)

print(all_triple_copy)

if triple_save_num < aim_triple_num:
    a = aim_triple_num - triple_save_num # 差了多少triple
    # 从 all_triple_copy 里剩的 triple 里随机选取缺少的triple
    while a>0:
        key = random.choice(list(all_triple_copy)) # 随机选取一个key
        print(key)
        ftrain_triple.write(all_triple_copy[key])
        triple_save_num = triple_save_num +1
        all_triple_copy.pop(key)
        a = a-1
# train取完了
print("85% of (entity re entity) triples numbers are finished:",triple_save_num)
ftrain_triple.close()


# 下面提取val 部分
triple_save_num_val = 0
while triple_save_num_val < aim_triple_num_val:
    key = random.choice(list(all_triple_copy)) # 随机选取一个key
    print(key)
    fval_triple.write(all_triple_copy[key])
    triple_save_num_val = triple_save_num_val + 1
    all_triple_copy.pop(key)
print("15% of triple numbers used for val:", triple_save_num_val)
fval_triple.close()



# 下面提取test 部分
triple_save_num_test = 0
for value in all_triple_copy.values():
    ftest_triple.write(value)
    triple_save_num_test = triple_save_num_test + 1
print("5% of triple numbers used for test:", triple_save_num_test)
ftest_triple.close()











