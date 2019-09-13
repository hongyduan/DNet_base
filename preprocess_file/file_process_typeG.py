# -*- coding: utf-8 -*-
import os
import argparse
import random

# index 乱序， 有triple763个，生成0-762的乱序list
# list_triple_index = range(763)
list_triple_index = range(8962)
list_triple_index = list(list_triple_index)
random.shuffle(list_triple_index)

# ftype = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_type.txt', 'w')  # 保存所有的entity
# ftype_re = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_type_re.txt', 'w')  # 保存所有entity之间的关系
# ftrain_triple = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_type_Graph.txt', 'w')  # 训练triple的集合
# fval_triple = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/val_type_Graph.txt', 'w')  # 保存所有的entity
# ftest_triple = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/test_type_Graph.txt', 'w')


ftype = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_type.txt', 'w')  # 保存所有的entity
ftype_re = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_type_re.txt', 'w')  # 保存所有entity之间的关系
ftrain_triple = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/train_type_Graph.txt', 'w')  # 训练triple的集合
fval_triple = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/val_type_Graph.txt', 'w')  # 保存所有的entity
ftest_triple = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/test_type_Graph.txt', 'w')


# 获取所有type以及type之间的关系
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_onto_small_mini_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_ontonet_new.txt') as fin:
    all_triple_copy = dict() # 从这里边 pop 已经取的triple,把已经取的triple pop掉，防止之后增加 重复的
    type2id = dict()
    ty_id = 0
    type_relation2id = dict()
    ty_re_id = 0
    line_id = 0
    for line in fin:
        all_triple_copy[list_triple_index.pop(0)] = line
        # all_triple_copy[line_id] = line
        # print(line_id)
        type1, type_re, type2 = line.strip().split('\t')
        if line_id == 0:
            type2id[type1] = int(ty_id)
            ftype.write(type1+'\n')
            ty_id = ty_id+1
            if type2 not in type2id.keys():
                type2id[type2] = int(ty_id)
                ftype.write(type2 + '\n')
                ty_id = ty_id+1
            type_relation2id[type_re] = int(ty_re_id)
            ftype_re.write(type_re+'\n')
            ty_re_id = ty_re_id+1
        else:
            if type1 not in type2id.keys():
                type2id[type1] = int(ty_id)
                ftype.write(type1 + '\n')
                ty_id = ty_id+1
            if type2 not in type2id.keys():
                type2id[type2] = int(ty_id)
                ftype.write(type2 + '\n')
                ty_id = ty_id+1
            if type_re not in type_relation2id.keys():
                type_relation2id[type_re] = int(ty_re_id)
                ftype_re.write(type_re + '\n')
                ty_re_id = ty_re_id+1
        line_id = line_id+1
# 打印type 的 triple总共有多少行
print("type re type ,triple numbers:",line_id)
aim_triple_num = int(line_id*0.85)
print("85% of triple numbers:",aim_triple_num)
ftype.close()
ftype_re.close()

aim_triple_num_val = int(line_id*0.1)
print("15% of triple number:", aim_triple_num_val)

aim_triple_num_test = line_id - aim_triple_num - aim_triple_num_val
print("test of triple number:", aim_triple_num_test)


print("all of type numbers:",len(type2id))
print("all of relation numbers:",len(type_relation2id))




# 保证从type的图里边选出来 一个能涵盖所有type 以及 relation的最小子集
# 只要type2id_copy 和 type_relation2id_copy 还没有pop干净，就从type re type 列表里按行读取triple，
# 读取的这一行里的 type1 type2 type_re 里只要有一个元素 属于type2id_copy ，或者type_relation2id_copy
# 就保存这一行 triple, 并从type2id_copy 和 type_relation2id_copy里pop元素
#
# 直到type2id_copy 和 type_relation2id_copy 都为空，看一下取了多少行triple,
# 正常应该是没取够85% 然后看还差多少行才够85%，差的那些行，随机取
# 要保证，训练集里，type和relation都出现了
type2id_copy = dict()
type2id_copy = type2id # 从type2id_copy里pop 每一个type
type_relation2id_copy = dict()
type_relation2id_copy = type_relation2id # 从type2id_copy里pop 每一个type_relation

triple_save_num = 0
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_onto_small_mini_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_ontonet_new.txt') as fin:
        for line in fin:
            if len(type2id_copy.keys())>0 or len(type_relation2id_copy.keys())>0:
                type1, type_re, type2 = line.strip().split('\t')
                if type1 in type2id_copy.keys() or type2 in type2id_copy.keys() or type_re in type_relation2id_copy.keys():
                    ftrain_triple.write(line)  # type1 type2 type_re 只要又一个还没有被取到，就保存这条triple
                    triple_save_num = triple_save_num + 1

                    id_temp = list(all_triple_copy.keys())[list(all_triple_copy.values()).index(line)]
                    # print(id_temp)
                    all_triple_copy.pop(id_temp)
                    # 保存triple之后，从type2id_copy和type_relation2id_copy里 pop
                    if type1 in type2id_copy.keys():
                        type2id_copy.pop(type1)
                    if type2 in type2id_copy.keys():
                        type2id_copy.pop(type2)
                    if type_re in type_relation2id_copy.keys():
                        type_relation2id_copy.pop(type_re)
            else:
                break
        # 所有的type和relation都已经在train训练集里面出现了一次了
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












