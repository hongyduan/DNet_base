# -*- coding: utf-8 -*-

import json
import random
# fen = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en.txt', 'w')  # 保存所有的entity
# fty = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/ty.txt', 'w')  # 保存所有的entity

fen = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en.txt', 'w')  # 保存所有的entity
fty = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/ty.txt', 'w')  # 保存所有的entity

# # json.load()函数的使用，将读取json信息 entity
# file = open('/Users/bubuying/PycharmProjects/study2/data/all_entity.json','r',encoding='utf-8')
# all_entity = json.load(file)
# print(all_entity)
# print("all_entity:",len(all_entity.keys()))


# # json.load()函数的使用，将读取json信息 type
# file = open('/Users/bubuying/PycharmProjects/study2/data/all_type.json','r',encoding='utf-8')
# all_type = json.load(file)
# # print(all_type)
# print("all_type:",len(all_type.keys()))


# # json.load()函数的使用，将读取json信息 type
# file = open('/Users/bubuying/PycharmProjects/study2/data/all_triples.json','r',encoding='utf-8')
# all_triple = json.load(file)
# # print(all_triple)
# print("all_triples:",len(all_triple.keys()))


# 获取所有的entity type,写入文件en.txt和ty.txt
en = []
en_id = 0
re = []
re_id = 0
ty = []
ty_id = 0
line_id = 0
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_InsType_mini_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_InsType_mini_new.txt') as fin:
    for line in fin:
        # print(line_id)
        entity, relation, type = line.strip().split('\t')

        if entity not in en:
            en.append(entity)
            fen.write(entity+'\n')
            en_id = en_id + 1
        else:
            print(entity)

        if type not in ty:
            ty.append(type)
            fty.write(type+'\n')
            ty_id = ty_id + 1
        line_id = line_id + 1
print("line_id:",line_id)
print("en:",len(en))
print("ty:",len(ty))
fen.close()
fty.close()


# 得到entity_list
entity_list = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_entity.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_entity.txt') as fin:
    for line in fin:
        line = line.strip()
        entity_list.append(line)

# 得到type_list
type_list = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_type.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_type.txt') as fin:
    for line in fin:
        line = line.strip()
        type_list.append(line)

# 得到entity_big_list
entity_big_list = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en.txt') as fin:
    for line in fin:
        line = line.strip()
        entity_big_list.append(line)

# 得到type_big_list
type_big_list = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/ty.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/ty.txt') as fin:
    for line in fin:
        line = line.strip()
        type_big_list.append(line)

print("entity_list",len(entity_list))
# print(entity_list)
print("type_list",len(type_list))
# print(type_list)
print("entity_big_list",len(entity_big_list))
# print(entity_big_list)
print("type_big_list",len(type_big_list))
# print(type_big_list)



# 提取sub_graph;
# 已经提取的entity 的list   entity_list: 98336
# 已经提取的type 的list   type_list: 174
# 从db_InsType_mini文件中提取的 entity的list     entity_big_list
# 从db_InsType_mini文件中提取的 type的list      type_big_list

# 先把那些行（要么是entity在entity_list没出现过，要么是type在type_list没出现过）提取
# 读每一行，如果entity 不存在entity_list,存在entity_big_list
#          或者type  不存在type_list,存在type_big_list
# 这一行保存到sub_graph 同时从entity_big_list或者type_big_list pop出entity或type
# entity_list,type_list不变
# entity_big_list，type_big_list变

# 不够85% 从剩下的行里来补全 随机选取

all_list = []   # 保存那些没有被提取的那些行，用于后来为了补全到85%
# ftrain_en_ty = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_entity_typeG.txt', 'w')  # 保存所有的entity
# fval_en_ty = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/val_entity_typeG.txt', 'w')
# ftest_en_ty = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/test_entity_typeG.txt', 'w')
ftrain_en_ty = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/train_entity_typeG.txt', 'w')  # 保存所有的entity
fval_en_ty = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/val_entity_typeG.txt', 'w')
ftest_en_ty = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/test_entity_typeG.txt', 'w')
count = 0
entity_count =0
type_count = 0
line_count = 0
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_InsType_mini_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_InsType_mini_new.txt') as fin:
    for line in fin:
        entity, relation, type = line.strip().split('\t')
        if (entity not in entity_list and entity in entity_big_list) or (type not in type_list and type in type_big_list):
            ftrain_en_ty.write(line)
            count = count + 1
            if entity not in entity_list and entity in entity_big_list:
                entity_big_list.remove(entity)
                entity_count = entity_count + 1
            if type not in type_list and type in type_big_list:
                type_big_list.remove(type)
                type_count = type_count + 1
        else:
            all_list.append(line.strip())
        line_count = line_count + 1
line_aim_train = int(line_count * 0.85)
line_aim_val = int(line_count * 0.1)
line_aim_test = line_count - line_aim_train - line_aim_val
print("aim_train:",line_aim_train)
print("aim_val:",line_aim_val)
print("aim_test:",line_aim_test)


print("only_count:",count)
print("only_entity_count:",entity_count)
print("only_type_count:",type_count)


print("all_line:",line_count)
print("remain_line:",len(all_list))

# 补全train
if count < line_aim_train:
    temp = line_aim_train - count
    while temp > 0:
        add_item = all_list[random.randint(0,len(all_list)-1)]
        ftrain_en_ty.write(add_item+'\n')
        count = count + 1
        all_list.remove(add_item)
        temp = temp - 1
print("_________train finished_________")
print("train_count:", count)
ftrain_en_ty.close()

# 提取val
val_count = 0
while line_aim_val > 0:
    add_item = all_list[random.randint(0,len(all_list)-1)]
    fval_en_ty.write(add_item+'\n')
    val_count = val_count + 1
    all_list.remove(add_item)
    line_aim_val = line_aim_val - 1
print("_________val finished_________")
print("val_count:",val_count)
fval_en_ty.close()


# 得到test
test_count = 0
while len(all_list)>0:
    ftest_en_ty.write(all_list[0]+'\n')
    test_count = test_count + 1
    all_list.pop(0)
print("_________test finished_________")
print("test_count:",test_count)
ftest_en_ty.close()



