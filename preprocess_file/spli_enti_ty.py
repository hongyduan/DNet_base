#  把entity和type的关系 切割成子块
#  使得属于同一个 type的entity 在一个dic里面，就是{type:[entity1, entity2, entity3]}


#  读文件db_InsType_mini_new.txt  每读一行，就看这个type有没有存在dic的key里面
#  如果key在，就把这一行的entity写在key的内容里
#  如过key不在，就把key写进dic的key里，这一行的entity写在key的内容里


#  直到都读完，得到了一个dic里面, 分了许多块，每一块代表每一个type对应的entity

#  最后把dic里面的内容写进文件里：
#  #type1
#  entity1
#  entity2
#  entity3
#  ...
#  #type2
#  entity9
#  entity10
#  entity11
#  ...
#  #typen
#  entity90
#  entity91
#  entity92
# ______________________________________________________________
#  要写的文件流, en_ty_blocks是###type 以及所有属于type的entity
#  处理的文件是db_InsType_mini_new
#  对应的dict是：dic_temp
#  ### type_1
#  entity_1(1)
#  entity_1(2)
#  entity_1(3)
#  ...
#  entity_1(n)
#  ...
#  ### type_m
#  entity_m(1)
#  entity_m(2)
#  entity_m(3)
#  ...
#  entity_m(n)

# f_en_ty_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en_ty_blocks.txt', 'w')  #
f_en_ty_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en_ty_blocks.txt', 'w')  #
#  type1 number_of_entities_belong_type1
#  type2 number_of_entities_belong_type2
#  ...
#  typen number_of_entities_belong_typen

# f_en_ty_blocks_statistic = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en_ty_blocks_statistic.txt', 'w')  #
f_en_ty_blocks_statistic = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en_ty_blocks_statistic.txt', 'w')  #
#  dict文件

dic_temp = {}
dic_temp_num = {}


#  读文件
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/yago_InsType_mini_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_InsType_mini_new.txt') as fin:
    for line in fin:
        entity, re, type = line.strip().split('\t')
        if type not in dic_temp:
            dic_temp[type] = set()
            dic_temp_num[type] = 0
        dic_temp[type].add(entity)
        dic_temp_num[type] = dic_temp_num[type] + 1

list1_a = []
list2_a = []




#  f_en_ty_blocks_statistic排序
dic_temp_num_new = {}
dic_temp_num = sorted(dic_temp_num.items(), key=lambda item: item[1], reverse=True)
for item in dic_temp_num:
    dic_temp_num_new[item[0]] = item[1]
for key, value in dic_temp_num_new.items():
    f_en_ty_blocks_statistic.write("###"+key+'\t'+str(value)+'\n')


#  根据排序后的dic_temp_num_new，写f_en_ty_blocks
for key in dic_temp_num_new.keys():
    f_en_ty_blocks.write("###"+key+'\n')
    for values in dic_temp[key]:
        f_en_ty_blocks.write(values+'\n')
f_en_ty_blocks.close()













#  要写的文件流
#  en_en_re_blocks是entity和entity之间的关系里 统计的左边entity 对应的所有和其他entity之间的关系
#  处理的文件是db_insnet_new
#  对应的dict: en_en_d
#  ###entity_1
#  entity_1 r entity_1_right(1)
#  entity_1 r entity_1_right(2)
#  ...
#  entity_1 r entity_1_right(n)
#  ...
#  ###entity_m
#  entity_m r entity_m_right(1)
#  entity_m r entity_m_right(2)
#  ...
#  entity_m r entity_m_right(n)
#
#  en_en_re_blocks是entity和entity之间的关系里 统计的右边的entity 对应的所有和其他entity之间的关系
#  处理的文件是db_insnet_new
#  对应的dict: en_en_d_re
#  ###entity_1
#  entity_1_left(1) r entity_1
#  entity_1_left(2) r entity_1
#  ...
#  entity_1_left(n) r entity_1
#  ...
#  ###entity_m
#  entity_m_left(1) r entity_m
#  entity_m_left(2) r entity_m
#  ...
#  entity_m_left(n) r entity_m

# en_en_re_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en_en_re_blocks.txt', 'w')  #  #  读取entity之间的关系，写进dict
# en_en_re_blocks_re = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en_en_re_blocks_re.txt', 'w')

en_en_re_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en_en_re_blocks.txt', 'w')  #  #  读取entity之间的关系，写进dict
en_en_re_blocks_re = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en_en_re_blocks_re.txt', 'w')
#  读文件
en_en_d = {}
en_en_d_re = {}

# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/db_insnet_new.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/yago_insnet_mini_new.txt') as fin:
    for line in fin:
        entity1, re, entity2 = line.strip().split('\t')
        if entity1 not in en_en_d:
            en_en_d[entity1] = list()
        en_en_d[entity1].append(str(entity1+'\t'+re+'\t'+entity2)) # entity  在左边的所有关系
        if entity2 not in en_en_d_re:
            en_en_d_re[entity2] = list()
        en_en_d_re[entity2].append(str(entity1+'\t'+re+'\t'+entity2))  # entity 在右边的所有关系

#  把读取的entity之间的关系写进文件
for key,value in en_en_d.items():
    en_en_re_blocks.write("#"+key+'\n')
    for values in value:
        en_en_re_blocks.write(values+'\n')
en_en_re_blocks.close()

#  把读取的entity之间的关系写进文件,反向的
for key,value in en_en_d_re.items():
    en_en_re_blocks_re.write("#"+key+'\n')
    for values in value:
        en_en_re_blocks_re.write(values+'\n')
en_en_re_blocks_re.close()



blocks_final_dic = {}
#  用blocks_final_dic 来保存 {type1:{en r1 en, en r2 en, ..., en r3 en}, type2{}     ... }
#  遍历en_ty_blocks，dict 看属于 同一个type的所有的 entity
#  提取这些entity之间的关系
#  提取这些entity之间关系的时候
#  维护了两个表，分别是en_en_re_blocks和en_en_re_blocks_re
#  type_entity之间的关系 维护一个表，是dic_temp
#  对属于一个type的里面的entity
#  遍历每一个entity, 对这个entity查找   en_en_re_blocks和en_en_re_blocks_re  表里key是entity的那些关系，
#  如果这个关系不在blocks_final_dic 就保存； 　关系已经存在就不保存了
#  看数量
blocks_final_dic_count = dict()
blocks_final_dic_count_end = dict()
for key,value in dic_temp.items():    # key是type, value是属于这个type的所有entity
    blocks_final_dic[key] = set()
    count_start = 0
    blocks_final_dic_count[key] = count_start
    for values in value:  # 遍历属于这个type的所有entity
        # en_en_re_blocks_re存储的是entity2: entity1+'\t'+re+'\t'+entity2
        # en_en_re_blocks存储的是entity1: entity1+'\t'+re+'\t'+entity2
        if values in en_en_d.keys():
            for all_re in en_en_d[values]:
                if all_re.strip().split('\t')[2] in dic_temp[key]:  # 如果entity2和entity1属于一个type
                    if all_re not in blocks_final_dic[key]:
                        blocks_final_dic[key].add(all_re)
                        blocks_final_dic_count[key] = blocks_final_dic_count[key] + 1
        if values in en_en_d_re.keys():
            for all_re_r in en_en_d_re[values]:
                if all_re_r.strip().split('\t')[0] in dic_temp[key]:
                    if all_re_r not in blocks_final_dic[key]:
                        blocks_final_dic[key].add(all_re_r)
                        blocks_final_dic_count[key] = blocks_final_dic_count[key] + 1
blocks_final_dic_count = sorted(blocks_final_dic_count.items(),key=lambda  item: item[1], reverse=True)
blocks_final_dic_count_statisctic = dict()
start = 1
all_start = 0
for ite in blocks_final_dic_count:
    blocks_final_dic_count_end[ite[0]] = ite[1]
    list1_a.append(ite[0])
    list2_a.append(ite[1])



#   看每个type下所包含的entity关系数目 的 百分比，分布情况
    if str(ite[1]) not in blocks_final_dic_count_statisctic.keys():
        blocks_final_dic_count_statisctic[str(ite[1])] = 1
        all_start = all_start + 1      #
    else:
        blocks_final_dic_count_statisctic[str(ite[1])] = blocks_final_dic_count_statisctic[str(ite[1])] + 1
        all_start = all_start + 1

for key, item in blocks_final_dic_count_statisctic.items():   # key 代表多少条关系，item代表包含这么多条关系的 type有多少个， all_start带表总共多少type
    item_t = float(item)/float(all_start)
    blocks_final_dic_count_statisctic[key] = float(item_t)


list1 = []
list2 = []
for key, item in blocks_final_dic_count_statisctic.items():
    list1.append(key)
    list2.append(item)

#  blocks_final是属于同一个type内所有entity，这些entity之间的关系
#  处理的文件是db_insnet_new，db_InsType_mini_new
#  对应的dict: blocks_final_dic
#  ###type1
#  entity_belong_type1 r entity_belong_type1
#  entity_belong_type1 r entity_belong_type1
#  ...
#  entity_belong_type1 r entity_belong_type1
#  ...
#  ###typem
#  entity_belong_typem r entity_belong_typem
#  entity_belong_typem r entity_belong_typem
#  ...
#  entity_belong_typem r entity_belong_typem

# blocks_final = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/blocks_final.txt', 'w')
blocks_final = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/blocks_final.txt', 'w')
#  把blocks_final_dic写入文件
for key, value in blocks_final_dic.items():
    blocks_final.write("###"+key+'\n')
    for values in value:
        blocks_final.write(values+'\n')
blocks_final.close()
print("final")

#  blocks_final是属于同一个type内所有entity，这些entity之间的关系数目
#  处理的文件是db_insnet_new，db_InsType_mini_new
#  对应的dict: blocks_final_dic_count_end
#  ###type1  number_of_relations_for_entities_which_belong_to_type1
#  ###type2  number_of_relations_for_entities_which_belong_to_type2
#  ###type3  number_of_relations_for_entities_which_belong_to_type3
#  ...
#  ###typem  number_of_relations_for_entities_which_belong_to_typem

# blocks_final_count = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/blocks_final_count.txt', 'w')
blocks_final_count = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/blocks_final_count.txt', 'w')
#  把blocks_final_dic  按照count排序后的写入文件
for key, num in blocks_final_dic_count_end.items():
    blocks_final_count.write("###"+key+"\t"+str(num)+'\n')
    # for values in blocks_final_dic[key]:
    #     blocks_final_count.write(values+'\n')
blocks_final_count.close()

print("end")



import matplotlib.pyplot as plt


# 保证圆形
plt.axes(aspect=1)
plt.pie(x=list2, labels=list1)
# plt.savefig("/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/temp1.png")
plt.savefig("/Users/bubuying/PycharmProjects/DNet/data/yago_result/temp1.png")
plt.show()


# 柱状图 dic_temp，每个type下的所有entity之间的关系的数目
plt.bar(range(len(list1_a)), list2_a)
# plt.savefig("/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/temp2.png")
plt.savefig("/Users/bubuying/PycharmProjects/DNet/data/yago_result/temp2.png")
plt.show()


#  绘制一个柱状图每一个type下面有多少entity
#  dic_temp
#  根据排序后的dic_temp_num_new 从dic_temp里抽取内容
dic_temp_list1 = []  # 所有的type
dic_temp_list2 = []  # 和type对应的entity的数量
for key,value in dic_temp_num_new.items():
    dic_temp_list1.append(key)
    dic_temp_list2.append(value)
plt.bar(range(len(dic_temp_list1)), dic_temp_list2)
# plt.savefig("/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/temp3.png")
plt.savefig("/Users/bubuying/PycharmProjects/DNet/data/yago_result/temp3.png")
plt.show()









