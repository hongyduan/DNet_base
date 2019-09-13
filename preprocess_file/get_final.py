import os

# ffinal_en = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/final_entity.txt', 'w')  #
# ffinal_ty = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/final_type.txt', 'w')
ffinal_en = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/final_entity.txt', 'w')  #
ffinal_ty = open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/final_type.txt', 'w')

# 获取所有实体 从all_entity, 和 en, 保存进final_entity
final_entity = []
fi_enid = 0

# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_entity.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_entity.txt') as fin:
    for line in fin:
        if line.strip() not in final_entity:
            final_entity.append(line.strip())
            ffinal_en.write(line)
            fi_enid = fi_enid + 1
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/en.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/en.txt') as fin:
    for line in fin:
        if line.strip() not in final_entity:
            final_entity.append(line.strip())
            ffinal_en.write(line)
            fi_enid = fi_enid + 1
ffinal_en.close()
print("number of final entities:", fi_enid)


# 获取所有type  从all_type, 和 ty, 保存进final_type
final_type = []
fi_tyid = 0

# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_type.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/all_type.txt') as fin:
    for line in fin:
        if line.strip() not in final_type:
            final_type.append(line.strip())
            ffinal_ty.write(line)
            fi_tyid = fi_tyid + 1
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/ty.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/ty.txt') as fin:
    for line in fin:
        if line.strip() not in final_type:
            final_type.append(line.strip())
            ffinal_ty.write(line)
            fi_tyid = fi_tyid + 1
ffinal_ty.close()
print("number of final typies:", fi_tyid)




# 获取所有的train  ，包括entityG, entity_typeG
# 检查train 里的entity 是否涵盖了所有的 final_entity
# entity
final_entity_com = []
final_type_com = []
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_entity_Graph.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/train_entity_Graph.txt') as fin:
    for line in fin:
        entity1 = line.strip().split('\t')[0]
        entity2 = line.strip().split('\t')[2]
        if entity1 not in final_entity_com:
            final_entity_com.append(entity1)
            if entity1 in final_entity:
                final_entity.remove(entity1)
        if entity2 not in final_entity_com:
            final_entity_com.append(entity2)
            if entity2 in final_entity:
                final_entity.remove(entity2)
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_type_Graph.txt') as fin:
with open('/Users/bubuying/PycharmProjects/DNet/data/yago_result/train_type_Graph.txt') as fin:
    for line in fin:
        type1 = line.strip().split('\t')[0]
        type2 = line.strip().split('\t')[2]
        if type1 not in final_type_com:
            final_type_com.append(type1)
            if type1 in final_type:
                final_type.remove(type1)
        if type2 not in final_type_com:
            final_type_com.append(type2)
            if type2 in final_type:
                final_type.remove(type2)
# with open ('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_entity_typeG.txt') as fin:
with open ('/Users/bubuying/PycharmProjects/DNet/data/yago_result/train_entity_typeG.txt') as fin:
    for line in fin:
        entity = line.strip().split('\t')[0]
        type = line.strip().split('\t')[2]
        if entity not in final_entity_com:
            final_entity_com.append(entity)
            if entity in final_entity:
                final_entity.remove(entity)
        if type not in final_type_com:
            final_type_com.append(type)
            if type in final_type:
                final_type.remove(type)
print("final_entity:",len(final_entity))  # 应该是0
print("final_entity_com:",len(final_entity_com))  # 应该和fi_enid一样

print("fianl_typies:",len(final_type)) # 应该是0
print("final_type_com:",len(final_type_com))  # 应该和fi_tyid一样
