#
# # entity
# # 写
# ffinal_en_order = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/final_entity_order.txt', 'w')  #
#
#
# start = 0
# # 读
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/final_entity.txt') as fin:
#     for line in fin:
#         ffinal_en_order.write(str(start)+'\t'+line)
#         start = start + 1
# ffinal_en_order.close()

#
# # type
# # 写
# ffinal_ty_order = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/final_type_order.txt', 'w')  #
#
#
# start = 0
# # 读
# with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/final_type.txt') as fin:
#     for line in fin:
#         ffinal_ty_order.write(str(start)+'\t'+line)
#         start = start + 1
# ffinal_ty_order.close()


# entity_relation
# 写
ffinal_en_relation_order = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/ffinal_en_relation_order.txt', 'w')  #


start = 0
# 读
with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_entity_re.txt') as fin:
    for line in fin:
        ffinal_en_relation_order.write(str(start)+'\t'+line)
        start = start + 1
ffinal_en_relation_order.close()


# type_relation
# 写
ffinal_ty_relation_order = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/ffinal_ty_relation_order.txt', 'w')  #


start = 0
# 读
with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/all_type_re.txt') as fin:
    for line in fin:
        ffinal_ty_relation_order.write(str(start)+'\t'+line)
        start = start + 1
ffinal_ty_relation_order.close()
