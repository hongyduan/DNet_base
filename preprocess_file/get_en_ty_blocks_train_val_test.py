
# train
f_train_en_ty_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_en_ty_blocks.txt', 'w')  #

d = dict()
# c = 0
with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/train_entity_typeG.txt') as fin:
    for line in fin:
        entity, r, type = line.strip().split('\t')
        if type not in d:
            d[type] = set()
            # c = c+1
        d[type].add(entity)
# print(c)

for key, value in d.items():
    f_train_en_ty_blocks.write("###"+'\t'+key+'\n')
    for values in value:
        f_train_en_ty_blocks.write(values+'\n')
f_train_en_ty_blocks.close()



# val
f_val_en_ty_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/val_en_ty_blocks.txt', 'w')  #

# c = 0
d = dict()
with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/val_entity_typeG.txt') as fin:
    for line in fin:
        entity, r, type = line.strip().split('\t')
        if type not in d:
            d[type] = set()
            # c = c +1
        d[type].add(entity)
# print(c)

for key, value in d.items():
    f_val_en_ty_blocks.write("###"+'\t'+key+'\n')
    for values in value:
        f_val_en_ty_blocks.write(values+'\n')
f_val_en_ty_blocks.close()



# test
f_test_en_ty_blocks = open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/test_en_ty_blocks.txt', 'w')  #
d = dict()
with open('/Users/bubuying/PycharmProjects/DNet/data/dbpedia_result/test_entity_typeG.txt') as fin:
    for line in fin:
        entity, r, type = line.strip().split('\t')
        if type not in d:
            d[type] = set()
        d[type].add(entity)


for key, value in d.items():
    f_test_en_ty_blocks.write("###"+'\t'+key+'\n')
    for values in value:
        f_test_en_ty_blocks.write(values+'\n')
f_test_en_ty_blocks.close()






