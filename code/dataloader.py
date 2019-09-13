from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


class TrainDataset_en(Dataset):
    def __init__(self, train_entity_triples, nentity, nentity_relation, negative_sample_size_en, mode):
        self.len = len(train_entity_triples)
        self.train_entity_triples = train_entity_triples
        self.nentity = nentity
        self.nentity_relation = nentity_relation
        self.negative_sample_size_en = negative_sample_size_en
        self.mode =mode
        self.en_en_count = self.count_frequency(train_entity_triples)
        self.en_en_true_head, self.en_en_true_tail = self.get_true_head_and_tail(train_entity_triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample_en_en = self.train_entity_triples[idx]

        head_en_en, relation_en_en, tail_en_en = positive_sample_en_en

        subsampling_weight_en_en = self.en_en_count[(head_en_en,relation_en_en)]+self.en_en_count[(tail_en_en,-relation_en_en-1)]
        subsampling_weight_en_en = torch.sqrt(1/torch.Tensor([subsampling_weight_en_en]))

        negative_sample_list_en_en = []
        negative_sample_size_en_en = 0

        while negative_sample_size_en_en < self.negative_sample_size_en:
            negative_sample_en_en = np.random.randint(self.nentity, size=self.negative_sample_size_en*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample_en_en,
                    self.en_en_true_head[(relation_en_en, tail_en_en)],
                    assume_unique=True,
                    invert = True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample_en_en,
                    self.en_en_true_tail[(head_en_en, relation_en_en)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)

            negative_sample_en_en = negative_sample_en_en[mask]
            negative_sample_list_en_en.append(negative_sample_en_en)
            negative_sample_size_en_en = negative_sample_size_en_en + negative_sample_size_en_en.__sizeof__()

        negative_sample_en_en = np.concatenate(negative_sample_list_en_en)[:self.negative_sample_size_en]
        negative_sample_en_en = torch.from_numpy(negative_sample_en_en)
        positive_sample_en_en = torch.LongTensor(positive_sample_en_en)

        return positive_sample_en_en, negative_sample_en_en, subsampling_weight_en_en, self.mode

    # @staticmethod 静态方法 类或实例均可调用； 静态方法函数里不传入self 或 cls
    # 在若干张图片拼接为一个batch的时候，我们需要组织一种数据拼接方式，
    # 这个数据拼接方式就是collate_fn。这里说明了batch当中每个data由哪些元素组成
    @staticmethod
    def collate_fn(data):
        positive_sample_en_en = torch.stack([_[0] for _ in data], dim=0)
        negative_sample_en_en = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight_en_en = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample_en_en, negative_sample_en_en, subsample_weight_en_en, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count



    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

class TestDataset_en(Dataset):
    def __init__(self, test_entity_triples,all_true_triples_entity, nentity,nentity_re, mode):
        self.len = len(test_entity_triples)
        self.en_triple_set = set(all_true_triples_entity)
        self.en_triples = test_entity_triples
        self.nentity = nentity
        self.nentity_re = nentity_re
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head_en_en, relation_en_en, tail_en_en = self.en_triples[idx]

        if self.mode == 'head-batch':
            tmp_en_en = [(0, rand_head_en_en) if (rand_head_en_en, relation_en_en, tail_en_en) not in self.en_triple_set else (-1, head_en_en) for rand_head_en_en in range(self.nentity)]
            tmp_en_en[head_en_en] = (0, head_en_en)
        elif self.mode == 'tail-batch':
            tmp_en_en = [(0, rand_tail_en_en) if (head_en_en, relation_en_en, rand_tail_en_en) not in self.en_triple_set else (-1, tail_en_en) for rand_tail_en_en in range(self.nentity)]
            tmp_en_en[tail_en_en] = (0, tail_en_en)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp_en_en = torch.LongTensor(tmp_en_en)
        filter_bias_en_en = tmp_en_en[:,0].float()
        negative_sample_en_en = tmp_en_en[:,1]
        positive_sample_en_en = torch.LongTensor((head_en_en, relation_en_en, tail_en_en))

        return positive_sample_en_en, negative_sample_en_en, filter_bias_en_en, self.mode
    @staticmethod
    def collate_fn(data):
        positive_sample_en_en = torch.stack([_[0] for _ in data], dim=0)
        negative_sample_en_en = torch.stack([_[1] for _ in data], dim=0)
        filter_bias_en_en = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample_en_en, negative_sample_en_en, filter_bias_en_en, mode




class TrainDataset_ty(Dataset):
    def __init__(self, train_type_triples, ntype, ntype_relation, negative_sample_size_ty, mode):
        self.len = len(train_type_triples)
        self.train_type_triples = train_type_triples
        self.ntype = ntype
        self.ntype_relation = ntype_relation
        self.negative_sample_size_ty = negative_sample_size_ty
        self.mode =mode
        self.ty_ty_count = self.count_frequency(train_type_triples)
        self.ty_ty_true_head, self.ty_ty_true_tail = self.get_true_head_and_tail(train_type_triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample_ty_ty = self.train_type_triples[idx]

        head_ty_ty, relation_ty_ty, tail_ty_ty = positive_sample_ty_ty

        subsampling_weight_ty_ty = self.ty_ty_count[(head_ty_ty,relation_ty_ty)]+self.ty_ty_count[(tail_ty_ty,-relation_ty_ty-1)]
        subsampling_weight_ty_ty = torch.sqrt(1/torch.Tensor([subsampling_weight_ty_ty]))

        negative_sample_list_ty_ty = []
        negative_sample_size_ty_ty = 0

        while negative_sample_size_ty_ty < self.negative_sample_size_ty:
            negative_sample_ty_ty = np.random.randint(self.ntype, size=self.negative_sample_size_ty*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample_ty_ty,
                    self.ty_ty_true_head[(relation_ty_ty, tail_ty_ty)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample_ty_ty,
                    self.ty_ty_true_tail[(head_ty_ty, relation_ty_ty)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample_ty_ty = negative_sample_ty_ty[mask]
            negative_sample_list_ty_ty.append(negative_sample_ty_ty)
            negative_sample_size_ty_ty = negative_sample_size_ty_ty + negative_sample_size_ty_ty.size
        negative_sample_ty_ty = np.concatenate(negative_sample_list_ty_ty)[:self.negative_sample_size_ty]
        negative_sample_ty_ty = torch.from_numpy(negative_sample_ty_ty)
        positive_sample_ty_ty = torch.LongTensor(positive_sample_ty_ty)


        return positive_sample_ty_ty, negative_sample_ty_ty, subsampling_weight_ty_ty, self.mode

    # @staticmethod 静态方法 类或实例均可调用； 静态方法函数里不传入self 或 cls
    # 在若干张图片拼接为一个batch的时候，我们需要组织一种数据拼接方式，
    # 这个数据拼接方式就是collate_fn。这里说明了batch当中每个data由哪些元素组成
    @staticmethod
    def collate_fn(data):
        positive_sample_ty_ty = torch.stack([_[0] for _ in data], dim=0)
        negative_sample_ty_ty = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight_ty_ty = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample_ty_ty, negative_sample_ty_ty, subsample_weight_ty_ty, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count



    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset_ty(Dataset):
    def __init__(self,test_type_triples,all_true_triples_type,ntype,ntype_re,mode):
        self.len = len(test_type_triples)
        self.ty_triple_set = set(all_true_triples_type)
        self.ty_triples = test_type_triples
        self.ntype = ntype
        self.ntype_re = ntype_re
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head_ty_ty, relation_ty_ty, tail_ty_ty = self.ty_triples[idx]

        if self.mode == 'head-batch':
            tmp_ty_ty = [(0, rand_head_ty_ty) if (rand_head_ty_ty, relation_ty_ty, tail_ty_ty) not in self.ty_triple_set else (-1, head_ty_ty) for rand_head_ty_ty in range(self.ntype)]
            tmp_ty_ty[head_ty_ty] = (0, head_ty_ty)

        elif self.mode == 'tail-batch':
            tmp_ty_ty = [(0, rand_tail_ty_ty) if (head_ty_ty, relation_ty_ty, rand_tail_ty_ty) not in self.ty_triple_set else (-1, tail_ty_ty) for rand_tail_ty_ty in range(self.ntype)]
            tmp_ty_ty[tail_ty_ty] = (0, tail_ty_ty)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp_ty_ty = torch.LongTensor(tmp_ty_ty)
        filter_bias_ty_ty = tmp_ty_ty[:, 0].float()
        negative_sample_ty_ty = tmp_ty_ty[:, 1]
        positive_sample_ty_ty = torch.LongTensor((head_ty_ty, relation_ty_ty, tail_ty_ty))

        return positive_sample_ty_ty, negative_sample_ty_ty, filter_bias_ty_ty, self.mode
    @staticmethod
    def collate_fn(data):
        positive_sample_ty_ty = torch.stack([_[0] for _ in data], dim=0)
        negative_sample_ty_ty = torch.stack([_[1] for _ in data], dim=0)
        filter_bias_ty_ty = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample_ty_ty, negative_sample_ty_ty, filter_bias_ty_ty, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
