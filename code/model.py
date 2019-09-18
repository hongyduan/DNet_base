
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from dataloader import TestDataset_en, TestDataset_ty

# 实体embedding 模型
class DKGE_Model(nn.Module):

    def __init__(self, model_name, nnode, nnode_re, hidden_dim, gamma, gamma_intra, double_node_embedding=False, double_node_re_embedding=False):
        super(DKGE_Model, self).__init__()
        self.model_name = model_name
        self.nnode = nnode
        self.nnode_re = nnode_re
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.gamma_intra = gamma_intra
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad = False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.node_dim = hidden_dim*2 if double_node_embedding else hidden_dim
        self.node_re_dim = hidden_dim*2 if double_node_re_embedding else hidden_dim

        self.node_embedding = nn.Parameter(torch.zeros(nnode, self.node_dim))
        nn.init.uniform_(
            tensor=self.node_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.node_re_embedding = nn.Parameter(torch.zeros(nnode_re, self.node_re_dim))
        nn.init.uniform_(
            tensor=self.node_re_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_node_embedding or double_node_re_embedding):
            raise ValueError('RotatE should use --double_node_embedding')

        if model_name == 'ComplEx' and (not double_node_embedding or not  double_node_re_embedding):
            raise ValueError('ComplEx should use --double_node_embedding and --double_node_re_embedding')

    def forward(self, sample, mode='single'):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.node_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.node_re_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.node_embedding,
                dim=0,
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample

            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.node_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.node_re_embedding,
                dim=0,
                index=tail_part[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.node_embedding,
                dim=0,
                index=tail_part[:,2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample

            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)


            head = torch.index_select(
                self.node_embedding,
                dim=0,
                index=head_part[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.node_re_embedding,
                dim=0,
                index=head_part[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.node_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func={
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail,mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = -torch.norm(score, p=2, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail,mode):
        re_head, im_head= torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)

        score = score.norm(dim = 0)
        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self,  head, relation, tail, mode):
        pi = 3.14159262358979323846

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)
        score = self.gamma.item() - score.sum(dim=2) * self.modulis

        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight= subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)
        positive_sample_loss = positive_score
        negative_sample_loss = negative_score
        gg = torch.ones((2))
        gg = gg.new_full(positive_score.size(), model.gamma_intra).cuda()
        loss = F.relu(gg - positive_score + negative_score).mean()

        if args.regularization != 0.0:
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + model.relation_embedding.norm(p = 3).norm(p = 3) **3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        loss.backward()
        optimizer.step()
        log={
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.mean().item(),
            'negative_sample_loss': negative_sample_loss.mean().item(),
            'loss': loss.item()
        }
        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):

        model.eval()
        if args.fl == 0: # entity
            test_dataloader_head = DataLoader(
                TestDataset_en(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nentity_re,
                    args.negative_sample_size_en_test,
                    'head-batch'
                ),
                batch_size=args.test_batch_size_en,
                num_workers=max(1,args.cpu_num//2),
                collate_fn=TestDataset_en.collate_fn
            )
            test_dataloader_tail = DataLoader(
                TestDataset_en(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nentity_re,
                    args.negative_sample_size_en_test,
                    'tail-batch'
                ),
                batch_size = args.test_batch_size_en,
                num_workers = max(1, args.cpu_num//2),
                collate_fn = TestDataset_en.collate_fn
            )
        else:
            test_dataloader_head = DataLoader(
                TestDataset_ty(
                    test_triples,
                    all_true_triples,
                    args.ntype,
                    args.ntype_re,
                    args.negative_sample_size_ty_test,
                    'head-batch'
                ),
                batch_size=args.test_batch_size_ty,
                num_workers=max(1,args.cpu_num//2),
                collate_fn=TestDataset_ty.collate_fn
            )
            test_dataloader_tail = DataLoader(
                TestDataset_ty(
                    test_triples,
                    all_true_triples,
                    args.ntype,
                    args.ntype_re,
                    args.negative_sample_size_ty_test,
                    'tail-batch'
                ),
                batch_size = args.test_batch_size_ty,
                num_workers = max(1, args.cpu_num//2),
                collate_fn = TestDataset_ty.collate_fn
            )


        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                    batch_size = positive_sample.size(0)
                    
                    score = model((positive_sample, negative_sample),mode)
                    score = score + filter_bias

                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:,0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:,2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        ranking = 1 + ranking.item()
                        
                    if args.fl == 0:
                        
                        logs.append(
                            {
                                'MRR_entity':1.0/ranking,
                                'MR_entity':float(ranking),
                                'HIT@1_entity':1.0 if ranking <= 1 else 0.0,
                                'HIT@3_entity':1.0 if ranking <= 3 else 0.0,
                                'HIT@10_entity':1.0 if ranking <= 10 else 0.0,
                            }
                        )
                        if step % args.test_log_steps_en == 0:
                            logging.info('Evaluating the model in entity graph... (%d/%d)' % (step, total_steps))
                    else:
                        
                        logs.append(
                            {
                                'MRR_type':1.0/ranking,
                                'MR_type':float(ranking),
                                'HIT@1_type':1.0 if ranking <= 1 else 0.0,
                                'HIT@3_type':1.0 if ranking <= 3 else 0.0,
                                'HIT@10_type':1.0 if ranking <= 10 else 0.0,
                            }
                        )
                        if step % args.test_log_steps_ty == 0:
                            logging.info('Evaluating the model in type graph... (%d/%d)' % (step, total_steps))

                    step = step + 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)


        return metrics
























