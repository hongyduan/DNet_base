import os
import argparse
import logging
import torch
import json
import numpy as np
from model import DKGE_Model
from torch.utils.data import DataLoader
from dataloader import TrainDataset_en, TrainDataset_ty, TestDataset_en, TestDataset_ty
from dataloader import BidirectionalOneShotIterator

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--fl', default=0, type=int) #0:en, 1:ty
    parser.add_argument('-r_e', '--regularization', default=0.0, type=float)

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--data_path', type=str, default="/storage/hyduan/DNet/DNet_base/data/dbpedia_result")
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nentity_re', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--ntype', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--ntype_re', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--do_train_en', action='store_true', default=True)
    parser.add_argument('--do_train_ty', action='store_true', default=True)
    parser.add_argument('--do_valid_en', action='store_true', default=True)
    parser.add_argument('--do_valid_ty', action='store_true', default=True)
    parser.add_argument('--do_test_en', action='store_true', default=True)
    parser.add_argument('--do_test_ty', action='store_true', default=True)
    parser.add_argument('--entity_model', default='TransE', type=str)
    parser.add_argument('--type_model', default='TransE', type=str)
    parser.add_argument('-n_en', '--negative_sample_size_en', default=128, type=int)
    parser.add_argument('-n_en_test', '--negative_sample_size_en_test', default=200, type=int)
    parser.add_argument('-n_ty', '--negative_sample_size_ty', default=128, type=int)
    parser.add_argument('-n_ty_test', '--negative_sample_size_ty_test', default=200, type=int)
    parser.add_argument('-den', '--hidden_dim_en', default=500, type=int)
    parser.add_argument('-dty', '--hidden_dim_ty', default=500, type=int)
    parser.add_argument('-g_en', '--gamma_en', default=12.0, type=float)
    parser.add_argument('-g_ty', '--gamma_ty', default=12.0, type=float)
    parser.add_argument('-b_en', '--batch_size_en', default=1024, type=int)
    parser.add_argument('-b_ty', '--batch_size_ty', default=512, type=int)
    parser.add_argument('--test_batch_size_en', default=4, type=int, help='valid/test batch size in entity graph')
    parser.add_argument('--test_batch_size_ty', default=4, type=int, help='valid/test batch size in type graph')
    parser.add_argument('--uni_weight_en', action='store_true', help='Otherwise use subsampling weighting like in word2vec in entity graph')
    parser.add_argument('--uni_weight_ty', action='store_true', help='Otherwise use subsampling weighting like in word2vec in type graph')
    parser.add_argument('-lr_en', '--learning_rate_en', default=0.001, type=float)
    parser.add_argument('-lr_ty', '--learning_rate_ty', default=0.001, type=float)
    parser.add_argument('-init_en', '--init_checkpoint_en', default=None, type=str)
    parser.add_argument('-init_ty', '--init_checkpoint_ty', default=None, type=str)
    parser.add_argument('-save_en', '--save_path_en', default="/storage/hyduan/DNet/DNet_base/save/en", type=str)
    parser.add_argument('-save_ty', '--save_path_ty', default="/storage/hyduan/DNet/DNet_base/save/ty", type=str)
    parser.add_argument('--max_steps_en', default=109400, type=int)
    parser.add_argument('--max_steps_ty', default=1000, type=int)
    parser.add_argument('--warm_up_steps_en', default=None, type=int)
    parser.add_argument('--warm_up_steps_ty', default=None, type=int)
    parser.add_argument('--save_checkpoint_steps_en', default=5000, type=int)
    parser.add_argument('--save_checkpoint_steps_ty', default=50, type=int)
    parser.add_argument('--valid_steps_en', default=1000, type=int)
    parser.add_argument('--valid_steps_ty', default=10, type=int)
    parser.add_argument('--log_steps_en', default=500, type=int, help='train log every xx steps in entity graph')
    parser.add_argument('--log_steps_ty', default=10, type=int, help='train log every xx steps in type graph')
    parser.add_argument('--test_log_steps_en', default=60, type=int, help='valid/test log every xx steps in entity graph')
    parser.add_argument('--test_log_steps_ty', default=2, type=int, help='valid/test log every xx steps in type graph')

    parser.add_argument('-edou', '--double_node_embedding_en', action='store_true')
    parser.add_argument('-erdou', '--double_node_re_embedding_en', action='store_true')
    parser.add_argument('-tdou', '--double_node_embedding_ty', action='store_true')
    parser.add_argument('-trdou', '--double_node_re_embedding_ty', action='store_true')

    return parser.parse_args(args)

def override_config(args):
    if args.fl == 0:
        with open(os.path.join(args.init_checkpoint_en, 'config.json'), 'r') as fjson:
            argparse_dict_en = json.load(fjson)
        if args.data_path is None:
            args.data_path = argparse_dict_en['data_path']
        args.model_en = argparse_dict_en['entity_model']
        args.double_node_embedding_en = argparse_dict_en['double_node_embedding_en']
        args.double_node_re_embedding_en = argparse_dict_en['double_node_re_embedding_en']
        args.hidden_dim_en = argparse_dict_en['hidden_dim_en']
        args.test_batch_size_en = argparse_dict_en['test_batch_size_en']
    else:
        with open(os.path.join(args.init_checkpoint_ty, 'config.json'), 'r') as fjson:
            argparse_dict_ty = json.load(fjson)
        if args.data_path is None:
            args.data_path = argparse_dict_ty['data_path']
        args.model_ty = argparse_dict_ty['type_model']
        args.double_node_embedding_ty = argparse_dict_ty['double_node_embedding_ty']
        args.double_node_re_embedding_ty = argparse_dict_ty['double_node_re_embedding_ty']
        args.hidden_dim_ty = argparse_dict_ty['hidden_dim_ty']
        args.test_batch_size_ty = argparse_dict_ty['test_batch_size_ty']


def save_model(model, optimizer, save_variable_list, args):

    # fl == 0: entity graph
    if args.fl == 0:
        argparse_dict = vars(args)
        with open(os.path.join(args.save_path_en, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path_en, 'checkpoint_en')
        )
        entity_embedding = model.node_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path_en, 'node_embedding'),
            entity_embedding
        )
        en_en_relation_embedding = model.node_re_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path_en, 'node_re_embedding'),
            en_en_relation_embedding
        )
    # fl == 1: type graph
    else:
        argparse_dict = vars(args)
        with open(os.path.join(args.save_path_ty, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)
        torch.save({
            **save_variable_list,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            os.path.join(args.save_path_ty, 'checkpoint_ty')
        )
        type_embedding = model.node_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path_ty, 'node_embedding'),
            type_embedding
        )
        ty_ty_relation_embedding = model.node_re_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path_ty, 'ty_ty_relation_embedding'),
            ty_ty_relation_embedding
        )


def read_triple(file_path, node2id, node_relation2id):
    triples = []
    with open(file_path, encoding='utf8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((node2id[h], node_relation2id[r], node2id[t]))
    return triples


def read_en_ty_blocks_triple(file_path, entity2id, type2id):
    triples = []
    temporary_type = "a"
    with open(file_path) as fin:
        for line in fin:
            first_str = line.strip().split('\t')[0]
            if first_str == "###":
                temporary_type = line.strip().split('\t')[1]
            else:
                triples.append((type2id[temporary_type],entity2id[first_str]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    # 0: entity
    if args.fl == 0:
        if args.do_train_en:
            log_file_en = os.path.join(args.save_path_en or args.init_checkpoint_en, 'train_en.log')
        else:
            log_file_en = os.path.join(args.save_path_en or args.init_checkpoint_en, 'test_en.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file_en,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    # 1: type
    else:
        if args.do_train_ty:
            log_file_ty = os.path.join(args.save_path_ty or args.init_checkpoint_ty, 'train_ty.log')
        else:
            log_file_ty = os.path.join(args.save_path_ty or args.init_checkpoint_ty, 'test_ty.log')

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file_ty,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)



def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def main(args):



    # fl == 0 : entity graph
    if args.fl == 0:

        if (not args.do_train_en) and (not args.do_valid_en) and (not args.do_test_en):
            raise ValueError('One of train/valid/test mode must be choosed.')

        if args.init_checkpoint_en:
            override_config(args)
        elif args.data_path is None:
            raise ValueError('One of init_checkpoint/data_path must be choosed.')

        if args.do_train_en and args.save_path_en is None:
            raise ValueError('Where do you want to save your trained model?')

        if args.save_path_en and not os.path.exists(args.save_path_en):
            os.makedirs(args.save_path_en)

        set_logger(args)
        # 获取所有实体 final_entity.txt
        entity2id = dict()
        with open(os.path.join(args.data_path, 'final_entity_order.txt'), encoding='utf8') as fin:
            for line in fin.readlines():
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)
        nentity = len(entity2id)

        # 获取所有entity_relation 从ffinal_en_relation_order.txt
        entity_relation2id = dict()
        with open(os.path.join(args.data_path, 'ffinal_en_relation_order.txt')) as fin:
            for line in fin:
                en_reid, entity_relation = line.strip().split('\t')
                entity_relation2id[entity_relation] = int(en_reid)
        nentity_relation = len(entity_relation2id)

        args.nentity = nentity
        args.nentity_re = nentity_relation

        logging.info('entity_Model: %s' % args.entity_model)
        logging.info('Data Path: %s' % args.data_path)
        logging.info('number of entity: %d' % nentity)
        logging.info('number of entity_re: %d' % nentity_relation)

        #加载训练三元组集合，
        train_entity_triples = read_triple(os.path.join(args.data_path, 'train_entity_Graph.txt'), entity2id, entity_relation2id)
        logging.info('#train en_en triples: %s' % len(train_entity_triples))
        #加载valid三元组集合，
        val_entity_triples = read_triple(os.path.join(args.data_path, 'val_entity_Graph.txt'), entity2id, entity_relation2id)
        logging.info('#val en_en triples: %s' % len(val_entity_triples))
        #加载test三元组集合，
        test_entity_triples= read_triple(os.path.join(args.data_path, 'test_entity_Graph.txt'), entity2id, entity_relation2id)
        logging.info('#test en_en triples: %s' % len(test_entity_triples))
        # all_triples
        all_true_triples_entity = train_entity_triples + val_entity_triples + test_entity_triples

        # entity model
        entity_kge_model = DKGE_Model(
            model_name = args.entity_model,
            nnode = args.nentity,
            nnode_re = args.nentity_re,
            hidden_dim = args.hidden_dim_en,
            gamma = args.gamma_en,
            double_node_embedding=args.double_node_embedding_en,
            double_node_re_embedding=args.double_node_re_embedding_en
        )
        logging.info('Entity Model Parameter Configuration:')
        for name, param in entity_kge_model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if args.cuda:
            entity_kge_model = entity_kge_model.cuda()

        if args.do_train_en:
            train_dataloader_head_en = DataLoader(
                TrainDataset_en(train_entity_triples, nentity, nentity_relation, args.negative_sample_size_en, 'head-batch'),
                batch_size=args.batch_size_en,
                shuffle=True,
                num_workers=int(max(1, args.cpu_num//2)),
                collate_fn=TrainDataset_en.collate_fn
            )
            train_dataloader_tail_en = DataLoader(
                TrainDataset_en(train_entity_triples, nentity, nentity_relation, args.negative_sample_size_en, 'tail-batch'),
                batch_size=args.batch_size_en,
                shuffle=True,
                num_workers=int(max(1, args.cpu_num//2)),
                collate_fn=TrainDataset_en.collate_fn
            )
            train_iterator_en = BidirectionalOneShotIterator(train_dataloader_head_en, train_dataloader_tail_en)
            # Set training configuration
            current_learning_rate_en = args.learning_rate_en
            optimizer_en = torch.optim.Adam(
                filter(lambda p: p.requires_grad, entity_kge_model.parameters()),
                lr=current_learning_rate_en
            )
            if args.warm_up_steps_en:
                warm_up_steps_en = args.warm_up_steps_en
            else:
                warm_up_steps_en = args.max_steps_en // 2

        if args.init_checkpoint_en:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % args.init_checkpoint_en)
            checkpoint_en = torch.load(os.path.join(args.init_checkpoint_en, 'checkpoint_en'))
            init_step_en = checkpoint_en['step']
            entity_kge_model.load_state_dict(checkpoint_en['model_state_dict'])
            if args.do_train_en:
                current_learning_rate_en = checkpoint_en['current_learning_rate']
                warm_up_steps_en = checkpoint_en['warm_up_steps']
                optimizer_en.load_state_dict(checkpoint_en['optimizer_state_dict'])
        else:
            logging.info('Ramdomly Initializing %s Model...' % args.entity_model)
            init_step_en = 0

        step_en = init_step_en
        logging.info('Start Training of entity graph...')
        logging.info('init_step_en = %d' % init_step_en)
        logging.info('learning_rate_en = %f' % current_learning_rate_en)
        logging.info('batch_size_en = %d' % args.batch_size_en)
        logging.info('hidden_dim_en = %d' % args.hidden_dim_en)
        logging.info('gamma_en = %f' % args.gamma_en)

        if args.do_train_en:
            training_logs_en = []
            for step_en in range(init_step_en, args.max_steps_en):
                log_en = entity_kge_model.train_step(entity_kge_model, optimizer_en, train_iterator_en, args)
                training_logs_en.append(log_en)

                if step_en >= warm_up_steps_en:
                    current_learning_rate_en = current_learning_rate_en / 10
                    logging.info('Changing learning_rate_en to %f at step %d' % (current_learning_rate_en, step_en))
                    optimizer_en = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, entity_kge_model.parameters()),
                        lr=current_learning_rate_en
                    )
                    warm_up_steps_en = warm_up_steps_en * 3

                if step_en % args.save_checkpoint_steps_en == 0:
                    save_variable_list_en = {
                        'step': step_en,
                        'current_learning_rate': current_learning_rate_en,
                        'warm_up_steps': warm_up_steps_en
                    }
                    save_model(entity_kge_model, optimizer_en, save_variable_list_en, args)

                if step_en % args.log_steps_en == 0:
                    metrics_en = {}
                    for metric_en in training_logs_en[0].keys():
                        metrics_en[metric_en] = sum([log_en[metric_en] for log_en in training_logs_en]) / len(
                            training_logs_en)
                    log_metrics('Training average in entity graph', step_en, metrics_en)
                    training_logs_en = []

                if args.do_valid_en and step_en % args.valid_steps_en == 0 and step_en!=0 and False:
                    logging.info('Evaluating on Valid Dataset in entity graph ...')
                    metrics_en = entity_kge_model.test_step(entity_kge_model, val_entity_triples, all_true_triples_entity, args)
                    log_metrics('Valid', step_en, metrics_en)

            save_variable_list_en = {
                'step': step_en,
                'current_learning_rate': current_learning_rate_en,
                'warm_up_steps': warm_up_steps_en
            }
            save_model(entity_kge_model, optimizer_en, save_variable_list_en, args)

        if args.do_valid_en and False:
            logging.info('Evaluating on Valid Dataset in entity graph...')
            metrics_en = entity_kge_model.test_step(entity_kge_model, val_entity_triples, all_true_triples_entity, args)
            log_metrics('Valid', step_en, metrics_en)

        if args.do_test_en:
            logging.info('Testing on Test Dataset in entity graph...')
            metrics_en = entity_kge_model.test_step(entity_kge_model, test_entity_triples, all_true_triples_entity, args)
            log_metrics('Test', step_en, metrics_en)

    # type
    else:
        if (not args.do_train_ty) and (not args.do_valid_ty) and (not args.do_test_ty):
            raise ValueError('One of train/valid/test mode must be choosed.')

        if args.init_checkpoint_ty:
            override_config(args)
        elif args.data_path is None:
            raise ValueError('One of init_checkpoint/data_path must be choosed.')

        if args.do_train_ty and args.save_path_ty is None:
            raise ValueError('Where do you want to save your trained model?')

        if args.save_path_ty and not os.path.exists(args.save_path_ty):
            os.makedirs(args.save_path_ty)

        set_logger(args)
        # 获取所有type 从final_type.txt
        type2id = dict()
        with open(os.path.join(args.data_path, 'final_type_order.txt')) as fin:
            for line in fin:
                tid, type = line.strip().split('\t')
                type2id[type] = int(tid)
        ntype = len(type2id)

        # 获取所有type_relation 从ffinal_ty_relation_order.txt
        type_relation2id = dict()
        with open(os.path.join(args.data_path, 'ffinal_ty_relation_order.txt')) as fin:
            for line in fin:
                ty_reid, type_relation = line.strip().split('\t')
                type_relation2id[type_relation] = int(ty_reid)
        ntype_relation = len(type_relation2id)

        args.ntype = ntype
        args.ntype_re = ntype_relation

        logging.info('type_Model: %s' % args.type_model)
        logging.info('Data Path: %s' % args.data_path)
        logging.info('number of type: %d' % ntype)
        logging.info('number of type_re: %d' % ntype_relation)

        #加载训练三元组集合，
        train_type_triples = read_triple(os.path.join(args.data_path, 'train_type_Graph.txt'), type2id, type_relation2id)
        logging.info('#train ty_ty triples: %s' % len(train_type_triples))
        #加载valid三元组集合，
        val_type_triples = read_triple(os.path.join(args.data_path, 'val_type_Graph.txt'), type2id, type_relation2id)
        logging.info('#val ty_ty triples: %s' % len(val_type_triples))
        #加载test三元组集合，
        test_type_triples = read_triple(os.path.join(args.data_path, 'test_type_Graph.txt'), type2id, type_relation2id)
        logging.info('#test ty_ty triples: %s' % len(test_type_triples))
        # all_triples
        all_true_triples_type = train_type_triples + val_type_triples + test_type_triples

        # type model
        type_kge_model = DKGE_Model(
            model_name = args.type_model,
            nnode = args.ntype,
            nnode_re = args.ntype_re,
            hidden_dim = args.hidden_dim_ty,
            gamma = args.gamma_ty,
            double_node_embedding=args.double_node_embedding_ty,
            double_node_re_embedding=args.double_node_re_embedding_ty
        )
        logging.info('Type Model Parameter Configuration:')
        for name, param in type_kge_model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if args.cuda:
            type_kge_model = type_kge_model.cuda()

        if args.do_train_ty:
            train_dataloader_head_ty = DataLoader(
                TrainDataset_ty(train_type_triples, ntype, ntype_relation, args.negative_sample_size_ty, 'head-batch'),
                batch_size=args.batch_size_ty,
                shuffle=True,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TrainDataset_ty.collate_fn
            )
            train_dataloader_tail_ty = DataLoader(
                TrainDataset_ty(train_type_triples, ntype, ntype_relation, args.negative_sample_size_ty, 'tail-batch'),
                batch_size=args.batch_size_ty,
                shuffle=True,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TrainDataset_ty.collate_fn
            )
            train_iterator_ty = BidirectionalOneShotIterator(train_dataloader_head_ty, train_dataloader_tail_ty)
            # Set training configuration
            current_learning_rate_ty = args.learning_rate_ty
            optimizer_ty = torch.optim.Adam(
                filter(lambda p: p.requires_grad, type_kge_model.parameters()),
                lr=current_learning_rate_ty
            )
            if args.warm_up_steps_ty:
                warm_up_steps_ty = args.warm_up_steps_ty
            else:
                warm_up_steps_ty = args.max_steps_ty // 2

        if args.init_checkpoint_ty:
            # Restore model from checkpoint directory
            logging.info('Loading checkpoint %s...' % args.init_checkpoint_ty)
            checkpoint_ty = torch.load(os.path.join(args.init_checkpoint_ty, 'checkpoint_ty'))
            init_step_ty = checkpoint_ty['step']
            type_kge_model.load_state_dict(checkpoint_ty['model_state_dict'])
            if args.do_train:
                current_learning_rate_ty = checkpoint_ty['current_learning_rate']
                warm_up_steps = checkpoint_ty['warm_up_steps']
                optimizer_ty.load_state_dict(checkpoint_ty['optimizer_state_dict'])
        else:
            logging.info('Ramdomly Initializing %s Model...' % args.type_model)
            init_step_ty = 0

        step_ty = init_step_ty
        logging.info('Start Training of type graph...')
        logging.info('init_step_ty = %d' % init_step_ty)
        logging.info('learning_rate_ty = %f' % current_learning_rate_ty)
        logging.info('batch_size_ty = %d' % args.batch_size_ty)
        logging.info('hidden_dim_ty = %d' % args.hidden_dim_ty)
        logging.info('gamma_ty = %f' % args.gamma_ty)

        if args.do_train_ty:
            training_logs_ty = []
            for step_ty in range(init_step_ty, args.max_steps_ty):
                log_ty = type_kge_model.train_step(type_kge_model, optimizer_ty, train_iterator_ty, args)
                training_logs_ty.append(log_ty)

                if step_ty >= warm_up_steps_ty:
                    current_learning_rate_ty = current_learning_rate_ty / 10
                    logging.info('Changing learning_rate_ty to %f at step %d' % (current_learning_rate_ty, step_ty))
                    optimizer_ty = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, type_kge_model.parameters()),
                        lr = current_learning_rate_ty
                    )
                    warm_up_steps_ty = warm_up_steps_ty * 3

                if step_ty % args.save_checkpoint_steps_ty == 0:
                    save_variable_list_ty = {
                        'step': step_ty,
                        'current_learning_rate': current_learning_rate_ty,
                        'warm_up_steps': warm_up_steps_ty
                    }
                    save_model(type_kge_model, optimizer_ty, save_variable_list_ty, args)

                if step_ty % args.log_steps_ty == 0:
                    metrics_ty = {}
                    for metric_ty in training_logs_ty[0].keys():
                        metrics_ty[metric_ty] = sum([log_ty[metric_ty] for log_ty in training_logs_ty])/len(training_logs_ty)
                    log_metrics('Training average in type graph', step_ty, metrics_ty)
                    training_logs_ty = []

                if False and args.do_valid_ty and step_ty % args.valid_steps_ty == 0:
                    logging.info('Evaluating on Valid Dataset in type graph ...')
                    metrics_ty = type_kge_model.test_step(type_kge_model, val_type_triples, all_true_triples_type, args)
                    log_metrics('Valid', step_ty, metrics_ty)

            save_variable_list_ty = {
                'step': step_ty,
                'current_learning_rate': current_learning_rate_ty,
                'warm_up_steps': warm_up_steps_ty
            }
            save_model(type_kge_model, optimizer_ty, save_variable_list_ty, args)

        if args.do_valid_ty and False:
            logging.info('Evaluating on Valid Dataset in type graph...')
            metrics_ty = type_kge_model.test_step(type_kge_model, val_type_triples, all_true_triples_type, args)
            log_metrics('Valid', step_ty, metrics_ty)

        if args.do_test_ty:
            logging.info('Testing on Test Dataset in type graph...')
            metrics_ty = type_kge_model.test_step(type_kge_model, test_type_triples, all_true_triples_type, args)
            log_metrics('Test', step_ty, metrics_ty)


if __name__ == '__main__':
    main(parse_args())
