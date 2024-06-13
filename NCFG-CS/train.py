import argparse
import os
import sys
import os.path
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn.functional as F
from torch.nn import MarginRankingLoss, CrossEntropyLoss
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前目录路径添加到 sys.path 的开头
sys.path.insert(0, current_dir)
from data.CodeSearchNetDataSet import CodeSearchNetDataSet
from losses import CosineLoss
from model.GCNfusion import GCNfusion
from data.DataLoader import DataLoader

import random
import numpy as np
import math
# seed all

seed = 123456
random.seed(seed)
torch.random.manual_seed(seed)
np.random.seed(seed)
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def train(run_args):
    # get the command params
    batch_size = int(run_args.batch_size)
    num_epoch = int(run_args.num_epoch)
    emb_size = int(run_args.emb_size)
    hidden_dim = int(run_args.hidden_dim)
    pool_size = int(run_args.pool_size)
    
    current_dir = os.path.dirname(__file__)
     
    # data preprocessing 
    train_dataset = CodeSearchNetDataSet(
        os.path.join(current_dir, 'preprocessed_data/CSN/train'), 'train')
    valid_dataset = CodeSearchNetDataSet(
        os.path.join(current_dir, 'preprocessed_data/CSN/valid'), 'valid')
    
    # load the data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=True)
    print("data is already proprecessed！")
    
    # define neural networks
    model = GCNfusion(device=device, embedding_size=emb_size, hidden_dim=hidden_dim)
    
    model.to(device)
    loss_type = run_args.loss_type
    
    # set loss function
    if loss_type == 'ce':
        loss_func = CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters())
    elif loss_type == 'mrl':
        loss_func = MarginRankingLoss(margin=0.4)
        opt = torch.optim.Adam(model.parameters())
    elif loss_type == 'cos':
        loss_func = CosineLoss()
        opt = torch.optim.Adam(model.parameters())
    else:
        raise RuntimeError(f'Unsupported loss type \'{loss_type}\'')
    # the high learning rate will lead to too fast convergence
    total_steps = len(train_dataloader) * num_epoch
    num_cycles = 0.01            
    scheduler = get_cosine_schedule_with_warmup(opt, 100, total_steps, num_cycles)
    
    print('train set size：', len(train_dataset))
    print('valide set size：', len(valid_dataset))
    
    best_mrr = 0
    best_epoch = 0
    train_losses = []
    validate_results = []

    for epoch in range(1, num_epoch + 1):
        print(f'Epoch {epoch}:')
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        epoch_loss = 0
        count = 0
        for batch_graphs, batch_pos_desc, batch_neg_desc, batch_pos_lens, batch_neg_lens in progress_bar:
            count += 1
            model.train()        # update model params
            if loss_type == 'mrl':
                pos_scores = model(batch_graphs.to(device), batch_pos_desc.to(device), batch_pos_lens.to(device))
                neg_scores = model(batch_graphs.to(device), batch_neg_desc.to(device), batch_neg_lens.to(device))
                loss = loss_func(pos_scores, neg_scores, torch.ones(batch_graphs.num_graphs).to(device))
            elif loss_type == 'ce':
                similarities = model(batch_graphs.to(device), batch_pos_desc.to(device),
                                     batch_pos_lens.to(device), cross_entropy=True, normalize_sim_matrix=False)
                labels = torch.arange(0, batch_pos_desc.size()[0]).to(device)
                loss = loss_func(similarities, labels)
            elif loss_type == 'cos':
                similarities = model(batch_graphs.to(device), batch_pos_desc.to(device),
                                     batch_pos_lens.to(device), cross_entropy=True, normalize_sim_matrix=True)
                loss = loss_func(similarities, device=device)
            else:
                raise RuntimeError(f'Unsupported loss type \'{loss_type}\'')
            current_lr = scheduler.get_last_lr()[0]
            print(f"current Epoch {count} Learning Rate: {current_lr}")
            epoch_loss += loss.item()
            progress_bar.set_description_str(f'loss:{epoch_loss / count}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()
        print('---model parameters---')
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        print(num_params / 1e6)
        print("model size：")
        print((4 * num_params) / (1024**3))
        train_losses.append(epoch_loss / len(train_dataloader))
        # after every epoch
        valid_result = validate(valid_dataset, model, pool_size, 1, 'cos')
        print('valid top 1 result:\n', valid_result)
        # valid_result2 = validate(valid_dataset, model, pool_size, 100, 'cos')
        # print('valid top 100 result:\n', valid_result2)
        validate_results.append(valid_result)
        if valid_result['mrr'] > best_mrr:
            best_mrr = valid_result['mrr']
            best_epoch = epoch
            print(f'Save epoch {epoch} model, the valid mrr is {best_mrr}.')
            if not os.path.exists(run_args.saved_model_dir):
                os.makedirs(run_args.saved_model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(run_args.saved_model_dir, f'best_model.bin'))
    print('Done!')
    print(f'best valid mrr: {best_mrr}, best epoch: {best_epoch}.')
    print(validate_results)


def validate(valid_set, model, pool_size, K, sim_measure):
    """
    simple validation in a code pool.
    @param: poolsize - size of the code pool, if -1, load the whole test set
    """

    def ACC(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + 1
        return sum / float(len(real))

    def MAP(real, predict):
        sum = 0.0
        for id, val in enumerate(real):
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum + (id + 1) / float(index + 1)
        return sum / float(len(real))

    def MRR(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
                # MRR_pool.append(index + 1)
            if index != -1:
                sum = sum + 1.0 / float(index + 1)
                # MRR_pool.append(index + 1)
        return sum / float(len(real))

    def NDCG(real, predict):
        dcg = 0.0
        idcg = IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i + 1
                dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
        return dcg / float(idcg)

    def IDCG(n):
        idcg = 0
        itemRelevance = 1
        for i in range(n):
            idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
        return idcg
    # model params could not updated
    model.eval()
    data_loader = DataLoader(dataset=valid_set, batch_size=pool_size,
                             shuffle=True, drop_last=True, num_workers=1)
    accs, mrrs, maps, ndcgs = [], [], [], []
    code_reprs, desc_reprs = [], []  # idxes record the best ranking result
    n_processed = 0
    for batch_graphs, batch_pos_desc, _, batch_pos_lens, _ in tqdm(data_loader,
                                                              total=len(data_loader)):
        with torch.no_grad():
            code_repr, desc_repr = model(batch_graphs.to(device),
                                         batch_pos_desc.to(device),
                                         batch_pos_lens.to(device),
                                         return_embedding=True)
            # desc_repr = model.encode_desc(batch_pos_desc.to(device), batch_pos_lens.to(device))
            code_repr = F.normalize(code_repr, dim=-1, p=2)
            desc_repr = F.normalize(desc_repr, dim=-1, p=2)
        code_reprs.append(code_repr.cpu().numpy())
        desc_reprs.append(desc_repr.cpu().numpy())
        # idxes.append(tem_idxes.cpu().numpy())           # get the real first one according index
        n_processed += batch_graphs.num_graphs
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)
    # idxes = idxes.flatten()
    
    for k in tqdm(range(0, n_processed, pool_size)):
        code_pool, desc_pool = code_reprs[k:k + pool_size], desc_reprs[k:k + pool_size]
        # rank_pool = []              # actrual predict rank
        for i in range(min(20000, pool_size)):  # for i in range(pool_size):
            desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]
            n_results = K
            sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]
            negsims = np.negative(sims)
            predict_origin = np.argsort(negsims)  # predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict_origin[:n_results]
            predict = [int(k) for k in predict]
            predict_origin = [int(k) for k in predict_origin]
            # predict = np.argpartition(negsims, kth=n_results - 1)  # predict=np.argsort(negsims)#
            # predict = predict[:n_results]
            # predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
    return {'acc': np.mean(accs), 'mrr': np.mean(mrrs),
            'map': np.mean(maps), 'ndcg': np.mean(ndcgs)}


def test(run_args):
    emb_size = int(run_args.emb_size)
    pool_size = int(run_args.pool_size)
    hidden_dim = int(run_args.hidden_dim)

    # set data preprocessing and model
    test_dataset = CodeSearchNetDataSet(os.path.join(current_dir, 'preprocessed_data/CSN/test'), 'test')
    model = GCNfusion(device=device, embedding_size=emb_size, hidden_dim=hidden_dim)
    
    print("test set size：", len(test_dataset))
    model.load_state_dict(torch.load(run_args.test_model_path))
    model.to(device)
    top_k = 1
    print(f'top {top_k}:\n', validate(test_dataset, model, pool_size, top_k, 'cos'))
    top_k = 5
    print(f'top {top_k}:\n', validate(test_dataset, model, pool_size, top_k, 'cos'))
    top_k = 10
    print(f'top {top_k}:\n', validate(test_dataset, model, pool_size, top_k, 'cos'))
    top_k = 100
    print(f'top {top_k}:\n', validate(test_dataset, model, pool_size, top_k, 'cos'))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == '__main__':
    print("start training ！")
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', dest='test', help='test mode', action='store_true')
    parser.add_argument('-train', dest='train', help='train mode', action='store_true')
    parser.add_argument('-num_epoch', dest='num_epoch', action='store', type=int, help='train epoch num', default=100)
    parser.add_argument('-batch_size', dest='batch_size', action='store', type=int,
                        help='train and dev batch size', default=128)
    parser.add_argument('-pool_size', dest='pool_size', type=int, default=2000)
    parser.add_argument('-emb_size', dest='emb_size', default=128, type=int)
    parser.add_argument('-hidden_dim', dest='hidden_dim', default=256, type=int)
    parser.add_argument('-loss_type', dest='loss_type', help='select loss', default='cos', action='store',
                        choices={'ce', 'mrl', 'cos'})
    parser.add_argument('-use_desc_lstm', dest='use_desc_lstm', default=False, required=False, type=bool, help='')
    parser.add_argument('-test_model_path', dest='test_model_path', required=False,
                        type=str, action='store', help='test model store path')
    parser.add_argument('-saved_model_dir', dest='saved_model_dir', required=False,
                        type=str, action='store', help='model save directory')
    # parser.add_argument('-gpu', dest='gpu', action='store', type=int, default=0, help='specify gpu id')
    run_args = parser.parse_args()
    print('params setting\n', run_args)
    if run_args.train:
        train(run_args)
    if run_args.test:
        test(run_args)

