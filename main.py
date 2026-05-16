import os
import os.path as osp
import time
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from dataset import GraphDataset
from dataset import SequentialSampler, SubsetRandomSampler
from torch_geometric.loader import DataLoader

from models import GCNNet, GINNet, GATNet, MDGCNNet, MDGINNet, MDGATNet
from strategies import query
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

models = {}
models['gcn'] = (GCNNet)
models['gin'] = (GINNet)
models['gat'] = (GATNet)
models['mdgcn'] = (GCNNet)
models['mdgin'] = (GINNet)
models['mdgat'] = (GATNet)

methods = ['random', 'entropy', 'coreset', 'mdnal']

writer = SummaryWriter()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)

def train(args, model, data, optimizer):
    device = args.device

    if args.method == 'mdnal':
        model.train()

        data = data.to(device)
        target = data.y.reshape(-1).to(device) # [B]

        optimizer.zero_grad()

        logits_pi, logits_mu = model(data)
        # 1. 混合权重
        pi = torch.softmax(logits_pi, dim=-1)  # [B, K]
        # 2. 类别分布参数
        mu = torch.softmax(logits_mu, dim=-1)  # [B, K, C]
        # 3. GT
        target_onehot = F.one_hot(target, num_classes=mu.size(2)).float() # [B, C]
        # 4. 计算单个成分似然
        likelihood = torch.sum(mu * target_onehot.unsqueeze(1), dim=-1) # [B, K]
        # 5. 计算混合似然
        likelihoods = torch.sum(pi * likelihood, dim=-1) # [B]
        likelihoods = torch.clamp(likelihoods, min=1e-8) # 数值稳定性
        # 6. 负对数损失
        loss=-torch.log(likelihoods).mean()
        loss.backward()

        optimizer.step()

    else:
        criterion = torch.nn.CrossEntropyLoss()
        model.train()

        data = data.to(device)
        target = data.y.reshape(-1).to(device)

        optimizer.zero_grad()

        output, embed = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

    return loss.item()

def evaluate(args, model, data):
    device = args.device

    if args.method == 'mdnal':
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            target = data.y.reshape(-1).to(device) # [B]

            logits_pi, logits_mu = model(data)
            # 1. 混合权重
            pi = torch.softmax(logits_pi, dim=-1)  # [B, K]
            # 2. 类别分布参数
            mu = torch.softmax(logits_mu, dim=-1)  # [B, K, C]
            # 3. 类别概率分布
            prob = torch.sum(pi.unsqueeze(-1) * mu, dim=1) # [B, C]
            # 4. 取最大概率
            pred = torch.argmax(prob, dim=-1)  # [B]
        return pred.cpu(), target.cpu(), prob.cpu()
    else:
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            target = data.y.reshape(-1).to(device)

            logit, embed = model(data)
            pred = logit.max(1)[1]
            
        return pred.cpu(), target.cpu(), logit.cpu()

def calculate_metrics(args, preds, gts, logits, num_classes=3):
    # Confusion Matrix
    cm = confusion_matrix(gts, preds, labels=np.arange(num_classes))
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (fp + fn + tp)

    # ACC, SEN, SPE, PRE, F1
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1_scores = 2 * (precision * sensitivity) / (precision + sensitivity)

    average_accuracy = np.sum(tp) / np.sum(cm)
    average_sensitivity = np.mean(sensitivity)
    average_specificity = np.mean(specificity)
    average_precision = np.mean(precision)
    average_f1_score = np.mean(f1_scores)

    # AUROC
    auroc_scores = []
    gts = np.array(gts)
    logits = np.concatenate(logits, axis=0)
    if args.method == 'mdnal':
        probabilities = logits
    else:
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    for i in range(num_classes):
        binary_y = (gts == i).astype(int)
        auroc = roc_auc_score(binary_y, probabilities[:, i])
        auroc_scores.append(auroc)

    average_auroc_scores = np.mean(auroc_scores)

    # AUPRC
    auprc_scores = []
    for i in range(num_classes):
        binary_y = (gts == i).astype(int)
        auprc = average_precision_score(binary_y, probabilities[:, i])
        auprc_scores.append(auprc)

    average_auprc_scores = np.mean(auprc_scores)

    return cm, average_accuracy, average_sensitivity, average_specificity, average_precision, average_f1_score, average_auroc_scores, average_auprc_scores


def plot_confusion_matrix(cm):
    """将混淆矩阵绘制为热力图"""
    fig, ax = plt.subplots()
    # 使用imshow创建热力图
    im = ax.imshow(cm, cmap='Blues')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # 设置标签
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    
    # 添加文本标签
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='PyTorch GraNet for sparse training')
    # Data Settings
    parser.add_argument('--data', type=str, default='CPTAC', choices=['CPTAC', 'TCGA', 'CPTAC2TCGA'])
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 100)')
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer to use. Default: adam.')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--cuda', type=int, default=0, help='CUDA id')
    # parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    # parser.add_argument('--save', type=str, default=''.join(str(time.time()).split('.')) + '.pt', help='path to save the final model')
    # AL Settings
    parser.add_argument('--method', type=str, default='mdnal', choices=['random', 'entropy', 'coreset', 'mdnal'])
    parser.add_argument('--rounds', type=int, default=9, help='Number of round of active learning.')
    parser.add_argument('--init', type=float, default=0.1, help='Number of initial of active learning.')
    parser.add_argument('--queries', type=float, default=0.1, help='Number of query of active learning.')
    # GCN+GAT：λ = 0.5, GIN：λ = 1.0
    parser.add_argument('--lambd', type=float, default=1.0, help='λ epistemic + (1-λ) aleatoric')   # λ=0.0, 0.25, 0.5, 0.75, 1.0
    # GCN+GAT：p = 1.5, GIN：p = 1.2
    parser.add_argument('--ratio', type=float, default=1.2, help='The ratio of query samples in stage one. (ratio >= 1.0)')   # p=1.0, 1.2, 1.5, 2.0, 3.0
    # MDN Settings
    parser.add_argument('--model', type=str, default='mdgin', choices=['gcn', 'gat', 'gin', 'mdgcn', 'mdgat', 'mdgin'])
    parser.add_argument('--dim', type=int, default=256, help='Feature dimensions of GNN layers.')
    parser.add_argument('--com', type=int, default=4, help='Components of MDN.')   # K=2,4,8,16,32,64
    parser.add_argument('--ns', type=int, default=10, help='Number of samples.')   # N=2,5,10,16,25
    
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.cuda) if use_cuda else "cpu")

    print('='*80)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    #######################################################################################
    ############################# Datasets ################################################
    #######################################################################################
    if args.data in ['CPTAC', 'TCGA', 'CPTAC2TCGA']:
        data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.data)

        train_ids = open(osp.join(data_path, 'train.txt')).readlines()
        trainset = GraphDataset(data_path, train_ids, train_val='train')
        test_ids = open(osp.join(data_path, 'test.txt')).readlines()
        testset = GraphDataset(data_path, test_ids, train_val='test')

        indices = list(range(len(trainset)))
        random.shuffle(indices)
        labeled_indices = indices[:int(len(trainset)*args.init)]
        unlabeled_indices = [x for x in indices if x not in labeled_indices]
        args.queries = int(len(trainset)*args.queries)

        trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
                                 sampler=SubsetRandomSampler(labeled_indices))
        queryloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
                                 sampler=SequentialSampler(unlabeled_indices))
        testloader = DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False)

    #######################################################################################
    ############################# Models ##################################################
    #######################################################################################
    if args.method not in methods:
        print('You need to select an existing method via the --method argument. Available methods include: ')
        for key in methods:
            print('\t{0}'.format(key))
        raise Exception('You need to select a method')
    
    if args.model not in models:
        print('You need to select an existing model via the --model argument. Available models include: ')
        for key in models:
            print('\t{0}'.format(key))
        raise Exception('You need to select a model')

    if args.method == 'mdnal':
        if args.model == 'mdgcn':
            model = MDGCNNet(trainset, args).to(args.device)
        elif args.model == 'mdgat':
            model = MDGATNet(trainset, args).to(args.device)
        elif args.model == 'mdgin':
            model = MDGINNet(trainset, args).to(args.device)
        else:
            raise Exception('Using {} must usd mdgcn/mdgat/mdgin!'.format(args.method))
    else:
        if args.model == 'gcn':
            model = GCNNet(trainset, args).to(args.device)
        elif args.model == 'gat':
            model = GATNet(trainset, args).to(args.device)
        elif args.model == 'gin':
            model = GINNet(trainset, args).to(args.device)
        else:
            raise Exception('Using {} must usd gcn/gat/gin!'.format(args.method))

    print('Using {} with {} network...'.format(args.method, args.model))

    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer.')

    for round in range(1, args.rounds+1):
        print("="*60)
        print('Labeled samples: {}'.format(len(trainloader.sampler)))
        print('Unlabeled samples: {}'.format(len(queryloader.sampler)))
        for epoch in range(1, args.epochs+1):
            print("="*60)
            print("Round:{} Epoch:{}".format(round, epoch))

            t0 = time.time()
            epoch_loss = []
            for i_batch, sample_batched in enumerate(trainloader):
                batch_loss = train(args, model, sample_batched, optimizer)
                epoch_loss.append(batch_loss)
            t1 = time.time()
            writer.add_scalar('training loss', sum(epoch_loss), round*epoch)

            all_preds, all_gts, all_logits  = [], [], []
            t2 = time.time()
            for i_batch, sample_batched in enumerate(testloader):
                preds, gts, logits = evaluate(args, model, sample_batched)
                all_preds.extend(preds.numpy())
                all_gts.extend(gts.numpy())
                all_logits.append(logits.numpy())
            t3 = time.time()

            cm, acc, sen, spe, pre, f1, auroc, auprc = calculate_metrics(args, all_preds, all_gts, all_logits)
            fig_cm = plot_confusion_matrix(cm)
            writer.add_figure('confusion matrix', fig_cm, round*epoch)
            writer.add_scalar('accuracy', acc, round*epoch)
            writer.add_scalar('sensitivity(recall)', sen, round*epoch)
            writer.add_scalar('specificity', spe, round*epoch)
            writer.add_scalar('precision', pre, round*epoch)
            writer.add_scalar('f1_scores', f1, round*epoch)
            writer.add_scalar('auroc_score', auroc, round*epoch)
            writer.add_scalar('auprc_score', auprc, round*epoch)

            print('Training time:{:.3f} seconds, testing time:{:.3f} seconds.\
                            \n{}\
                            \naverage accuracy: \t\t{:.5f}\
                            \naverage sensitivity(recall): \t{:.5f}\
                            \naverage specificity: \t\t{:.5f}\
                            \naverage precision: \t\t{:.5f}\
                            \naverage f1_scores: \t\t{:.5f}\
                            \naverage auroc_score: \t\t{:.5f}\
                            \naverage auprc_score: \t\t{:.5f}'.format(t1-t0, t3-t2, cm, acc, sen, spe, pre, f1, auroc, auprc))
        
        # save models
        # save_path = './save/' + str(args.data) + '/' + str(args.method) + '/' + str(args.model) + '/' + str(round)
        # save_subfolder = os.path.join(save_path, args.save)
        # if not os.path.exists(save_subfolder): os.makedirs(save_subfolder)

        if round == (args.rounds):
            print("="*60)
            print("Training Finished.")
            writer.close()
            break

        t4 = time.time()
        query_indices = query(args, model, trainloader, queryloader, labeled_indices, unlabeled_indices)
        t5 = time.time()
        
        labeled_indices = labeled_indices + query_indices
        unlabeled_indices = [x for x in indices if x not in labeled_indices]

        trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
                                sampler=SubsetRandomSampler(labeled_indices))
        queryloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
                                sampler=SequentialSampler(unlabeled_indices))
        
        print("="*60)
        print('Querying time {:.3f} seconds.'.format(t5-t4))


if __name__ == '__main__':
    print("Start Runing!")
    main()
