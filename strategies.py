import random
import numpy as np
import torch
import torch.nn.functional as F

from kcenterGreedy import kCenterGreedy


def get_max_entropy(args, model, unlabeled_loader, unlabeled_indices):
    device = args.device
    model.eval()

    uncertainty = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in unlabeled_loader:
            inputs = data.to(device)

            _, embed = model(inputs)
            embed = embed.clamp(min=1e-8)
            entropy = -(embed * torch.log(embed)).sum(dim=1)

            uncertainty = torch.cat((uncertainty, entropy), 0)

    uncertainty = uncertainty.cpu().numpy()

    topk_positions = uncertainty.argsort()[::-1][:args.queries].tolist()
    topk_indices = [unlabeled_indices[i] for i in topk_positions]

    return topk_indices


def get_kcg(args, model, labeled_loader, unlabeled_loader, labeled_indices):
    device = args.device
    model.eval()

    features = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in labeled_loader:
            inputs = data.to(device)
            _, embed = model(inputs)
            features = torch.cat((features, embed), 0)

        for data in unlabeled_loader:
            inputs = data.to(device)
            _, embed = model(inputs)
            features = torch.cat((features, embed), 0)

        features = features.cpu().numpy()

        sampling = kCenterGreedy(features)
        topk_indices = sampling.select_batch_(labeled_indices, args.queries)
        
    return topk_indices


def get_uncertainty_diversity(args, model, labeled_loader, unlabeled_loader, labeled_indices, unlabeled_indices):
    device = args.device
    model.eval()
    assert int(args.ratio * args.queries) <= len(unlabeled_loader.sampler)

    # Stage 1
    unlabel_samples = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in unlabeled_loader:
            samples = mdn_samping(args, model, data, num_samples=args.ns)
            unlabel_samples = torch.cat((unlabel_samples, samples), 0)

    epistemic, aleatoric = compute_uncertainty(unlabel_samples)
    epistemic, aleatoric = normalize(epistemic), normalize(aleatoric)
    uncertainty = args.lambd*epistemic + (1-args.lambd)*aleatoric

    uncertainty = uncertainty.cpu().numpy()
    uncer_indices = uncertainty.argsort()[::-1][:int(args.ratio * args.queries)].tolist()

    # Stage 2
    dist_x = torch.mean(unlabel_samples, dim=1)[uncer_indices]
    dist_y = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in labeled_loader:
            y = F.one_hot(data.y.view(-1)).to(device)
            dist_y = torch.cat((dist_y, y), 0)

    diversity = torch.sum(js_divergence(dist_x, dist_y), dim=-1)

    diversity = diversity.cpu().numpy()
    diver_indices = diversity.argsort()[::-1][:int(args.queries)].tolist()

    # 索引映射: diver_indices -> uncer_indices -> unlabeled_indices
    topk_indices = []
    for d in diver_indices:
        idx_in_unlabeled = uncer_indices[d]
        total_idx = unlabeled_indices[idx_in_unlabeled]
        topk_indices.append(total_idx)

    return topk_indices

def mdn_samping(args, model, data, num_samples=1):
    device = args.device
    model.eval()
    with torch.no_grad():
        data = data.to(device)

        logits_pi, logits_mu = model(data)
        # 1. 混合权重
        pi = torch.softmax(logits_pi, dim=-1)  # [B, K]
        # 2. 类别分布参数
        mu = torch.softmax(logits_mu, dim=-1)  # [B, K, C]

        samples = torch.zeros((mu.size(0), num_samples, mu.size(-1)), dtype=torch.float).to(device)
        for i in range(pi.size(0)):
            # 3. 选择混合成分
            comp_samples = torch.multinomial(pi[i], num_samples, replacement=True)
            for j in range(num_samples):
                comp_idx = comp_samples[j].item()
                # 4. 从对应成分
                # class_sample = torch.multinomial(mu[i, comp_idx], 1)
                # samples[i, j] = class_sample.item()
                samples[i, j] = mu[i, comp_idx]

    return samples

def compute_uncertainty(all_samples, eps = 1e-8):
    # 计算每次采样的熵 [N, K]
    entropy_per_draw = -torch.sum(all_samples * torch.log(all_samples + eps), dim=-1)
    # 计算偶然不确定性 = 平均熵 [N]
    aleatoric = torch.mean(entropy_per_draw, dim=-1)
    # 计算平均预测概率 [N, C]
    mean_probs = torch.mean(all_samples, dim=1)
    # 计算平均预测的熵 (总不确定性) [N]
    entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
    # 计算认知不确定性 = 总不确定性 - 偶然不确定性 [N]
    epistemic = entropy_mean - aleatoric
    # 确保非负（数值计算可能导致微小负数）
    epistemic = torch.clamp(epistemic, min=0.0)
    return epistemic, aleatoric

# 归一化函数（避免除零）
def normalize(tensor):
    t_min, t_max = tensor.min(), tensor.max()
    # 防止所有值相同导致除零
    if torch.isclose(t_min, t_max):
        return torch.zeros_like(tensor)
    return (tensor - t_min) / (t_max - t_min + 1e-8)

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    # Step 1: 归一化输入为概率分布
    p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)
    q_norm = q / (q.sum(dim=-1, keepdim=True) + eps)
    
    # 扩展维度用于广播计算
    p_exp = p_norm.unsqueeze(1)  # [N1, 1, C]
    q_exp = q_norm.unsqueeze(0)  # [1, N2, C]
    
    # Step 2: 计算平均分布 M
    m = 0.5 * (p_exp + q_exp)  # [N1, N2, C]
    
    # Step 3: 计算 KL 散度 (使用 log(eps) 确保数值稳定)
    kl_p = p_exp * (torch.log(p_exp.clamp(min=eps)) - torch.log(m.clamp(min=eps)))
    kl_p = kl_p.sum(dim=-1)  # [N1, N2]
    
    kl_q = q_exp * (torch.log(q_exp.clamp(min=eps)) - torch.log(m.clamp(min=eps)))
    kl_q = kl_q.sum(dim=-1)  # [N1, N2]
    
    # Step 4: 计算 JS 散度
    js = 0.5 * (kl_p + kl_q)  # [N1, N2]
    
    return js


def query(args, model, trainloader, queryloader, labeled_indices, unlabeled_indices):
    method = args.method
    queries = args.queries
    
    assert queries <= len(unlabeled_indices)
    if method == 'random':
        query_indices = random.sample(unlabeled_indices, queries)
    elif method == 'entropy':
        query_indices = get_max_entropy(args, model, queryloader, unlabeled_indices)
    elif method == 'coreset':
        query_indices = get_kcg(args, model, trainloader, queryloader, labeled_indices)
    elif method == 'mdnal':
        query_indices = get_uncertainty_diversity(args, model, trainloader, queryloader, labeled_indices, unlabeled_indices)
    else:
        raise ValueError('AL method {} is not support!'.format(method))
    
    return query_indices