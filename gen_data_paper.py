"""生成论文实验数据：4 种分布 × 2 种标签类型"""
import numpy as np
import pandas as pd
import argparse
import json
import os


def generate_y(X, nonzero, beta_strength, y_type, seed):
    """生成标签 Y，返回 (y, beta)"""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    k = len(nonzero)

    # 随机符号，每个相关特征的 beta 为 ±beta_strength / sqrt(n)
    signs = rng.choice([-1, 1], k)
    beta = np.zeros(p)
    beta[nonzero] = signs * beta_strength / np.sqrt(n)

    logit = X @ beta
    if y_type == 'logit':
        prob = 1 / (1 + np.exp(-logit))
        y = rng.binomial(1, prob)
    elif y_type == 'gauss':
        y = logit + rng.normal(0, 1, n)
    return y, beta, signs


def gaussian_ar1(n, p, k, rho, beta_strength, y_type, seed=0):
    """AR(1) 过程 + 高斯边缘分布"""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, p))
    X[:, 0] = rng.normal(0, np.sqrt(1.0 / n), n)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + rng.normal(0, np.sqrt((1 - rho**2) / n), n)

    nonzero_idx = rng.choice(p, k, replace=False)
    y, beta, signs = generate_y(X, nonzero_idx, beta_strength, y_type, seed + 1)
    return X, y, nonzero_idx, beta, signs


def gaussian_iid(n, p, k, beta_strength, y_type, seed=0):
    """独立同分布高斯"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, np.sqrt(1.0 / n), (n, p))

    nonzero_idx = rng.choice(p, k, replace=False)
    y, beta, signs = generate_y(X, nonzero_idx, beta_strength, y_type, seed + 1)
    return X, y, nonzero_idx, beta, signs


def gmm4(n, p, k, beta_strength, y_type, seed=0):
    """4-高斯混合模型（论文 §5.1.3）"""
    rng = np.random.default_rng(seed)

    # 4 个 cluster 的均值向量
    m1 = np.zeros(p)
    m1[:100] = 1.0

    m2 = np.zeros(p)
    m2[:50] = 1.0
    m2[50:100] = -1.0

    m3 = np.zeros(p)
    m3[:50] = -1.0
    m3[50:100] = 1.0

    m4 = np.zeros(p)
    m4[:100] = -1.0

    # 每个样本等概率来自 4 个 cluster
    comp = rng.integers(0, 4, n)
    means = np.array([m1, m2, m3, m4])

    X = np.zeros((n, p))
    for i in range(n):
        X[i] = means[comp[i]] + rng.normal(0, np.sqrt(1.0), p)

    # 归一化方差为 1/n
    feature_std = X.std(axis=0, ddof=1)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X = X / feature_std * np.sqrt(1.0 / n)

    nonzero_idx = rng.choice(p, k, replace=False)
    y, beta, signs = generate_y(X, nonzero_idx, beta_strength, y_type, seed + 1)
    return X, y, nonzero_idx, beta, signs


def uniform_ar1(n, p, k, rho, beta_strength, y_type, seed=0):
    """AR(1) 过程 + 均匀边缘分布 U(-sqrt(3/n), sqrt(3/n))"""
    rng = np.random.default_rng(seed)
    bound = np.sqrt(3.0 / n)

    X = np.zeros((n, p))
    X[:, 0] = rng.uniform(-bound, bound, n)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + rng.uniform(-bound, bound, n)

    nonzero_idx = rng.choice(p, k, replace=False)
    y, beta, signs = generate_y(X, nonzero_idx, beta_strength, y_type, seed + 1)
    return X, y, nonzero_idx, beta, signs


DISTRIBUTIONS = {
    'gaussian_ar1': gaussian_ar1,
    'gaussian_iid': gaussian_iid,
    'gmm4': gmm4,
    'uniform_ar1': uniform_ar1,
}


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for KnockoffGAN experiments')
    parser.add_argument('--dist', default='gaussian_ar1',
                        choices=list(DISTRIBUTIONS.keys()),
                        help='Feature distribution')
    parser.add_argument('--n', default=3000, type=int, help='Number of samples')
    parser.add_argument('--p', default=1000, type=int, help='Number of features')
    parser.add_argument('--k', default=60, type=int, help='Number of relevant features')
    parser.add_argument('--rho', default=0.3, type=float, help='AR correlation (for AR distributions)')
    parser.add_argument('--beta', default=5.0, type=float, help='Signal strength')
    parser.add_argument('--ytype', default='logit', choices=['logit', 'gauss'],
                        help='Y type: logit (classification) or gauss (regression)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--odir', default='.', help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    gen_fn = DISTRIBUTIONS[args.dist]

    kwargs = dict(n=args.n, p=args.p, k=args.k,
                  beta_strength=args.beta, y_type=args.ytype,
                  seed=args.seed)
    if 'ar1' in args.dist:
        kwargs['rho'] = args.rho

    X, y, nonzero_idx, beta, signs = gen_fn(**kwargs)

    # 保存数据 CSV
    cols = [f'V{j}' for j in range(args.p)]
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y

    tag = f"{args.dist}_n{args.n}_p{args.p}_k{args.k}_beta{args.beta}_seed{args.seed}"
    if 'ar1' in args.dist:
        tag += f"_rho{args.rho}"
    tag += f"_{args.ytype}"

    csv_path = os.path.join(args.odir, f'data_{tag}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Data saved: {csv_path}')

    # 保存 ground truth
    truth = {
        'nonzero': nonzero_idx.tolist(),
        'beta': beta.tolist(),
        'signs': signs.tolist(),
        'distribution': args.dist,
        'n': args.n, 'p': args.p, 'k': args.k,
        'beta_strength': args.beta,
        'y_type': args.ytype,
        'seed': args.seed,
    }
    if 'ar1' in args.dist:
        truth['rho'] = args.rho

    json_path = os.path.join(args.odir, f'ground_truth_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(truth, f, indent=2)
    print(f'Ground truth saved: {json_path}')
    print(f'Relevant variables ({len(nonzero_idx)}): {nonzero_idx[:10]}...')


if __name__ == '__main__':
    main()
