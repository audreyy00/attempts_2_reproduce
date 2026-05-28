"""评估 KnockoffGAN 输出的 TPR 和 FDR"""
import numpy as np
import pandas as pd
import json
import argparse
from sklearn.linear_model import LassoCV


def knockoff_threshold(W, fdr=0.1):
    """论文 Theorem 1 的 knockoff 阈值选择"""
    W_sorted = np.sort(np.abs(W[W < 0]))  # 只用负值确定阈值
    for t in W_sorted:
        num_false = 1 + np.sum(W <= -t)
        num_true = max(np.sum(W >= t), 1)
        if num_false / num_true <= fdr:
            return t
    # 如果所有负值都不够，返回一个很大的阈值（不选任何特征）
    return np.max(np.abs(W)) + 1.0


def lasso_coef_diff(X, X_k, y, y_type='logit'):
    """计算 LASSO Coefficient Difference 统计量"""
    X_aug = np.hstack([X, X_k])
    p = X.shape[1]

    if y_type == 'logit':
        # 用线性 LASSO 近似（论文在合成实验中用 LASSO，即使 Y 是 binary）
        # 也加了 family="binomial" 的 glmnet 版本
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(Cs=20, penalty='l1', solver='saga',
                                      max_iter=5000, cv=3, n_jobs=-1)
        model.fit(X_aug, y)
        coef = model.coef_.flatten()
    else:
        model = LassoCV(cv=3, max_iter=5000, n_jobs=-1)
        model.fit(X_aug, y)
        coef = model.coef_.flatten()
        if len(coef) < 2 * p:
            coef = np.pad(coef, (0, 2 * p - len(coef)))

    W = np.abs(coef[:p]) - np.abs(coef[p:])
    return W


def evaluate(X, X_k, y, nonzero, y_type='logit', fdr=0.1):
    """计算 TPR 和 FDR"""
    p = X.shape[1]
    W = lasso_coef_diff(X, X_k, y, y_type)
    t = knockoff_threshold(W, fdr)
    selected = np.where(W >= t)[0]

    null = np.setdiff1d(np.arange(p), nonzero)
    n_selected = len(selected)
    n_false = len(np.intersect1d(selected, null))
    n_true = len(np.intersect1d(selected, nonzero))

    tpr = n_true / len(nonzero) if len(nonzero) > 0 else 0
    fdr_val = n_false / n_selected if n_selected > 0 else 0

    return {
        'tpr': tpr,
        'fdr': fdr_val,
        'n_selected': n_selected,
        'n_true': n_true,
        'n_false': n_false,
        'selected': selected.tolist(),
        'W': W.tolist(),
        'threshold': float(t),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='原始数据 CSV')
    parser.add_argument('--knockoff', required=True, help='Knockoff CSV')
    parser.add_argument('--truth', required=True, help='Ground truth JSON')
    parser.add_argument('--target', default='label', help='标签列名')
    parser.add_argument('--fdr', default=0.1, type=float, help='FDR 阈值')
    parser.add_argument('--ytype', default='logit', choices=['logit', 'gauss'])
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df_k = pd.read_csv(args.knockoff)

    with open(args.truth) as f:
        truth = json.load(f)

    features = [c for c in df.columns if c != args.target]
    X = df[features].values
    X_k = df_k[features].values
    y = df[args.target].values
    nonzero = np.array(truth['nonzero'])

    result = evaluate(X, X_k, y, nonzero, args.ytype, args.fdr)

    print(json.dumps(result, indent=2))
    print(f"\nTPR: {result['tpr']:.4f}")
    print(f"FDR: {result['fdr']:.4f} (target: {args.fdr})")
    print(f"Selected: {result['n_selected']} / {X.shape[1]} features")
    print(f"True positives: {result['n_true']} / {len(nonzero)}")


if __name__ == '__main__':
    main()
