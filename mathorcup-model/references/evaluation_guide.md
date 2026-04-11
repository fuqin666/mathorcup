# 评价/聚类/分类问题建模指南

> 适用问题类型：综合评价、聚类分析、分类预测
> 本指南包含：评价体系构建 → 权重确定 → 综合评价模型 → 聚类算法 → 分类模型

---

## 一、综合评价类问题

### 1.1 评价指标体系构建

**指标选取原则：**
- **完备性**：覆盖评价对象各个维度
- **独立性**：指标之间无重复/高度相关
- **可操作性**：指标可量化、数据可获取
- **敏感性**：能区分不同评价对象

**指标筛选流程：**
```
Step 1: 初选指标（尽可能全）
Step 2: 专家筛选（合并重复/删除无关）
Step 3: 相关性分析（剔除相关系数>0.9的指标）
Step 4: 主成分分析验证（确保主成分累计贡献>80%）
```

### 1.2 权重确定方法

| 方法 | 原理 | 适用场景 | 优缺点 |
|------|------|----------|--------|
| 层次分析法（AHP）| 专家两两比较，构造判断矩阵 | 有专家可用 | 主观性强，但有理论依据 |
| 熵权法 | 信息熵越小，权重越大 | 数据驱动 | 客观，但可能忽视指标重要性差异 |
| CRITIC法 | 综合考虑对比强度和冲突性 | 数据驱动 | 客观，考虑指标相关性 |
| 组合赋权 | AHP×0.4 + 熵权×0.6 | 综合 | 兼顾主客观 |

**AHP实现代码：**
```python
import numpy as np

def ahp pairwise_comparison matrix(judge_matrix):
    """
    judge_matrix: 专家两两比较矩阵（ Saaty 1-9 标度）
    返回: 各指标权重
    """
    # 归一化（按列）
    col_sum = judge_matrix.sum(axis=0)
    norm_matrix = judge_matrix / col_sum
    
    # 按行求均值即为权重
    weights = norm_matrix.mean(axis=1)
    
    # 一致性检验
    n = len(judge_matrix)
    lambda_max = np.mean(
        [sum(judge_matrix[i] * weights) / weights[i] for i in range(n)]
    )
    CI = (lambda_max - n) / (n - 1)
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    CR = CI / RI.get(n, 1.45)
    
    if CR < 0.1:
        print(f'一致性检验通过: CR={CR:.4f}')
    else:
        print(f'一致性检验未通过: CR={CR:.4f}，建议重新打分')
    
    return weights

# 示例：4个指标两两比较
judge_matrix = np.array([
    [1, 3, 5, 7],
    [1/3, 1, 3, 5],
    [1/5, 1/3, 1, 3],
    [1/7, 1/5, 1/3, 1]
])
weights = ahp_pairwise_comparison_matrix(judge_matrix)
print(f'权重: {weights}')
```

**熵权法实现代码：**
```python
def entropy_weight(data):
    """
    data: 标准化后的指标矩阵 (n_samples, n_indicators)
    返回: 各指标权重
    """
    # 归一化（每个指标归一化到[0,1]）
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
    
    # 计算比重
    n, m = data_norm.shape
    pij = data_norm / (data_norm.sum(axis=0) + 1e-10)
    
    # 计算熵值
    ej = -np.sum(pij * np.log(pij + 1e-10), axis=0) / np.log(n)
    
    # 计算权重
    gj = 1 - ej
    weights = gj / gj.sum()
    
    return weights

# 使用
data = pd.read_csv('evaluation_data.csv')
weights = entropy_weight(data.values)
print(f'权重: {weights}')
```

### 1.3 综合评价模型

| 模型 | 适用场景 | 实现难度 |
|------|----------|----------|
| TOPSIS | 有限方案多指标决策 | 低 |
| 熵权TOPSIS | 权重未知 | 低 |
| 灰色关联分析 | 信息不完整 | 中 |
| 模糊综合评价 | 定性指标多 | 中 |
| 主成分分析 | 指标过多 | 低 |
| 秩和比法（RSR）| 综合实力评价 | 中 |

**TOPSIS实现代码：**
```python
def topsis(data, weights, benefit_idx):
    """
    data: 标准化后的决策矩阵
    weights: 指标权重
    benefit_idx: 收益型指标索引（越大越好）
    """
    # 加权决策矩阵
    weighted = data * weights
    
    # 最优/最劣方案
    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)
    ideal_best[~benefit_idx] = np.min(weighted[:, ~benefit_idx], axis=0)
    ideal_worst[benefit_idx] = np.max(weighted[:, benefit_idx], axis=0)
    
    # 距离
    dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    
    # 相对贴近度
    closeness = dist_worst / (dist_best + dist_worst)
    
    return closeness

# 使用
result = topsis(data_norm, weights, benefit_idx)
rank = np.argsort(result)[::-1] + 1  # 排名（1=最优）
```

---

## 二、聚类问题

### 2.1 算法选择

| 算法 | 适用场景 | 特点 | 参数 |
|------|----------|------|------|
| K-means | 球形簇、大规模数据 | 快速，需预设K | n_clusters |
| DBSCAN | 任意形状、有噪声 | 无需预设K | eps, min_samples |
| 层次聚类 | 需层次结构 | 可解释 | n_clusters 或 distance_threshold |
| 谱聚类 | 非凸簇 | 效果好 | n_clusters |
| GMM | 软聚类（概率）| 可得概率归属 | n_components |

### 2.2 K-means完整实现

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np

def kmeans_analysis(data, k_range=range(2, 11)):
    """
    K-means聚类分析
    包括：K选择、聚类、评估
    """
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 评估指标
        silhouette = silhouette_score(X, labels)  # 轮廓系数（-1到1）
        ch = calinski_harabasz_score(X, labels)  # CH指数（越大越好）
        inertia = kmeans.inertia_  # SSE（越小越好，肘部法则）
        
        results.append({
            'k': k, 'silhouette': silhouette, 
            'ch_index': ch, 'inertia': inertia
        })
    
    df = pd.DataFrame(results)
    
    # 绘制肘部法则图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df['k'], df['inertia'], 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['k'], df['silhouette'], 'rs-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/kmeans_evaluation.png', dpi=150)
    plt.close()
    
    # 最优K
    best_k = df.loc[df['silhouette'].idxmax(), 'k']
    print(f'最优K（轮廓系数）: {best_k}')
    
    return df, best_k

# 聚类结果分析
def analyze_clusters(data, labels, feature_names):
    """分析各簇特征"""
    df = pd.DataFrame(data, columns=feature_names)
    df['Cluster'] = labels
    
    # 各簇均值
    cluster_means = df.groupby('Cluster').mean()
    print('各簇特征均值:')
    print(cluster_means.round(2))
    
    # 各簇样本数
    print(f'\n各簇样本数:')
    print(df['Cluster'].value_counts().sort_index())
    
    # 各簇差异最大的特征
    overall_mean = df[feature_names].mean()
    cluster_profile = cluster_means - overall_mean
    print('\n各簇相对优势（相比均值）:')
    for c in cluster_means.index:
        top_feature = cluster_profile.loc[c].abs().idxmax()
        direction = '+' if cluster_profile.loc[c, top_feature] > 0 else '-'
        print(f'  Cluster {c}: {direction}{top_feature}')
    
    return cluster_means
```

---

## 三、分类问题

### 3.1 算法选择

| 算法 | 适用场景 | 特点 |
|------|----------|------|
| 逻辑回归 | 二分类、可解释 | 可得概率，简单快速 |
| 决策树 | 规则提取 | 可解释，易过拟合 |
| 随机森林 | 通用、鲁棒 | 不易过拟合 |
| XGBoost/LightGBM | 竞赛首选 | 高精度，速度快 |
| SVM | 小样本高维 | 核函数选择关键 |
| 神经网络 | 复杂模式 | 需大量数据 |

### 3.2 分类评估指标详解

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

def classification_eval(y_true, y_pred, y_prob=None):
    """完整分类评估"""
    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-score': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None:
        results['AUC-ROC'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    print(pd.DataFrame([results]).T)
    print('\n混淆矩阵:')
    print(confusion_matrix(y_true, y_pred))
    print('\n完整报告:')
    print(classification_report(y_true, y_pred))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=150)
    plt.close()
    
    return results
```

---

## 五、真实论文指标统计（41篇优秀论文深度扫描）

> 数据来源：MathorCup 2020-2024 共 41 篇优秀论文前15页深度分析。
> 原始数据：`paper_analysis/deep_scan/deep_scan_summary.json`

### 5.1 高频评估指标（Top 15）

| 排名 | 指标 | 频率 | 说明 |
|------|------|------|------|
| 🥇 | **RMSE** | 14/41 (34%) | 预测类首选，有量纲，对异常值敏感 |
| 🥈 | **相关系数** | 13/41 (32%) | 皮尔逊/斯皮尔曼相关分析 |
| 🥉 | **MSE** | 9/41 (22%) | 均方误差 |
| 4 | **MAE** | 6/41 (15%) | 平均绝对误差 |
| 5 | DTW | 4/41 (10%) | 动态时间规整，路径相似度 |
| 6 | AUC | 4/41 (10%) | ROC曲线下面积，分类问题 |
| 7 | 召回率 | 4/41 (10%) | 分类召回 |
| 8 | R²/决定系数 | 3/41 (7%) | 模型解释力 |
| 9 | 皮尔逊/斯皮尔曼 | 3/41 (7%) | 相关性 |
| 10 | F1 | 3/41 (7%) | 精确率-召回率调和 |
| 11 | MAPE | 3/41 (7%) | 百分比误差 |
| 12 | ROC | 3/41 (7%) | 分类曲线 |

**历年最常见组合：**
- 回归预测：RMSE + MAE + R² 三件套
- 分类问题：Accuracy + F1 + AUC 三件套
- 路径/序列相似度：DTW + MAE + RMSE 三件套

### 5.2 预处理方法频率

| 方法 | 频率 | 说明 |
|------|------|------|
| **特征提取** | 17/41 | 最常见，约41%论文涉及 |
| 数据增强 | 7/41 | 深度学习类 |
| **特征选择** | 6/41 | 降维、剔除冗余 |
| 指数平滑 | 3/41 | 传统时序 |
| 滑动窗口 | 3/41 | 时序构造特征 |
| 移动平均 | 3/41 | 平滑处理 |
| Z-score | 2/41 | 标准化 |
| 缺失值处理 | 2/41 | 含KNN、均值等 |
| 异常值处理 | 2/41 | 含剔除、正态转换 |

### 5.3 典型数值参考（来自真实论文）

| 场景 | RMSE | MAE | AUC | MAPE |
|------|------|-----|-----|------|
| 货量/销量预测 | 0.08~2.14 | 1.0~1.5 | — | <15% |
| 路径相似度(DTW) | 越<1越好 | — | — | — |
| 分类(AUC) | — | — | 0.68~0.98 | — |

### 5.4 灵敏度/鲁棒性分析（重要！）

> **68%的优秀论文（28/41篇）包含灵敏度或鲁棒性分析。**
> 这是 MathorCup 的高频加分项，评委必看内容之一。

| 分析类型 | 典型做法 |
|---------|---------|
| 参数灵敏度 | 改变某个参数，观察指标变化（如改变学习率、窗口大小） |
| 数据扰动 | 对输入数据加噪声，测试模型稳定性 |
| 鲁棒性检验 | 对抗样本攻击（图像类）、异常值干扰测试 |
| 误差分析 | 分析预测误差分布，找出系统偏差 |

**必须包含的内容（历年优秀论文标配）：**
1. ✅ 至少1种参数灵敏度分析
2. ✅ 误差分布图/残差图
3. ✅ 对比表格（消融实验）

---

## 六、多模型融合策略

### 4.1 融合方法对比

| 方法 | 原理 | 适用场景 | 实现难度 |
|------|------|----------|----------|
| 简单平均 | 多个模型预测取平均 | 模型效果接近 | 极低 |
| 加权平均 | 按效果加权 | 效果差异明显 | 低 |
| Stacking | 用元学习器组合 | 效果差异大 | 高 |
| Blending | 类似Stacking，更简单 | 数据量不大 | 中 |
| 模型选择 | 每个样本选最佳模型 | 模型差异明显 | 中 |

### 4.2 Stacking完整实现

```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
import numpy as np

def stacking_ensemble(X_train, y_train, X_test, base_models, meta_model=None):
    """
    Stacking集成
    base_models: 基学习器列表
    meta_model: 元学习器（默认Ridge）
    """
    if meta_model is None:
        meta_model = Ridge(alpha=1.0)
    
    # 用交叉验证生成基学习器OOF预测
    oof_train = np.zeros((X_train.shape[0], len(base_models)))
    oof_test = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models.items()):
        print(f'Training {name}...')
        # OOF预测
        oof_train[:, i] = cross_val_predict(
            model, X_train, y_train, cv=5, method='predict'
        )
        # 测试集预测（用全量训练）
        model.fit(X_train, y_train)
        oof_test[:, i] = model.predict(X_test)
    
    # 评估各基学习器
    for i, name in enumerate(base_models.keys()):
        mae = mean_absolute_error(y_train, oof_train[:, i])
        print(f'{name} OOF MAE: {mae:.4f}')
    
    # 元学习器
    meta_model.fit(oof_train, y_train)
    final_pred = meta_model.predict(oof_test)
    
    print(f'\nMeta model weights: {meta_model.coef_}')
    print(f'Stacking OOF MAE: {mean_absolute_error(y_train, oof_train @ meta_model.coef_):.4f}')
    
    return final_pred, oof_train, oof_test
```

---

*本指南参考 param_tuning.md（调参手册）*
