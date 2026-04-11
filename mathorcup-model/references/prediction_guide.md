# 预测类问题建模指南

> 适用问题类型：时间序列预测、数值回归预测、分类预测
> 本指南包含：问题识别 → 数据分析 → 模型选型 → 代码模板 → 评估体系

---

## 一、问题识别

### 1.1 关键词判定

| 关键词 | 问题类型 | 常见赛题 |
|--------|----------|----------|
| 预测、估计、推断 | 时序/回归预测 | 货量预测、销量预测 |
| 货量/流量/需求/订单 | 回归预测 | 物流/运输预测 |
| 分类、识别、判别 | 分类预测 | 图像/文本分类 |
| 估计、推断、推算 | 回归预测 | 成本/收益估计 |

### 1.2 典型赛题模式

```
问题一：给出历史数据，预测未来某时间段的数值
问题二：基于预测结果进行优化决策
问题三/四：模型改进/鲁棒性分析
```

---

## 二、数据特征分析

### 2.1 必做分析清单

| 分析维度 | 具体内容 | 判断依据 |
|----------|----------|----------|
| 时间粒度 | 分钟/小时/天/周/月 | 决定特征工程方向 |
| 时间跨度 | 数据覆盖多久 | 决定历史特征选取范围 |
| 季节性 | 是否有周期性波动 | 决定是否需要周期特征 |
| 趋势性 | 是否有明显上升/下降趋势 | 决定是否需要差分 |
| 异常点 | 是否有异常高/低点 | 决定异常处理策略 |
| 缺失情况 | 缺失比例、缺失模式 | 决定填充方法 |
| 预知信息 | 是否有提前知道的信息 | 决定如何使用预知数据 |
| 空间关联 | 多节点/多区域数据 | 决定是否用图神经网络 |

### 2.2 探查代码模板

```python
import pandas as pd
import numpy as np
from config import Config as C, DATA, OUT, FIG, MODEL  # 统一路径配置
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载
df = pd.read_csv(DATA('data.csv'))
print(f"数据规模: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值:\n{df.isnull().sum()}")
print(f"\n统计描述:\n{df.describe()}")

# 缺失值分析
missing_pct = df.isnull().sum() / len(df) * 100
print(f"\n缺失比例:\n{missing_pct}")

# 时间特征提取（如有时序列）
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

# 可视化：时间序列趋势
df.set_index('date').plot(figsize=(14, 5))
plt.title('时间序列趋势')
plt.tight_layout()
plt.savefig(FIG('time_series.png'), dpi=C.figure_dpi)

# 相关性分析
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('特征相关性矩阵')
plt.tight_layout()
plt.savefig(FIG('correlation.png'), dpi=C.figure_dpi)

# 周期性分析（以天为单位重采样）
daily = df.set_index('date').resample('D').mean()
weekly_pattern = daily.groupby(daily.index.dayofweek).mean()  # 星期模式
hourly_pattern = df.groupby('hour').mean()  # 小时模式
```

---

## 三、模型选型决策

### 3.1 根据数据特征选模型

| 数据特征 | 推荐模型 | 备选模型 |
|----------|----------|----------|
| 数据量小（<1000）| 线性回归、ARIMA | XGBoost、随机森林 |
| 数据量中等（1000-10万）| XGBoost、LightGBM | 随机森林、LSTM |
| 数据量大（>10万）| LightGBM、CatBoost | 深度学习模型 |
| 强时间依赖 | LSTM、Transformer | ARIMA + 特征工程 |
| 强周期依赖 | Prophet、Transformer | 周期分解 + 预测 |
| 有预知信息 | 两阶段模型 | 特征构造法 |
| 类别特征丰富 | CatBoost | LightGBM、XGBoost |
| 多节点时空数据 | STGCN、ASTGCN | 多任务学习 |

### 3.2 多节点时空数据专用：图神经网络

**何时用：** 数据包含多个空间节点+时间维度，如：
- 交通流量预测（多路口/多路段）
- 物流网络预测（多仓库/多站点）
- 电力网络预测（多节点/多线路）

**核心思想：** 用图结构建模空间依赖，用卷积/注意力建模时间依赖

**STGCN（时空图卷积网络）：**
```python
class STGCN(torch.nn.Module):
    """
    时空图卷积网络
    - 图卷积层：建模空间依赖（基于邻接矩阵）
    - 时间卷积层：建模时间依赖（1D卷积）
    - 输出层：1x1卷积得到预测值
    """
    def __init__(self, adj_matrix, seq_len, pred_len, channels=[64,64,64,128]):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        K, K_t = 3, 3  # 图卷积核、时间卷积核大小
        # 图卷积层（空间）
        self.gc1 = GraphConv(K=K, channels=channels[0:2])
        self.gc2 = GraphConv(K=K, channels=channels[1:3])
        self.gc3 = GraphConv(K=K, channels=channels[2:4])
        # 时间卷积层
        self.tc1 = TemporalConv(K=K_t, channels=channels[0])
        self.tc2 = TemporalConv(K=K_t, channels=channels[1])
        self.tc3 = TemporalConv(K=K_t, channels=channels[2])
        self.out = torch.nn.Conv2d(channels[-1], pred_len, kernel_size=1)
    
    def forward(self, x):
        # x: (batch, seq_len, num_nodes)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, N, T, 1)
        # 图卷积 + 时间卷积 × 3
        x = self.gc1(x) + self.tc1(x)
        x = self.gc2(x) + self.tc2(x)
        x = self.gc3(x) + self.tc3(x)
        x = self.out(x).squeeze(-1).permute(0, 2, 1)  # (B, pred_len, N)
        return x
```

**ASTGCN（注意力时空图卷积）：**
- 额外加入**空间注意力**和**时间注意力**层
- 适合数据动态变化（不同时刻不同权重）
- 参数量更大，需要更多数据

**邻接矩阵构造方法：**

| 方法 | 适用场景 | 构造方式 |
|------|----------|----------|
| 距离矩阵 | 交通/电力网络 | 基于地理距离的倒数或高斯核 |
| 相关性矩阵 | 任意数据 | 基于节点历史数据相关性 + 阈值过滤 |
| 邻接关系 | 已知拓扑结构 | 直接使用拓扑图 |
| 可学习矩阵 | 数据量大 | 用参数学习最优图结构 |

```python
# 距离矩阵构造（高斯核）
def get_adj_matrix_distance(distances, sigma=0.1, threshold=0.5):
    adj = np.exp(-distances**2 / sigma**2)
    adj[adj < threshold] = 0  # 稀疏化
    return adj

# 相关性矩阵构造
def get_adj_matrix_correlation(data, threshold=0.3):
    corr = np.corrcoef(data.T)  # (num_nodes, num_nodes)
    adj = np.where(np.abs(corr) > threshold, corr, 0)
    return adj
```

---

## 四、预知数据处理（⭐ 2025 D题核心考点）

### 4.1 什么是预知数据

**预知数据 = 未来时间段内可以提前知道的信息。**

| 场景 | 预知数据示例 |
|------|-------------|
| 物流配送 | 已知未来订单量 |
| 交通预测 | 已知未来活动/事件（如节假日）|
| 电力预测 | 已知未来电价/天气 |
| 需求预测 | 已知未来促销活动 |

### 4.2 三种处理策略

**策略1：残差预测法（⭐ 推荐，评分最高）**
```python
"""
原理：先用基础模型预测整体趋势，再用预知数据预测残差
效果：能将MAPE从10%降到6%
"""
# Step 1: 用历史数据训练基础预测模型
base_model = LightGBM(n_estimators=500)
base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_train)

# Step 2: 计算残差
residuals = y_train - base_pred

# Step 3: 用预知数据预测残差
residual_model = Ridge(alpha=1.0)
residual_model.fit(X_known_train, residuals)  # X_known=预知特征
residual_pred = residual_model.predict(X_known_test)

# Step 4: 叠加
final_pred = base_model.predict(X_test) + residual_pred
```

**策略2：特征构造法（简单直接）**
```python
"""
原理：将预知数据作为模型输入特征
注意：预知特征只能用于预测时确实已知的时间段
"""
# 构造预知特征
df['is_holiday'] = df['date'].isin(holiday_list)
df['promotion_type'] = df['promotion_plan']  # 提前知道的促销类型
df['expected_volume'] = df['planned_orders']  # 已知计划订单量

# 用增强特征训练
X = df[['hour', 'dayofweek', 'is_holiday', 'expected_volume', ...]]
y = df['actual_volume']
model = LightGBM()
model.fit(X, y)
```

**策略3：两阶段模型**
```python
"""
原理：
- 阶段1：用无预知数据训练通用模型
- 阶段2：用有预知数据训练修正模型
- 最终 = 阶段1预测 + 阶段2修正
"""
# 同残差预测法思路，但两个模型可以独立调参
```

---

## 五、特征工程

### 5.1 时序预测特征清单

| 特征类型 | 示例 | 作用 |
|----------|------|------|
| 时间特征 | hour/month/dayofweek | 捕捉周期性 |
| 滞后特征 | y_{t-1}, y_{t-7}, y_{t-24} | 捕捉自相关性 |
| 滑动统计 | 7日均值/方差/最大值 | 捕捉趋势和波动 |
| 差分特征 | diff(y_t), y_t - y_{t-7} | 去除趋势/季节性 |
| 周期分解 | trend, seasonal, residual | 分解后可分别预测 |
| 外部特征 | 天气、温度、促销 | 捕捉外部影响 |
| 节点特征 | 节点ID编码/嵌入 | 图神经网络专用 |

### 5.2 特征工程代码模板

```python
import pandas as pd
import numpy as np

def create_time_features(df, date_col='date'):
    """时间特征构造"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['hour'] = df[date_col].dt.hour
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['quarter'] = df[date_col].dt.quarter
    # 周期性编码（sin/cos）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 24]):
    """滞后特征构造"""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows=[7, 14, 24]):
    """滑动统计特征"""
    df = df.copy()
    for w in windows:
        df[f'{target_col}_roll_mean{w}'] = df[target_col].rolling(w).mean()
        df[f'{target_col}_roll_std{w}'] = df[target_col].rolling(w).std()
        df[f'{target_col}_roll_max{w}'] = df[target_col].rolling(w).max()
        df[f'{target_col}_roll_min{w}'] = df[target_col].rolling(w).min()
    return df

def create_diff_features(df, target_col, periods=[1, 7]):
    """差分特征"""
    df = df.copy()
    for p in periods:
        df[f'{target_col}_diff{p}'] = df[target_col].diff(p)
    return df
```

---

## 六、多模型对比框架

### 6.1 必须对比的baseline模型

| 类别 | 模型 | 目的 |
|------|------|------|
| 朴素 | 历史均值（HA）、上期值 | 最低标准 |
| 统计 | ARIMA、指数平滑 | 经典方法 |
| 机器学习 | 随机森林、XGBoost、LightGBM | ML基准 |
| 深度学习 | LSTM、Transformer | DL基准 |
| 本文提出 | 完整方案 | 参赛方案 |

### 6.2 评估指标选择

| 指标 | 公式 | 适用场景 |
|------|------|----------|
| MAE | 1/n Σ\|y_i - ŷ_i\| | 常用，量纲一致 |
| RMSE | √(1/n Σ(y_i - ŷ_i)²) | 对大误差更敏感 |
| MAPE | 1/n Σ\|(y_i - ŷ_i)/y_i\| | 相对误差，百分比 |
| SMAPE | 2/n Σ\|y_i - ŷ_i\|/(\|y_i\|+\|ŷ_i\|) | 对称MAPE |
| R² | 1 - Σ(y_i-ŷ_i)²/Σ(y_i-ȳ)² | 模型解释度 |

**选择规则：** 评估指标必须与**题目要求一致**。

### 6.3 多模型对比代码

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def evaluate_model(y_true, y_pred, name):
    """评估单个模型"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': f'{mape:.2f}%'}

# 多模型评估
results = []
for name, model in models.items():
    pred = model.predict(X_test)
    results.append(evaluate_model(y_test, pred, name))

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('MAE')
print(comparison_df.to_string(index=False))
```

---

## 七、分段评估（⭐ 高分技巧）

### 7.1 为什么需要分段评估

预测误差不是均匀分布的——早高峰、节假日、异常时段的误差通常更大。只报整体MAE会掩盖这些问题。

### 7.2 分段评估维度

| 分段维度 | 评估内容 | 价值 |
|----------|----------|------|
| 时间段 | 早高峰/晚高峰/夜间 | 知道最差时段 |
| 日期类型 | 工作日/周末/节假日 | 节假日预测是难点 |
| 数值区间 | 高值/中值/低值 | 高值预测更难 |
| 节点类型 | 高流量/中流量/低流量节点 | 差异化优化 |

### 7.3 分段评估代码

```python
def segment_evaluate(df, y_true, y_pred, segment_col):
    """分段评估"""
    results = []
    for segment in df[segment_col].unique():
        mask = df[segment_col] == segment
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        results.append({'Segment': segment, 'MAE': mae, 'Count': mask.sum()})
    return pd.DataFrame(results).sort_values('MAE', ascending=False)

# 示例：按时段评估
df_eval = df_test.copy()
df_eval['period'] = pd.cut(df_eval['hour'], 
    bins=[0, 7, 9, 12, 14, 18, 21, 24],
    labels=['深夜', '早高峰前', '早高峰', '午间', '下午', '晚高峰', '夜间'])
print(segment_evaluate(df_eval, y_test, y_pred, 'period'))
```

---

*本指南随技能更新持续补充。参考：param_tuning.md（调参手册）*
