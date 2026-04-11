# 预处理建模指南

> 适用问题类型：预测、分类、聚类、回归
> 本指南包含：数据探查 → 缺失值处理 → 异常值处理 → 特征工程 → 特征选择 → 数据增强

---

## 一、历年优秀论文预处理统计

> 以下数据来自 MathorCup 2020-2024 年 41 篇优秀论文全文扫描。
> 特征提取（41%）是最常见的预处理手段。

| 方法 | 频率 | 说明 |
|------|------|------|
| **特征提取** | 17/41 (41%) | 从原始数据中提取统计/时序/频域特征 |
| 数据增强 | 7/41 (17%) | 深度学习类论文常见 |
| **特征选择** | 6/41 (15%) | 降维、剔除冗余特征 |
| 指数平滑 | 3/41 (7%) | 传统时序平滑 |
| **滑动窗口** | 3/41 (7%) | 时序数据构造特征 |
| 移动平均 | 3/41 (7%) | 平滑处理 |
| Z-score | 2/41 (5%) | 标准化 |
| 缺失值处理 | 2/41 (5%) | 含 KNN 插值、均值等 |
| 异常值处理 | 2/41 (5%) | 含剔除、正态转换 |

---

## 二、数据探查（必做）

> 历年优秀论文中 100% 都包含数据探查环节。
> 这是评委第一眼看到的内容，决定论文专业度印象。

### 2.1 基础探查模板

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df, date_col=None, target_col=None):
    """完整数据探查模板"""
    print("=" * 50)
    print(f"数据规模: {df.shape[0]}行 × {df.shape[1]}列")
    print("=" * 50)
    
    # 1. 数据类型
    print("\n[1] 数据类型:")
    print(df.dtypes)
    
    # 2. 缺失值（必做）
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    missing_df = pd.DataFrame({'缺失数': missing, '缺失率%': missing_pct})
    missing_df = missing_df[missing_df['缺失数'] > 0]
    if len(missing_df) > 0:
        print("\n[2] 含缺失值的列:")
        print(missing_df)
    else:
        print("\n[2] 无缺失值 ✓")
    
    # 3. 统计描述（必做）
    print("\n[3] 统计描述:")
    print(df.describe().T)
    
    # 4. 分布可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 目标变量分布
    if target_col and target_col in df.columns:
        axes[0,0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7)
        axes[0,0].set_title(f'{target_col} 分布')
        axes[0,0].set_xlabel(target_col)
        # 打印偏度和峰度
        from scipy import stats
        skew = stats.skew(df[target_col].dropna())
        kurt = stats.kurtosis(df[target_col].dropna())
        axes[0,0].text(0.05, 0.95, f'偏度: {skew:.2f}\n峰度: {kurt:.2f}', 
                       transform=axes[0,0].transAxes, va='top')
    
    # 箱线图（检测异常值）
    if target_col and target_col in df.columns:
        axes[0,1].boxplot(df[target_col].dropna())
        axes[0,1].set_title(f'{target_col} 箱线图')
    
    # 相关性热力图（数值列）
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=axes[1,0], annot_kws={'size': 8})
        axes[1,0].set_title('相关性矩阵')
    
    # 时间趋势（如果有时间列）
    if date_col and date_col in df.columns:
        df_sorted = df.sort_values(date_col)
        axes[1,1].plot(df_sorted[date_col], df_sorted[target_col], linewidth=0.8)
        axes[1,1].set_title(f'{target_col} 时间趋势')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIG('data_exploration.png'), dpi=C.figure_dpi)
    plt.show()
    return df
```

### 2.2 时间特征提取（时序类问题必做）

```python
def extract_time_features(df, date_col):
    """时间特征提取 - 历年论文高频技巧"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 基本时间特征
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['hour'] = df[date_col].dt.hour
    df['dayofweek'] = df[date_col].dt.dayofweek        # 0=周一
    df['dayofyear'] = df[date_col].dt.dayofyear
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_col].dt.quarter
    
    # 布尔特征
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    
    # 周期性编码（正弦/余弦，解决周期性边界问题）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df
```

---

## 三、缺失值处理

> 历年优秀论文中提及"缺失值处理"的频率相对较低（5%），但这不代表可以忽略。
> 真实比赛中缺失值普遍存在，处理方式直接影响模型效果。

### 3.1 处理策略决策树

```
缺失率 < 1%  →  直接删除或简单均值填充
缺失率 1-10% →  KNN插值 / 线性插值 / 向前向后填充
缺失率 10-50%→  模型预测填充（Random Forest / XGBoost）
缺失率 > 50% →  考虑删除该特征，或分桶处理
```

### 3.2 完整处理代码

```python
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

def handle_missing_values(df, strategy='auto'):
    """
    智能缺失值处理
    
    strategy:
        - 'simple': 均值/中位数填充
        - 'knn': KNN插值（适用于有空间/时间相关性的数据）
        - 'model': Random Forest预测填充（最准确但最慢）
        - 'auto': 根据缺失率自动选择
    """
    df = df.copy()
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0].index.tolist()
    
    if not missing_cols:
        print("✓ 数据无缺失值")
        return df
    
    for col in missing_cols:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        print(f"处理 {col}: 缺失率 {missing_pct:.1f}%")
        
        if strategy == 'auto':
            if missing_pct < 5:
                s = 'simple'
            elif missing_pct < 30:
                s = 'knn'
            else:
                s = 'model'
        else:
            s = strategy
        
        if s == 'simple':
            # 数值型用中位数（抗异常），分类型用众数
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        elif s == 'knn':
            # KNN插值，保持数据局部结构
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if col in num_cols and len(num_cols) > 1:
                imp = KNNImputer(n_neighbors=5)
                df[num_cols] = imp.fit_transform(df[num_cols])
                
        elif s == 'model':
            # 用其他特征预测缺失值（最准确）
            if df[col].dtype in [np.float64, np.int64]:
                train_mask = df[col].notnull()
                test_mask = df[col].isnull()
                if train_mask.sum() > 50 and test_mask.sum() > 0:
                    features = [c for c in df.columns if c != col and df[c].dtype in [np.float64, np.int64]]
                    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, col]
                    X_test = df.loc[test_mask, features]
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)
                    df.loc[test_mask, col] = rf.predict(X_test)
    
    print(f"✓ 缺失值处理完成，剩余缺失: {df.isnull().sum().sum()}")
    return df
```

---

## 四、异常值处理

> 异常值检测是数据质量评估的重要环节。
> 历年优秀论文中通常会单独说明异常值处理策略。

### 4.1 多种检测方法

```python
def detect_outliers(df, col, method='iqr'):
    """
    异常值检测
    
    method:
        - 'iqr': IQR四分位法（适用于非正态分布）
        - 'zscore': Z-score法（适用于正态分布，|z| > 3 为异常）
        - 'isolation_forest': 孤立森林（适用于多维数据）
    """
    if method == 'iqr':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        
    elif method == 'zscore':
        from scipy import stats
        z = np.abs(stats.zscore(df[col].dropna()))
        mask = z > 3
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        X = df[[col]].dropna()
        iso = IsolationForest(contamination=0.01, random_state=42)
        pred = iso.fit_predict(X)
        mask = pd.Series(pred == -1, index=X.index)
    
    outliers = df[mask]
    print(f"检测到 {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.1f}%)")
    return mask, outliers

def handle_outliers(df, col, method='cap'):
    """
    异常值处理
    
    method:
        - 'cap': 封顶（Winsorize），将异常值替换为边界值
        - 'remove': 删除异常样本
        - 'log': 对数变换（适用于右偏分布）
        - 'nan': 替换为缺失值后用模型填充
    """
    mask, outliers = detect_outliers(df, col)
    df_processed = df.copy()
    
    if method == 'cap':
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
        
    elif method == 'log':
        # 适用于右偏分布（如收入、流量数据）
        df_processed[col] = np.log1p(df_processed[col].clip(lower=0))
        
    elif method == 'remove':
        df_processed = df_processed[~mask]
        
    elif method == 'nan':
        df_processed.loc[mask, col] = np.nan
        
    return df_processed
```

---

## 五、特征工程

> 特征提取是历年优秀论文中使用最频繁的预处理手段（41%，17/41篇）。
> 这是拉开差距的关键环节。

### 5.1 时序特征提取（统计特征）

```python
def extract_statistical_features(series, window=7):
    """
    从时间序列中提取统计特征
    
    历年论文中 tsfresh 工具链最常用，自动提取大量统计特征。
    这里提供核心特征的手动实现。
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    s = pd.Series(series).rolling(window=window, min_periods=1)
    
    features = {
        # 基本统计
        'mean': s.mean().iloc[-1],
        'std': s.std().iloc[-1],
        'min': s.min().iloc[-1],
        'max': s.max().iloc[-1],
        'median': s.median().iloc[-1],
        # 极值相关
        'range': s.max().iloc[-1] - s.min().iloc[-1],
        'skew': s.skew().iloc[-1] if len(s) > 2 else 0,
        'kurt': s.kurt().iloc[-1] if len(s) > 3 else 0,
        # 变化率
        'diff_mean': s.diff().mean().iloc[-1],
        'diff_std': s.diff().std().iloc[-1],
        # 熵特征（区分规律性和随机性）
        'entropy': -np.sum(np.histogram(s.dropna(), bins=10, density=True)[0] * 
                          np.log(np.histogram(s.dropna(), bins=10, density=True)[0] + 1e-10)),
    }
    return features

def extract_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 24]):
    """
    滞后特征（滑动窗口）— 历年论文高频技巧
    
    lags: 滞后期数列表
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
        df[f'{target_col}_lead{lag}'] = df[target_col].shift(-lag)  # 未来值（标签用）
    return df

def extract_rolling_features(df, target_col, windows=[3, 7, 14, 30]):
    """
    滚动统计特征
    """
    df = df.copy()
    for w in windows:
        df[f'{target_col}_roll_mean{w}'] = df[target_col].rolling(w, min_periods=1).mean()
        df[f'{target_col}_roll_std{w}'] = df[target_col].rolling(w, min_periods=1).std()
        df[f'{target_col}_roll_max{w}'] = df[target_col].rolling(w, min_periods=1).max()
        df[f'{target_col}_roll_min{w}'] = df[target_col].rolling(w, min_periods=1).min()
    return df
```

### 5.2 tsfresh 自动特征提取（推荐）

> tsfresh 是历年论文中最常用的时序特征自动提取工具。
> 可以自动提取几百个统计特征，大大提升特征工程效率。

```python
# 安装: pip install tsfresh
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

def tsfresh_auto_extract(df, time_col, target_col, max_features=300):
    """
    使用 tsfresh 自动提取时序特征
    
    参数:
        max_features: 最大特征数量（防止维度灾难）
    """
    # 准备数据（tsfresh 要求 id + time + value 格式）
    df_melt = df.melt(id_vars=[time_col], value_vars=[target_col],
                      var_name='variable', value_name='value')
    df_melt['id'] = 1  # 统一为单条序列
    
    # 提取特征
    features = extract_features(df_melt, column_id='id', 
                                column_sort=time_col, column_value='value',
                                impute_function=impute,
                                disable_progressbar=False)
    
    # 特征选择（去除无关特征）
    labels = df[target_col].iloc[1:]  # 需要与特征对齐
    features_selected = select_features(features, labels)
    
    print(f"原始特征数: {features.shape[1]}, 选择后: {features_selected.shape[1]}")
    return features_selected
```

### 5.3 频域特征（周期性数据）

```python
def extract_frequency_features(series, fs=1.0):
    """
    频域特征提取 - 适用于有周期性的数据
    
    fs: 采样频率（默认1，即每天一个点）
    """
    from scipy.fft import fft
    from scipy.signal import welch
    
    s = np.array(series.dropna())
    n = len(s)
    
    # FFT 频谱
    fft_vals = fft(s - s.mean())
    fft_freq = np.fft.fftfreq(n, 1/fs)
    power = np.abs(fft_vals[:n//2])**2
    freqs = np.abs(fft_freq[:n//2])
    
    # 主周期
    dominant_freq = freqs[np.argmax(power[1:]) + 1]  # 排除直流分量
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
    
    # Welch 功率谱密度
    freqs_welch, psd = welch(s, fs=fs, nperseg=min(256, n))
    
    return {
        'dominant_freq': dominant_freq,
        'dominant_period': dominant_period,
        'spectral_entropy': -np.sum(psd * np.log(psd + 1e-10)),
        'spectral_energy': np.sum(psd),
    }
```

---

## 六、特征选择

> 特征选择是历年优秀论文的高频操作（15%，6/41篇）。
> 特征过多会导致过拟合，特征过少则欠拟合。

### 6.1 特征选择方法对比

| 方法 | 原理 | 适用场景 | 历年论文使用频率 |
|------|------|---------|---------------|
| 方差阈值 | 剔除低方差特征 | 初步筛选 | 中 |
| 相关系数 | 剔除高度相关特征（>0.9） | 去除冗余 | **高** |
| 随机森林重要性 | 基于树模型评估特征贡献 | 通用 | **高** |
| XGBoost 重要性 | 基于梯度提升评估 | 表格数据 | **高** |
| LASSO 正则化 | L1稀疏化，自动特征选择 | 高维数据 | 中 |
| RFE 递归消除 | 逐步剔除最不重要特征 | 小规模 | 低 |
| PCA/因子分析 | 降维到主成分 | 强相关特征组 | **高**（44%） |

### 6.2 组合选择代码

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def multi_strategy_feature_selection(X, y, task='regression'):
    """
    多策略组合特征选择
    
    综合方差过滤 + 相关性过滤 + 模型重要性
    """
    results = pd.DataFrame({'feature': X.columns})
    
    # 策略1: 方差过滤
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X)
    results['var_pass'] = var_selector.get_support()
    
    # 策略2: 相关性过滤（与目标变量）
    corr_scores = X.corrwith(y).abs()
    results['corr_score'] = corr_scores.values
    results['corr_pass'] = corr_scores > 0.05
    
    # 策略3: 随机森林重要性
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    results['rf_importance'] = rf_importance.values
    
    # 策略4: XGBoost 重要性
    xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    xgb_model.fit(X, y)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    results['xgb_importance'] = xgb_importance.values
    
    # 综合得分（归一化后加权平均）
    for col in ['corr_score', 'rf_importance', 'xgb_importance']:
        results[f'{col}_norm'] = (results[col] - results[col].min()) / (results[col].max() - results[col].min() + 1e-10)
    
    results['final_score'] = (results['corr_score_norm'] * 0.3 + 
                               results['rf_importance_norm'] * 0.35 + 
                               results['xgb_importance_norm'] * 0.35)
    
    results = results.sort_values('final_score', ascending=False)
    
    # 选择综合得分前50%的特征
    n_select = max(5, int(len(results) * 0.5))
    selected = results.head(n_select)['feature'].tolist()
    
    print(f"特征选择完成: {X.shape[1]} → {len(selected)} 个特征")
    print(f"Top 10 特征: {results.head(10)['feature'].tolist()}")
    
    return selected, results

# 使用示例
# selected_features, report = multi_strategy_feature_selection(X_train, y_train)
# X_train_selected = X_train[selected_features]
```

---

## 七、数据增强（深度学习类）

> 数据增强在历年论文中出现频率较低（17%），但对于深度学习类题目（图像、文本）
> 是提升模型泛化能力的核心手段。

### 7.1 时序数据增强

```python
def augment_time_series(series, n_augment=5):
    """时序数据增强 - 生成更多训练样本"""
    augmented = []
    
    for _ in range(n_augment):
        # 方法1: 噪声注入
        noise = np.random.normal(0, 0.01 * np.std(series), len(series))
        aug = series + noise
        
        # 方法2: 随机缩放
        scale = np.random.uniform(0.9, 1.1)
        aug = series * scale
        
        # 方法3: 随机时间偏移
        shift = np.random.randint(-3, 3)
        aug = pd.Series(series).shift(shift).fillna(method='bfill').values
        
        augmented.append(aug)
    
    return np.array(augmented)

def jitter(x, sigma=0.03):
    """加随机噪声"""
    return x + np.random.normal(loc=0, scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    """随机缩放"""
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1], 1))
    return x * factor
```

---

## 八、数据标准化与归一化

> 标准化方法的选择取决于数据分布和模型类型。

### 8.1 方法对比

| 方法 | 公式 | 适用场景 | 受异常值影响 |
|------|------|---------|------------|
| Z-score (StandardScaler) | (x-μ)/σ | 大部分场景 | 中等 |
| Min-Max | (x-min)/(max-min) | 需要有界输出 | 敏感 |
| RobustScaler | (x-median)/IQR | 有异常值 | **不敏感** |
| Log Transform | log(1+x) | 右偏分布 | 不敏感 |
| Box-Cox | 需优化λ参数 | 偏态数据 | 不敏感 |

### 8.2 完整实现

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

def robust_normalize(series):
    """
    健壮的归一化（自动检测最优方法）
    """
    # 右偏检测
    skewness = stats.skew(series.dropna())
    
    if abs(skewness) > 1:
        # 强偏态：先做对数变换
        normalized = np.log1p(series.clip(lower=0))
        print(f"应用对数变换，偏度 {skewness:.2f} → {stats.skew(normalized.dropna()):.2f}")
    else:
        normalized = series
    
    return normalized

def get_scaler(X, method='standard'):
    """
    获取标准化器
    
    method:
        - 'standard': Z-score 标准化
        - 'minmax': Min-Max 归一化
        - 'robust': RobustScaler（抗异常值）
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
    }
    return scalers.get(method, StandardScaler())

# 时序数据标准化注意事项：只用训练集统计量！
def scale_train_test(X_train, X_test, method='standard'):
    scaler = get_scaler(X_train, method)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # 用训练集的 scaler！
    return X_train_scaled, X_test_scaled, scaler
```

---

## 九、历年论文典型预处理流程

### 9.1 时序预测类问题（最常见）

```
原始数据
  ↓ 数据探查（分布、缺失、相关性）
  ↓ 缺失值处理（KNN 或 RF 预测）
  ↓ 异常值检测与处理（IQR / log变换）
  ↓ 时间特征提取（年/月/日/时/星期/周期性编码）
  ↓ 滑动窗口构造（lag特征 + 滚动统计）
  ↓ tsfresh 自动特征提取（可选）
  ↓ 相关性分析（剔除 >0.9 的特征）
  ↓ 特征选择（RF/XGB 重要性）
  ↓ Z-score 标准化
  → 输入模型
```

### 9.2 聚类类问题

```
原始数据
  ↓ 数据探查
  ↓ 缺失值处理
  ↓ 异常值处理
  ↓ 特征提取（统计特征、时间序列特征）
  ↓ PCA 降维（如果特征过多）
  ↓ 标准化（Min-Max 或 Z-score）
  → 聚类算法（K-Means / 层次聚类 / DBSCAN）
  → 轮廓系数 + 肘部法则 评估
  → 各类特点分析
```

### 9.3 优化类问题

```
原始数据
  ↓ 数据探查（了解约束条件分布）
  ↓ 缺失值/异常值处理
  ↓ 统计汇总（均值/方差/极值，用于参数设置）
  ↓ 归一化（如果是多指标问题）
  → 数学规划建模
  → 元启发式算法求解
```
