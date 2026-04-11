# 代码模板库

> 本文件是 mathorcup-model 技能的核心代码库。
> 包含预测、优化、评价三类问题的完整可运行代码模板。
> 所有模板均自带示例数据，可直接运行验证。

---

## 零、Config 全局路径配置

> **所有代码模板都从这里读取路径。**
> 只需修改此处一处，所有路径自动更新。
> 与 `preprocess_guide.md` 配合使用，数据探查→标准化→建模全链路统一路径。

```python
# ============================================================
# 全局配置 - 修改这里即可配置整个项目路径
# ============================================================

class Config:
    """MathorCup 建模项目全局配置"""

    # ── 基础路径 ────────────────────────────────────────────
    PROJECT_DIR = "./"          # 项目根目录（通常为 ./ 或 ../）
    DATA_DIR = "./data/"         # 原始数据目录
    OUTPUT_DIR = "./output/"     # 输出结果目录
    CODE_DIR = "./code/"         # 代码存放目录
    FIG_DIR = "./figures/"      # 图表输出目录
    MODEL_DIR = "./models/"     # 模型文件保存目录

    # ── 数据文件（按需修改文件名）──────────────────────────
    # 以下路径由 DATA_DIR + filename 自动拼接
    @staticmethod
    def data_path(filename):
        return os.path.join(Config.DATA_DIR, filename)

    @staticmethod
    def output_path(filename):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        return os.path.join(Config.OUTPUT_DIR, filename)

    @staticmethod
    def fig_path(filename):
        os.makedirs(Config.FIG_DIR, exist_ok=True)
        return os.path.join(Config.FIG_DIR, filename)

    @staticmethod
    def model_path(filename):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        return os.path.join(Config.MODEL_DIR, filename)

    # ── 数据列名（按题目修改）──────────────────────────────
    # 时间列
    date_col = "date"            # 时间戳列名
    # 目标列（预测目标）
    target_col = "value"         # 预测目标列名
    # ID列（多节点/多序列问题时使用）
    id_col = None                # 例如 "sensor_id" / "city_id"
    # 外部特征列（如果有）
    exogenous_cols = []          # 例如 ["temperature", "humidity"]

    # ── 数据划分 ───────────────────────────────────────────
    train_ratio = 0.7            # 训练集比例
    val_ratio = 0.15             # 验证集比例
    test_ratio = 0.15            # 测试集比例
    # 时序数据必须按时间顺序划分，禁止打乱！
    shuffle = False              # 时序数据=False，分类数据=True

    # ── 模型默认参数 ───────────────────────────────────────
    random_seed = 42             # 全局随机种子（保证可复现）
    n_estimators = 200           # 树模型默认迭代次数
    cv_folds = 5                 # 交叉验证折数
    early_stopping_rounds = 50   # 早停轮数
    verbose = 1                  # 日志详细程度 0/1/2

    # ── 输出控制 ───────────────────────────────────────────
    save_predictions = True      # 是否保存预测结果
    save_models = True           # 是否保存模型文件
    save_figures = True          # 是否保存图表
    figure_dpi = 150             # 图表分辨率


# ── 快捷别名（减少代码中 Config. 的敲击次数）─────────────
C = Config                       # 别名
DATA = Config.data_path          # 快速获取数据路径
OUT = Config.output_path          # 快速获取输出路径
FIG = Config.fig_path            # 快速获取图表路径
MODEL = Config.model_path         # 快速获取模型路径


# ============================================================
# 使用示例
# ============================================================
# from config import Config as C, DATA, OUT, FIG, MODEL
#
# # 加载数据
# df = pd.read_csv(DATA("train.csv"))
#
# # 保存结果
# results.to_csv(OUT("comparison.csv"), index=False)
#
# # 保存图表
# plt.savefig(FIG("correlation.png"), dpi=C.figure_dpi)
#
# # 保存模型
# joblib.dump(model, MODEL("xgb_model.pkl"))
```

---

## 模板索引

| 模板 | 文件 | 说明 |
|------|------|------|
| T1 | `template_t1_prediction.py` | 预测类标准流程（数据→特征→模型→评估）|
| T2 | `template_t2_optimization.py` | 优化类标准流程（建模→贪心→GA→结果分析）|
| T3 | `template_t3_evaluation.py` | 评价类标准流程（指标→权重→TOPSIS→排序）|
| T4 | `template_t4_visualization.py` | 可视化模板（对比图/热力图/收敛曲线）|
| T5 | `template_t5_pipeline.py` | 完整竞赛流程（多阶段+建模数据包输出）|

---

## T1: 预测类标准流程

```python
"""
template_t1_prediction.py
预测类问题标准流程模板
包含：数据探查 → 特征工程 → 多模型对比 → 评估 → 预测输出
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================
# Step 1: 配置
# =============================================
class Config:
    data_path = 'data/your_data.csv'
    target_col = 'target'
    date_col = 'date'
    test_ratio = 0.2  # 测试集比例
    val_ratio = 0.1   # 验证集比例
    random_state = 42
    output_dir = 'output/'
    fig_dir = 'figures/'

os.makedirs(Config.output_dir, exist_ok=True)
os.makedirs(Config.fig_dir, exist_ok=True)

# =============================================
# Step 2: 数据加载
# =============================================
def load_data(path):
    df = pd.read_csv(path)
    if Config.date_col in df.columns:
        df[Config.date_col] = pd.to_datetime(df[Config.date_col])
        df = df.sort_values(Config.date_col).reset_index(drop=True)
    print(f'数据规模: {df.shape}')
    print(f'时间范围: {df[Config.date_col].min()} ~ {df[Config.date_col].max()}')
    print(f'缺失值: {df.isnull().sum().sum()}')
    return df

# =============================================
# Step 3: 数据探查
# =============================================
def explore_data(df):
    """数据探查：缺失、分布、相关性"""
    print('\n=== 数据探查 ===')
    print(df.describe())

    # 缺失值分析
    missing = df.isnull().sum() / len(df) * 100
    if missing.sum() > 0:
        print('\n缺失比例>0%的字段:')
        print(missing[missing > 0])

    # 时间特征提取
    if Config.date_col in df.columns:
        df['hour'] = df[Config.date_col].dt.hour
        df['dayofweek'] = df[Config.date_col].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['month'] = df[Config.date_col].dt.month
        df['day'] = df[Config.date_col].dt.day

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 时序图
    axes[0,0].plot(df[Config.target_col].values[:500], linewidth=0.8)
    axes[0,0].set_title('时序趋势（前500点）')
    axes[0,0].set_xlabel('时间步')
    axes[0,0].set_ylabel(Config.target_col)

    # 分布图
    axes[0,1].hist(df[Config.target_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0,1].set_title('目标变量分布')
    axes[0,1].set_xlabel(Config.target_col)

    # 相关性热力图（数值列）
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('相关性矩阵')

    # 按星期分布
    if 'dayofweek' in df.columns:
        weekly = df.groupby('dayofweek')[Config.target_col].mean()
        axes[1,1].bar(range(7), weekly.values, alpha=0.7)
        axes[1,1].set_xticks(range(7))
        axes[1,1].set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        axes[1,1].set_title('按星期分布')

    plt.tight_layout()
    plt.savefig(f'{Config.fig_dir}data_exploration.png', dpi=150)
    plt.close()
    print(f'探查图已保存: {Config.fig_dir}data_exploration.png')

    return df

# =============================================
# Step 4: 特征工程
# =============================================
def create_features(df):
    """特征工程"""
    target = df[Config.target_col].copy()

    # 时间特征
    features = pd.DataFrame(index=df.index)
    if Config.date_col in df.columns:
        features['hour'] = df[Config.date_col].dt.hour
        features['dayofweek'] = df[Config.date_col].dt.dayofweek
        features['is_weekend'] = features['dayofweek'].isin([5, 6]).astype(int)
        features['month'] = df[Config.date_col].dt.month
        features['day'] = df[Config.date_col].dt.day
        # 周期性编码
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

    # 滞后特征（最重要！）
    for lag in [1, 2, 3, 7, 14, 24]:
        features[f'lag_{lag}'] = target.shift(lag)

    # 滑动统计特征
    for w in [7, 14, 24]:
        features[f'roll_mean_{w}'] = target.shift(1).rolling(w).mean()
        features[f'roll_std_{w}'] = target.shift(1).rolling(w).std()
        features[f'roll_max_{w}'] = target.shift(1).rolling(w).max()
        features[f'roll_min_{w}'] = target.shift(1).rolling(w).min()

    # 差分特征
    features['diff_1'] = target.diff(1)
    features['diff_7'] = target.diff(7)

    # 其他数值特征（排除目标列和日期列）
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in [Config.target_col, Config.date_col]:
            features[col] = df[col]

    # 移除滞后特征导致的NaN（保留足够训练的数据）
    features = features.dropna()
    target = target.loc[features.index]

    print(f'特征数量: {features.shape[1]}')
    print(f'样本数量: {features.shape[0]}')

    return features, target

# =============================================
# Step 5: 数据划分（时序数据专用）
# =============================================
def split_data(features, target):
    """时序数据划分：严格按时间顺序"""
    n = len(features)
    train_end = int(n * (1 - Config.test_ratio - Config.val_ratio))
    val_end = int(n * (1 - Config.test_ratio))

    X_train = features.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]

    y_train = target.iloc[:train_end]
    y_val = target.iloc[train_end:val_end]
    y_test = target.iloc[val_end:]

    print(f'\n数据划分:')
    print(f'  训练集: {len(y_train)} ({len(y_train)/n*100:.1f}%)')
    print(f'  验证集: {len(y_val)} ({len(y_val)/n*100:.1f}%)')
    print(f'  测试集: {len(y_test)} ({len(y_test)/n*100:.1f}%)')

    return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================
# Step 6: 多模型训练与对比
# =============================================
def train_models(X_train, y_train, X_val, y_val):
    """多模型训练"""
    models = {}

    # === LightGBM ===
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=Config.random_state,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models['LightGBM'] = lgb_model

    # === XGBoost ===
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=Config.random_state,
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['XGBoost'] = xgb_model

    # === 随机森林 ===
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=Config.random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model

    return models

def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    """多模型评估"""
    results = []
    best_model = None
    best_mae = np.inf

    for name, model in models.items():
        # 预测
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

        # 指标
        metrics = {
            'Model': name,
            'Train MAE': mean_absolute_error(y_train, pred_train),
            'Val MAE': mean_absolute_error(y_val, pred_val),
            'Test MAE': mean_absolute_error(y_test, pred_test),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
        }
        results.append(metrics)

        if metrics['Test MAE'] < best_mae:
            best_mae = metrics['Test MAE']
            best_model = model
            best_name = name

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Test MAE')
    print('\n=== 模型对比 ===')
    print(df_results.to_string(index=False))

    return df_results, best_model, best_name

# =============================================
# Step 7: 可视化对比
# =============================================
def plot_comparison(models, X_test, y_test):
    """多模型预测对比可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 预测曲线对比
    axes[0,0].plot(y_test.values[:200], 'k-', linewidth=2, label='True', alpha=0.8)
    colors = ['steelblue', 'coral', 'green', 'purple']
    for (name, model), color in zip(models.items(), colors):
        pred = model.predict(X_test)
        axes[0,0].plot(pred[:200], '--', color=color, linewidth=1.5, label=name, alpha=0.7)
    axes[0,0].legend()
    axes[0,0].set_title('预测对比（前200点）')
    axes[0,0].set_xlabel('时间步')
    axes[0,0].set_ylabel('值')

    # MAE柱状图
    mae_values = [mean_absolute_error(y_test, m.predict(X_test)) for m in models.values()]
    axes[0,1].bar(models.keys(), mae_values, color=colors[:len(models)], alpha=0.7)
    axes[0,1].set_title('Test MAE 对比')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].grid(axis='y', alpha=0.3)

    # 误差分布
    best_pred = list(models.values())[0].predict(X_test)
    errors = y_test.values - best_pred
    axes[1,0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1,0].axvline(0, color='red', linestyle='--')
    axes[1,0].set_title('预测误差分布')
    axes[1,0].set_xlabel('误差')

    # 特征重要性（取第一个树模型）
    if hasattr(list(models.values())[0], 'feature_importances_'):
        importance = list(models.values())[0].feature_importances_
        feature_names = X_test.columns
        top_n = 15
        top_idx = np.argsort(importance)[-top_n:]
        axes[1,1].barh(range(top_n), importance[top_idx][::-1], alpha=0.7)
        axes[1,1].set_yticks(range(top_n))
        axes[1,1].set_yticklabels([feature_names[i] for i in top_idx[::-1]])
        axes[1,1].set_title(f'{list(models.keys())[0]} 特征重要性 (Top{top_n})')

    plt.tight_layout()
    plt.savefig(f'{Config.fig_dir}model_comparison.png', dpi=150)
    plt.close()
    print(f'对比图已保存: {Config.fig_dir}model_comparison.png')

# =============================================
# Step 8: 输出建模数据包
# =============================================
def save_model_package(df_results, best_model, best_name, X_test, y_test):
    """输出建模数据包（供 paper-skill 读取）"""
    import json

    # 指标汇总
    best_pred = best_model.predict(X_test)
    metrics = {
        'best_model': best_name,
        'MAE': float(mean_absolute_error(y_test, best_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, best_pred))),
        'Test Size': int(len(y_test))
    }

    # 保存预测结果
    predictions = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': best_pred,
        'error': y_test.values - best_pred
    })
    predictions.to_csv(f'{Config.output_dir}predictions.csv', index=False)

    # 保存指标
    with open(f'{Config.output_dir}metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # 保存对比表
    df_results.to_csv(f'{Config.output_dir}comparison_table.csv', index=False)

    print(f'\n建模数据包已保存至 {Config.output_dir}')
    print(f'  metrics.json - 评估指标')
    print(f'  predictions.csv - 预测结果')
    print(f'  comparison_table.csv - 多模型对比')

# =============================================
# 主函数
# =============================================
def main():
    print('='*60)
    print('MathorCup 预测类建模流程')
    print('='*60)

    # 1. 数据加载
    df = load_data(Config.data_path)

    # 2. 数据探查
    df = explore_data(df)

    # 3. 特征工程
    features, target = create_features(df)

    # 4. 数据划分
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, target)

    # 5. 模型训练
    models = train_models(X_train, y_train, X_val, y_val)

    # 6. 模型评估
    df_results, best_model, best_name = evaluate_models(
        models, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # 7. 可视化
    plot_comparison(models, X_test, y_test)

    # 8. 输出数据包
    save_model_package(df_results, best_model, best_name, X_test, y_test)

    print('\n✅ 预测建模完成!')

if __name__ == '__main__':
    main()
```

---

## T2: 优化类标准流程

```python
"""
template_t2_optimization.py
优化类问题标准流程模板
包含：数学建模 → 贪心初始解 → GA迭代优化 → 收敛分析 → 敏感性分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import json, os, sys
sys.path.insert(0, '.')
from config import Config as C, OUT, FIG, MODEL  # 统一路径配置

# =============================================
# Step 1: 问题定义（根据具体题目修改）
# =============================================
class VRPProblem:
    """车辆路径问题模板"""
    def __init__(self, n_customers=50, n_vehicles=8, capacity=100, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.capacity = capacity

        # 客户坐标（0=仓库）
        self.nodes = np.random.rand(n_customers+1, 2) * 100
        self.demands = np.random.randint(5, 30, size=n_customers+1)
        self.demands[0] = 0  # 仓库无需求

        # 距离矩阵（欧氏距离）
        self.distance = np.zeros((n_customers+1, n_customers+1))
        for i in range(n_customers+1):
            for j in range(n_customers+1):
                self.distance[i][j] = np.sqrt(
                    (self.nodes[i][0]-self.nodes[j][0])**2 +
                    (self.nodes[i][1]-self.nodes[j][1])**2
                )

    def evaluate(self, routes):
        """计算总距离"""
        total = 0
        for route in routes:
            if not route:
                continue
            # 起点仓库
            total += self.distance[0][route[0]]
            for i in range(len(route)-1):
                total += self.distance[route[i]][route[i+1]]
            # 终点仓库
            total += self.distance[route[-1]][0]
        return total

    def is_valid(self, routes):
        """检查解是否满足容量约束"""
        for route in routes:
            load = sum(self.demands[c] for c in route)
            if load > self.capacity:
                return False
        return True

# =============================================
# Step 2: 贪心初始解
# =============================================
def greedy_vrp(problem):
    """最近邻贪心算法生成初始解"""
    routes = [[] for _ in range(problem.n_vehicles)]
    vehicle_loads = [0] * problem.n_vehicles
    unassigned = set(range(1, problem.n_customers+1))

    for v in range(problem.n_vehicles):
        current = 0  # 从仓库出发
        while unassigned:
            # 找最近的未分配客户
            nearest = min(unassigned,
                         key=lambda c: problem.distance[current][c])

            # 检查容量
            if vehicle_loads[v] + problem.demands[nearest] <= problem.capacity:
                routes[v].append(nearest)
                vehicle_loads[v] += problem.demands[nearest]
                unassigned.remove(nearest)
                current = nearest
            else:
                break  # 当前车辆满，换下一辆

        if not unassigned:
            break

    return routes

# =============================================
# Step 3: 遗传算法
# =============================================
class GAVRP:
    def __init__(self, problem, n_pop=100, n_iter=500, pc=0.8, pm=0.05):
        self.problem = problem
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.pc = pc
        self.pm = pm
        self.history = []
        self.best_routes = None
        self.best_fitness = np.inf

    def encode(self, routes):
        """解编码：将路线转为单条染色体"""
        chromosome = []
        for route in routes:
            chromosome.extend(route)
        return chromosome

    def decode(self, chromosome):
        """解码：将染色体转为路线列表"""
        routes = [[] for _ in range(self.problem.n_vehicles)]
        vehicle_loads = [0] * self.problem.n_vehicles
        used = set()

        for customer in chromosome:
            if customer in used:
                continue
            # 选择负载最少的车辆
            best_v = min(range(self.problem.n_vehicles),
                        key=lambda v: vehicle_loads[v])

            # 检查容量
            if vehicle_loads[best_v] + self.problem.demands[customer] <= \
               self.problem.capacity:
                routes[best_v].append(customer)
                vehicle_loads[best_v] += self.problem.demands[customer]
                used.add(customer)

        return routes

    def fitness(self, chromosome):
        return self.problem.evaluate(self.decode(chromosome))

    def crossover(self, p1, p2):
        """顺序交叉（OX）"""
        if random.random() > self.pc:
            return p1[:], p2[:]

        n = len(p1)
        a, b = sorted([random.randint(0, n), random.randint(0, n)])

        def ox(p1, p2):
            child = [None] * n
            child[a:b] = p1[a:b]
            remaining = [x for x in p2 if x not in child[a:b]]
            j = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = remaining[j]
                    j += 1
            return child

        return ox(p1, p2), ox(p2, p1)

    def mutate(self, chromosome):
        """插入变异"""
        if random.random() > self.pm:
            return chromosome
        n = len(chromosome)
        a, b = random.sample(range(n), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return chromosome

    def solve(self):
        """主循环"""
        # 贪心初始化
        pop = [self.encode(greedy_vrp(self.problem))]
        for _ in range(self.n_pop - 1):
            pop.append(self.encode(greedy_vrp(self.problem)))

        for gen in range(self.n_iter):
            fitness = [self.fitness(ch) for ch in pop]

            # 记录最优
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_routes = self.decode(pop[best_idx])
            self.history.append(self.best_fitness)

            # 锦标赛选择
            selected = []
            for _ in range(self.n_pop):
                idx = random.sample(range(self.n_pop), 3)
                winner = min(idx, key=lambda i: fitness[i])
                selected.append(pop[winner])

            # 交叉
            offspring = []
            for i in range(0, self.n_pop, 2):
                c1, c2 = self.crossover(selected[i], selected[i+1])
                offspring.extend([c1, c2])

            # 变异
            offspring = [self.mutate(ch) for ch in offspring[:self.n_pop]]

            # 精英保留
            pop = offspring

            if gen % 100 == 0:
                print(f'Gen {gen}: Best = {self.best_fitness:.2f}')

            # 早停
            if len(self.history) > 100:
                if max(self.history[-100:]) - min(self.history[-100:]) < 0.01:
                    print(f'早停于第 {gen} 代')
                    break

        return self.best_routes, self.best_fitness

    def plot_convergence(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.history, 'b-', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.title('GA Convergence')
        plt.grid(True, alpha=0.3)
        plt.savefig(FIG('ga_convergence.png'), dpi=C.figure_dpi)
        plt.close()

# =============================================
# 主函数
# =============================================
def main():
    print('='*60)
    print('MathorCup 优化类建模流程')
    print('='*60)

    # 创建问题
    problem = VRPProblem(n_customers=50, n_vehicles=8, capacity=100)

    # 贪心初始解
    greedy_routes = greedy_vrp(problem)
    greedy_cost = problem.evaluate(greedy_routes)
    print(f'贪心初始解: {greedy_cost:.2f}')

    # GA优化
    ga = GAVRP(problem, n_pop=100, n_iter=500)
    best_routes, best_cost = ga.solve()
    print(f'GA最优解: {best_cost:.2f}')
    print(f'改进幅度: {(greedy_cost-best_cost)/greedy_cost*100:.1f}%')

    # 收敛曲线
    ga.plot_convergence()

    # 保存结果
    result = {
        'greedy_cost': float(greedy_cost),
        'ga_cost': float(best_cost),
        'improvement': f'{(greedy_cost-best_cost)/greedy_cost*100:.1f}%',
        'routes': [list(r) for r in best_routes]
    }

    os.makedirs('output', exist_ok=True)
    with open(OUT('optimization_result.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print('\n✅ 优化建模完成!')

if __name__ == '__main__':
    main()
```

---

*完整模板文件请参考 output/ 目录下的生成文件*
