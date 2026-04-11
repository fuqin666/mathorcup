# 模型调参手册

> 本手册是 mathorcup-model 技能的参考资料。
> 包含各主流模型的调参参考，适用于竞赛实战。

---

## 零、快速推荐表（开箱即用）

> 根据问题类型 + 数据规模，直接抄参数。
> 适合第一轮 baseline，赛前熟悉参数含义，赛中快速出结果。
> **进阶调参**见各模型详细章节。

### 0.1 推荐参数速查表

| 问题类型 | 数据规模 | 首选模型 | 首选参数 | 备选模型 | 备选参数 |
|----------|----------|----------|----------|----------|----------|
| **时序预测** | < 5K 行 | **LSTM** | `seq_len=24, hidden=64, layers=2, dropout=0.2, lr=0.001, epochs=100` | XGBoost | `depth=6, lr=0.05, n_est=500, subsample=0.8` |
| **时序预测** | 5K-100K | **LightGBM** | `num_leaves=63, depth=8, lr=0.05, n_est=1000, min_child=20` | LSTM | `seq_len=168, hidden=128, layers=2, dropout=0.3` |
| **时序预测** | > 100K | **XGBoost** | `depth=8, lr=0.03, n_est=1500, subsample=0.7, colsample=0.6` | LightGBM | `num_leaves=127, depth=10, lr=0.02, n_est=2000` |
| **时序预测·多节点** | 任意 | **STGCN** | `seq_len=12, pred_len=12, channels=[64,64,128], dropout=0.2, lr=0.001` | LSTM+图特征 | `seq_len=24, hidden=128, layers=2` |
| **分类·表格** | < 5K | **随机森林** | `n_est=300, max_depth=15, min_samples_leaf=5, max_features=sqrt` | XGBoost | `depth=6, lr=0.1, n_est=300` |
| **分类·表格** | 5K-100K | **XGBoost** | `depth=8, lr=0.05, n_est=500, subsample=0.8, colsample=0.7` | LightGBM | `num_leaves=63, depth=8, lr=0.05, n_est=500` |
| **分类·表格** | > 100K | **LightGBM** | `num_leaves=127, depth=10, lr=0.03, n_est=1000, min_child=30` | CatBoost | `depth=8, lr=0.05, iterations=1000` |
| **回归·表格** | < 5K | **XGBoost** | `depth=5, lr=0.1, n_est=300, subsample=0.8, colsample=0.8` | 随机森林 | `n_est=200, max_depth=12` |
| **回归·表格** | 5K-100K | **LightGBM** | `num_leaves=63, depth=8, lr=0.05, n_est=800, min_child=20` | XGBoost | `depth=7, lr=0.03, n_est=1000` |
| **回归·表格** | > 100K | **XGBoost** | `depth=9, lr=0.02, n_est=2000, subsample=0.7, colsample=0.6` | LightGBM | `num_leaves=255, depth=12, lr=0.02, n_est=3000` |
| **聚类** | 任意 | **K-Means** | `n_clusters=5, init=k-means++, n_init=10, max_iter=300` | 层次聚类 | `n_clusters=5, linkage=ward` |
| **路径/调度优化** | < 50节点 | **模拟退火** | `T=1000, alpha=0.995, iter=5000, T_min=1` | 遗传算法 | `pop=100, gen=200, pc=0.8, pm=0.1` |
| **路径/调度优化** | 50-200节点 | **遗传算法** | `pop=150, gen=300, pc=0.8, pm=0.05, elitism=10` | 粒子群 | `pop=100, w=0.7, c1=1.5, c2=1.5` |
| **路径/调度优化** | > 200节点 | **NSGA-III** | `pop=200, gen=300, pc=0.9, pm=1/n, ref_points=paretos` | 混合遗传 | `pop=200, gen=200, local_search=true` |

> **关于"多节点"**：指有多个独立序列（如多个传感器、多个门店），STGCN 适合空间关系已知的情况，
> 若空间关系未知则用 LSTM 对每个节点单独建模后集成。

### 0.2 数据规模估算方法

```python
def estimate_data_scale(df, target_col):
    """估算数据规模，返回推荐模型路线"""
    n_rows = len(df)
    n_features = len([c for c in df.columns if c != target_col])
    
    # 估算时序周期（小时数据场景）
    if 'hour' in str(df.columns).lower():
        cycle_estimate = 24  # 日周期
    elif 'day' in str(df.columns).lower():
        cycle_estimate = 7   # 周周期
    else:
        cycle_estimate = None
    
    print(f"数据规模: {n_rows}行 × {n_features}特征")
    if n_rows < 5000:
        scale = "小规模"
    elif n_rows < 100000:
        scale = "中规模"
    else:
        scale = "大规模"
    print(f"推荐路线: {scale} → 对应模型见上方推荐表")
    return scale
```

### 0.3 开箱参数检查清单

> 运行第一个 baseline 前，逐项确认：

```
[ ] 数据规模估算：< 5K / 5K-100K / > 100K？
[ ] 问题类型：时序预测 / 分类 / 回归 / 聚类 / 优化？
[ ] 有无多节点：单序列 / 多节点（节点间是否有关联）？
[ ] 评估指标：RMSE / MAE / MAPE / ACC / F1 ？
[ ] 是否不平衡：类别/数值是否严重倾斜？
[ ] 有无节假日/特殊点：需要特殊处理吗？
[ ] 随机种子：全部设为 42

确认后 → 查上方推荐表 → 复制参数 → 开始跑！
```

---

## 一、调参基本原则

### 1.1 调参优先级

| 优先级 | 参数类型 | 说明 |
|--------|----------|------|
| 高 | 影响模型容量的参数 | max_depth、num_leaves、n_estimators |
| 中 | 影响学习过程的参数 | learning_rate、dropout |
| 低 | 正则化参数 | reg_alpha、reg_lambda、min_child_weight |

### 1.2 调参策略选择

| 策略 | 适用场景 | 优缺点 |
|------|----------|--------|
| 网格搜索 | 参数空间小（≤3个参数） | 保证找到最优，但计算量大 |
| 随机搜索 | 参数空间中等 | 比网格快 2-3 倍，适合大参数空间 |
| 贝叶斯优化 | 参数空间大，评估代价高 | 效率最高，需要 install optuna |
| 遗传算法 | 复杂非线性参数空间 | 全局搜索，适合特殊场景 |

### 1.3 时序数据调参注意事项

- **必须使用时序交叉验证**（不能用随机 KFold，会数据泄露）
- **训练集必须早于验证集和测试集**
- **滞后特征**注意序列长度与参数的关系
- **节假日/特殊日期**如无数据，不放入交叉验证折

---

## 二、XGBoost 调参

### 2.1 参数详解

| 参数 | 含义 | 常用范围 | 调参建议 |
|------|------|----------|----------|
| n_estimators | 树的数量 | 100-2000 | 与 learning_rate 联合调，值大选低学习率 |
| max_depth | 最大深度 | 3-10 | 数据量小→3-5，数据量大→6-10 |
| learning_rate | 学习率 | 0.01-0.3 | 通常 0.05-0.1，配合 n_estimators |
| min_child_weight | 最小叶子权重和 | 1-10 | 防止过拟合，数据噪声多时增大 |
| subsample | 行采样比例 | 0.6-1.0 | 0.7-0.8 常用，防止过拟合 |
| colsample_bytree | 列采样比例 | 0.6-1.0 | 0.6-0.8 常用 |
| gamma | 最小损失减少 | 0-5 | 控制分裂，值越大越保守 |
| reg_alpha | L1正则化 | 0-1 | 特征多时增大，减少过拟合 |
| reg_lambda | L2正则化 | 0-10 | 默认1，可增大防过拟合 |
| scale_pos_weight | 正负样本比例 | 自动计算 | 不平衡数据时使用 |
| objective | 目标函数 | reg:squarederror / reg:squaredlogerror | 回归默认前者，MAPE目标用后者 |
| eval_metric | 评估指标 | rmse / mae / mape | 与比赛评估指标一致 |

### 2.2 推荐调参顺序

```
Step 1: max_depth + min_child_weight → 固定树结构
Step 2: subsample + colsample_bytree → 防止过拟合
Step 3: learning_rate + n_estimators → 找最优组合
Step 4: reg_alpha + reg_lambda → 微调
```

### 2.3 典型竞赛配置

**普通回归题：**
```python
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42
}
```

**时序预测题（配合时序特征）：**
```python
xgb_params = {
    'max_depth': 8,
    'learning_rate': 0.03,
    'n_estimators': 1000,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'random_state': 42
}
```

---

## 三、LightGBM 调参

### 3.1 参数详解

| 参数 | 含义 | 常用范围 | 调参建议 |
|------|------|----------|----------|
| num_leaves | 叶子数 | 20-150 | 控制模型复杂度，2^max_depth 附近 |
| max_depth | 最大深度 | -1（无限制）或 6-12 | 与 num_leaves 配合 |
| learning_rate | 学习率 | 0.01-0.3 | 通常 0.05 |
| n_estimators | 树数量 | 100-5000 | 与 learning_rate 联合 |
| min_child_samples | 最小叶子样本数 | 10-100 | 数据量小增大，数据量大减小 |
| subsample | 行采样比例 | 0.6-1.0 | 0.7-0.8 常用 |
| colsample_bytree | 列采样比例 | 0.6-1.0 | 0.6-0.8 常用 |
| reg_alpha | L1正则化 | 0-10 | 稀疏特征时增大 |
| reg_lambda | L2正则化 | 0-10 | 默认0，增大防过拟合 |
| min_gain_to_split | 最小分裂增益 | 0-1 | 防止无效分裂 |
| path_smooth | 路径平滑 | 0-10 | 值越大越平滑 |
| objective | 目标函数 | regression / huber / mape | 按评估指标选 |
| metric | 评估指标 | l1 / l2 / mape | 与比赛评估指标一致 |
| boosting_type | 提升类型 | gbdt / dart / goss | 竞赛默认 gbdt |

### 3.2 推荐调参顺序

```
Step 1: num_leaves + max_depth → 固定结构
Step 2: min_child_samples + subsample → 控制过拟合
Step 3: learning_rate + n_estimators → 找最优组合
Step 4: reg_alpha + reg_lambda → 微调
```

### 3.3 典型竞赛配置

```python
lgb_params = {
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'regression',
    'metric': 'mae',
    'random_state': 42,
    'verbose': -1
}
```

---

## 四、CatBoost 调参

### 4.1 参数详解

| 参数 | 含义 | 常用范围 | 调参建议 |
|------|------|----------|----------|
| iterations | 迭代次数 | 500-3000 | 与 learning_rate 联合 |
| learning_rate | 学习率 | 0.01-0.3 | 通常 0.03-0.1 |
| depth | 深度 | 4-10 | 6-8 最常用 |
| l2_leaf_reg | L2正则化 | 1-10 | 默认3，增大防过拟合 |
| border_count | 数值特征分割数 | 32-254 | 默认254，高精度场景增大 |
| bagging_temperature | 贝叶斯采样温度 | 0-10 | 0=确定采样，>0=随机 |
| random_strength | 随机强度 | 0-10 | 防止过拟合 |
| rsm | 特征采样比例 | 0.6-1.0 | 减少过拟合 |
| loss_function | 损失函数 | RMSE / MAE / MAPE / Quantile | 与比赛评估指标一致 |
| eval_metric | 评估指标 | RMSE / MAE / MAPE | 与比赛评估指标一致 |
| task_type | 运行设备 | CPU / GPU | 数据量大用 GPU |

### 4.2 CatBoost 特殊优势

- **类别特征处理**：CatBoost 原生支持类别特征，不需要 one-hot
- **有序提升**：减少预测偏移，默认使用Ordered TS
- **对称树**：推理速度快，模型可解释性稍弱

### 4.3 典型竞赛配置

```python
cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.8,
    'random_strength': 1,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': 42,
    'verbose': 100
}
```

---

## 五、LSTM 调参

### 5.1 参数详解

| 参数 | 含义 | 常用范围 | 调参建议 |
|------|------|----------|----------|
| sequence_length | 输入序列长度 | 依据数据周期（7/14/24/168） | 需覆盖完整周期 |
| hidden_size | 隐层神经元数 | 32-256 | 数据量小→32-64，大→128-256 |
| num_layers | LSTM层数 | 1-4 | 通常1-2，复杂时序用2 |
| dropout | Dropout比例 | 0.1-0.5 | 防止过拟合，至少0.2 |
| recurrent_dropout | 循环层Dropout | 0-0.3 | 与 dropout 配合 |
| bidirectional | 是否双向 | True / False | 预测任务常用 True |
| learning_rate | 学习率 | 0.001-0.0001 | 常用 0.001，配合 EarlyStopping |
| batch_size | 批大小 | 16-128 | 数据量大用大值 |
| epochs | 训练轮数 | 50-500 | 配合 EarlyStopping |
| optimizer | 优化器 | Adam / RMSprop | Adam 最常用 |
| loss | 损失函数 | mse / mae / huber | 按评估指标选 |

### 5.2 时序预测典型配置

```python
lstm_params = {
    'sequence_length': 168,      # 1周数据预测下一天
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': True,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'optimizer': 'adam',
    'loss': 'mae',
    'patience': 15               # EarlyStopping patience
}
```

### 5.3 数据预处理要点

- **归一化**：必须对输入和输出都做 MinMaxScaler，预测后逆变换
- **序列构造**：用滑动窗口生成 (X, y) 对，步长=1
- **时序划分**：训练集在最前，验证集在中间，测试集在最后

---

## 六、Transformer / Informer 调参

### 6.1 Transformer 参数详解

| 参数 | 含义 | 常用范围 | 调参建议 |
|------|------|----------|----------|
| d_model | 模型维度 | 64-512 | 256 最常见 |
| n_heads | 注意力头数 | 4-16 | d_model 的因数 |
| num_encoder_layers | 编码器层数 | 2-6 | 2-4 足够 |
| num_decoder_layers | 解码器层数 | 2-6 | 通常等于或小于编码器 |
| d_ff | 前馈网络维度 | 512-2048 | 通常 4*d_model |
| dropout | Dropout比例 | 0.1-0.3 | 0.1 常用 |
| learning_rate | 学习率 | 0.0001-0.001 | 配合 warmup |
| batch_size | 批大小 | 16-64 | 数据量大用大值 |
| epochs | 训练轮数 | 50-300 | 配合 EarlyStopping |
| warmup_steps | 预热步数 | 1000-10000 | 通常总步数的 10% |

### 6.2 Informer（超长序列）特有参数

| 参数 | 含义 | 常用范围 |
|------|------|----------|
| seq_len | 输入序列长度 | 96-720（小时数据）|
| label_len | 解码器输入长度 | seq_len 的一半 |
| pred_len | 预测序列长度 | 24-168 |
| factor | ProbSparse 因子 | 1-5（越大越稀疏）|
| distil | 是否使用蒸馏 | True（训练快，效果略降）|
| e_layers | 编码器层数 | 2-4 |
| d_layers | 解码器层数 | 1-2 |

### 6.3 典型 Transformer 配置

```python
transformer_params = {
    'd_model': 256,
    'n_heads': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 2,
    'd_ff': 1024,
    'dropout': 0.1,
    'learning_rate': 0.0005,
    'warmup_steps': 1000,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15
}
```

---

## 七、STGCN（图时空卷积网络）调参

### 7.1 参数详解

| 参数 | 含义 | 常用范围 | 调参建议 |
|------|------|----------|----------|
| seq_len | 输入时间步 | 12-24 | 需覆盖完整周期 |
| pred_len | 预测时间步 | 12-24 | 通常等于 seq_len |
| batch_size | 批大小 | 8-32 | GPU显存限制，大图用小值 |
| learning_rate | 学习率 | 0.001-0.0001 | 配合 EarlyStopping |
| epochs | 训练轮数 | 100-500 | 配合 EarlyStopping |
| early_stop | 早停轮数 | 10-30 | 通常 15 |
| Ko | 一阶邻接矩阵时间核大小 | 3 | 常用 2-3 |
| Kt | 二阶邻接矩阵时间核大小 | 3 | 常用 3 |
| graph_kernel_size | 图卷积核大小 | 2-3 | 常用 2 |
| channels | 各层通道数 | [64, 64, 64, 128] 或类似 | 逐渐增加 |
| dropout | Dropout比例 | 0.1-0.5 | 防止过拟合 |

### 7.2 STGCN 特殊注意点

- **邻接矩阵构造**：基于空间距离或相关性阈值，需说明依据
- **数据归一化**：Z-score 归一化，逆变换后计算真实误差
- **时序数据 reshape**：从 (N, T, V) 转为 (T, N, V) 或 (N, T//seq_len, seq_len, V)
- **GPU 训练**：batch_size 大时需要 GPU，否则训练极慢

### 7.3 典型 STGCN 配置

```python
stgcn_params = {
    'seq_len': 12,
    'pred_len': 12,
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 200,
    'early_stop': 20,
    'Ko': 3,
    'Kt': 3,
    'graph_kernel_size': 2,
    'channels': [64, 64, 64, 128],
    'dropout': 0.2,
    'optimizer': 'adam',
    'loss': 'mae',
    'device': 'cuda'  # 或 'cpu'
}
```

---

## 八、贝叶斯优化（optuna）

### 8.1 安装与基本使用

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    # 交叉验证，返回评估指标
    ...

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f'最优参数: {study.best_params}')
print(f'最优值: {study.best_value}')
```

### 8.2 推荐搜索空间

| 模型 | 重点调参 | 搜索范围建议 |
|------|----------|-------------|
| XGBoost | learning_rate + n_estimators | 优先联合调，其他参数固定 |
| LightGBM | num_leaves + min_child_samples | 优先调结构参数 |
| CatBoost | depth + l2_leaf_reg | 深度最重要 |
| LSTM | hidden_size + dropout | 优先调容量参数 |
| Transformer | d_model + n_heads + num_layers | 通常 d_model=256, n_heads=8 不变 |

---

---

## 九、参数优先级与停机标准

### 9.1 各模型参数调参优先级

| 优先级 | XGBoost | LightGBM | LSTM | Transformer |
|--------|---------|----------|------|-------------|
| ⭐⭐⭐ 最重要 | `max_depth` + `n_estimators` | `num_leaves` + `n_estimators` | `hidden_size` + `seq_len` | `d_model` + `n_heads` |
| ⭐⭐ 重要 | `learning_rate` | `learning_rate` | `dropout` + `num_layers` | `num_layers` + `learning_rate` |
| ⭐ 一般 | `subsample` + `colsample` | `min_child_samples` | `batch_size` + `lr` | `dropout` + `d_ff` |
| 🔧 微调 | `reg_alpha` + `reg_lambda` | `reg_alpha` + `reg_lambda` | `recurrent_dropout` | `warmup_steps` |

> 调参顺序：**先固定最重要参数 → 再调次重要 → 最后微调**。
> 同时调太多参数无法判断哪个起作用。

### 9.2 停机标准（防止无限调参）

| 场景 | 停机条件 |
|------|----------|
| 网格/随机搜索 | 搜索 30+ 组参数后仍无明显改善 |
| 贝叶斯优化 | optuna 连续 10 轮无改善（`study.stop()`）|
| 模型训练 | 验证集指标连续 **15-20 轮无改善**（EarlyStopping）|
| 集成调参 | 3 个不同模型参数各调 2 轮后取最优 |
| **论文底线** | 每个问题至少跑 **3 个不同模型**（含 baseline）再选最优 |

### 9.3 调参记录模板

```python
# 记录每次调参结果，方便复盘和报告
tuning_log = []

def log_tuning(model_name, params, metric_name, metric_value):
    tuning_log.append({
        'model': model_name,
        'params': {k: round(v, 4) if isinstance(v, float) else v 
                   for k, v in params.items()},
        'metric': metric_name,
        'value': round(metric_value, 6),
        'timestamp': pd.Timestamp.now()
    })
    # 打印对比表
    df_log = pd.DataFrame(tuning_log).sort_values('value')
    print(df_log[['model', 'metric', 'value']].to_string(index=False))
    return df_log

# 使用示例
log_tuning('XGBoost_v1', {'max_depth': 6, 'lr': 0.05}, 'RMSE', 12.345)
log_tuning('XGBoost_v2', {'max_depth': 8, 'lr': 0.03}, 'RMSE', 11.891)
# → 选 v2，继续从 v2 参数出发微调
```

---

## 十、调参结果对比报告模板

> 论文中「参数敏感性分析」章节可直接使用以下格式。

```markdown
## 4.X 参数敏感性分析

### 4.X.1 参数设置对比

| 模型 | 关键参数 | RMSE | MAE | R² |
|------|----------|------|-----|-----|
| XGBoost (基准) | depth=6, lr=0.05, n_est=500 | 12.34 | 9.12 | 0.89 |
| XGBoost (深度↑) | depth=10, lr=0.03, n_est=800 | 11.78 | 8.65 | 0.91 |
| LightGBM | num_leaves=63, lr=0.05, n_est=800 | 11.52 | 8.43 | 0.92 |
| LSTM | hidden=128, seq=168, dropout=0.3 | 13.21 | 10.05 | 0.87 |
| 集成 (XGB+LGB) | 加权平均 | **10.87** | **7.92** | **0.93** |

### 4.X.2 最优参数组合

经 Optuna 贝叶斯优化 `{n_trials=50}` 搜索，得到最优参数：

| 参数 | 最优值 | 搜索范围 |
|------|--------|----------|
| learning_rate | 0.032 | [0.01, 0.3] |
| max_depth | 8 | [3, 12] |
| n_estimators | 1350 | [100, 2000] |
| subsample | 0.72 | [0.5, 1.0] |
| colsample_bytree | 0.64 | [0.5, 1.0] |
| min_child_weight | 4 | [1, 10] |

最终模型：RMSE = 10.87，MAE = 7.92，R² = 0.93
```

---

*本手册随技能更新持续补充。最新版本请参考 mathorcup-model/SKILL.md*
