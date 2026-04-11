# 问题类型判定与模型路由

> 本文档是 mathorcup-model 技能的决策入口。
> 当用户描述一道赛题时，先用本文件判定问题类型，再路由到对应指南。

---

## 一、问题类型判定树

```
收到赛题描述
    │
    ├─ 关键词: 预测/估计/推断/货量/流量/需求/销量
    │    └─ → 【预测类问题】
    │          ├─ 单时间序列 → prediction_guide.md §3
    │          ├─ 多节点时空数据 → prediction_guide.md §3.2（图神经网络）
    │          ├─ 有预知数据 → prediction_guide.md §4（残差预测法）
    │          └─ 分类预测 → evaluation_guide.md §3
    │
    ├─ 关键词: 调度/分配/安排/路径/成本最小/时间最短/车辆
    │    └─ → 【优化类问题】
    │          ├─ 数学规划（线性/整数）→ optimization_guide.md §2
    │          ├─ TSP/VRP/路径规划 → optimization_guide.md §3
    │          ├─ 启发式/元启发式 → optimization_guide.md §4
    │          └─ 多目标 → optimization_guide.md §6
    │
    ├─ 关键词: 评价/评估/排序/分级/综合实力/竞争力
    │    └─ → 【评价类问题】
    │          ├─ 指标体系 + 权重 → evaluation_guide.md §1
    │          └─ TOPSIS/熵权法 → evaluation_guide.md §1.3
    │
    └─ 关键词: 聚类/分群/发现/异常检测
         └─ → 【聚类类问题】
               └─ evaluation_guide.md §2
```

---

## 二、模型选型对照表（按数据特征）

### 2.1 预测类

| 数据特征 | 首选模型 | 对比模型 | 备选 |
|----------|----------|----------|------|
| 数据量 < 1K | 线性回归、ARIMA | Ridge/Lasso | XGBoost |
| 数据量 1K-50K | LightGBM、XGBoost | RandomForest | CatBoost |
| 数据量 > 50K | LightGBM、CatBoost | 深度学习 | 模型融合 |
| 强时间依赖 | LSTM | Transformer | Informer |
| 多节点时空 | STGCN | ASTGCN | DCRNN |
| 强周期依赖 | Prophet | Transformer | 周期分解+预测 |
| 有预知数据 | 残差预测法（LightGBM + Ridge）| 特征构造法 | 两阶段模型 |
| 类别特征丰富 | CatBoost | LightGBM | XGBoost |

### 2.2 优化类

| 问题类型 | 首选方法 | 对比方法 | 备选 |
|----------|----------|----------|------|
| 线性规划 | scipy.optimize.linprog | pulp | Gurobi |
| 整数/混合整数 | scipy.optimize.milp | pulp | — |
| TSP/VRP（大规模）| 贪心 + 2-opt | 遗传算法 | 蚁群算法 |
| 复杂组合优化 | 遗传算法 | 粒子群 | 模拟退火 |
| 多目标 | 加权求和法 | NSGA-II | MOEA/D |
| 含不确定性的优化 | 鲁棒优化 | 随机规划 | 仿真优化 |

### 2.3 评价类

| 场景 | 首选方法 | 权重方法 | 评估模型 |
|------|----------|----------|----------|
| 权重未知 | AHP | 熵权法 | TOPSIS |
| 数据驱动 | CRITIC | 熵权法 | 灰色关联 |
| 指标过多 | PCA | — | 主成分评价 |
| 定性+定量混合 | 模糊评价 | AHP | TOPSIS |

### 2.4 聚类类

| 数据特征 | 首选算法 | 备选 | 评估方法 |
|----------|----------|------|----------|
| 球形簇、大规模 | K-means | MiniBatchKMeans | 轮廓系数 |
| 任意形状、有噪声 | DBSCAN | HDBSCAN | CH指数 |
| 需要层次结构 | 层次聚类 | BIRCH | 树状图 |
| 非凸簇 | 谱聚类 | GMM | 轮廓系数 |

---

## 三、路由决策示例

### 示例1：2025 MathorCup D题（物流配送优化）

```
关键词：预测、车辆调度、预知数据、鲁棒性
→ 问题一：预测类（有预知数据）
  → prediction_guide.md §4（残差预测法）
→ 问题二：优化类（车辆调度）
  → optimization_guide.md §3（VRP + 贪心 + GA）
→ 问题三：评价/对比类
  → optimization_guide.md §7（消融实验）
→ 问题四：鲁棒性分析
  → optimization_guide.md §5 + scoring_tips.md §4
```

### 示例2：2024 MathorCup B题（评价类）

```
关键词：评价、排序、指标体系、综合实力
→ 评价类问题
  → evaluation_guide.md §1（指标体系 + AHP + TOPSIS）
→ 含时间维度 → 加时序预测
  → prediction_guide.md §3
```

### 示例3：交通流量预测（多节点时空）

```
关键词：交通流量、时空数据、多路段、预测
→ 预测类 + 多节点时空数据
  → prediction_guide.md §3.2（图神经网络）
  → STGCN/ASTGCN优先
```

---

## 四、全量优秀论文模型使用统计（2020-2024年，共41篇）

> 以下数据从 MathorCup 大数据挑战赛 2020-2024 年共 41 篇优秀论文中提取。
> 知识库原始文件：`C:\Users\sss\.qclaw\workspace\paper_analysis\full_knowledge_base_clean.json`

### 总体高频模型排名（5年汇总）

| 排名 | 模型 | 使用频率 | 备注 |
|------|------|---------|------|
| 🥇 | 随机森林 | ~18/41 | 各年份预测标配 |
| 🥈 | XGBoost/XGB | ~13/41 | 2019年后快速崛起 |
| 🥉 | LSTM | ~9/41 | 时序预测首选 |
| 4 | 机器学习 | ~9/41 | 泛指，含义多样 |
| 5 | LightGBM | ~8/41 | 大规模数据高效 |
| 6 | 特征工程 | ~7/41 | 含特征选择/构造 |
| 7 | Stacking | ~6/41 | 集成学习首选 |
| 8 | ARIMA | ~6/41 | 传统时序 |
| 9 | 时间序列 | ~5/41 | 泛指 |
| 10 | 神经网络 | ~5/41 | 通用 |

### 各年度特征

| 年份 | 篇数 | 主流模型 | 趋势 |
|------|------|---------|------|
| 2020 | 9 | 时间序列(3)、随机森林(2)、LSTM(2)、U-Net(2) | 深度学习刚起步，U-Net用于图像分割 |
| 2021 | 8 | XGB/XGBoost(5)、随机森林(5)、LightGBM(4)、CatBoost(3) | **GBDT时代**，Stacking集成成主流 |
| 2022 | 8 | 随机森林(5)、XGB/XGBoost(4)、特征工程(4)、神经网络(3) | 特征工程+GBDT成熟期 |
| 2023 | 8 | ARIMA(4)、机器学习(3)、PCA(2)、注意力机制(2) | 回归传统统计方法，注意力机制开始流行 |
| 2024 | 8 | 神经网络(5)、LSTM(5)、随机森林(5)、DTW(3) | **LSTM复兴**，时空预测+注意力成为标配 |

### 高频组合模式（历年不变）

- **时序预测首选**：LSTM > ARIMA > 随机森林 > XGBoost
- **聚类分析首选**：K-Means（95%论文使用） > 层次聚类 > DBSCAN
- **路径/相似度评估**：DTW（动态时间规整）几乎是必备
- **优化求解**：整数规划/LP → 元启发式（模拟退火/粒子群/NSGA-II）
- **集成策略**：Stacking > Voting > 单模型

### 全文扫描统计（2020-2024，41篇优秀论文）

> 数据来源：论文全文扫描（非仅摘要）
> 原始数据：`paper_analysis/deep_scan_v2/final_stats.json`

**论文基本信息：**
- 平均页数：37.4 页
- 平均字数：37,408 字符
- 平均参考文献：9.2 篇

**模型/方法频率（Top 15）：**

| 排名 | 模型/方法 | 频率 | 说明 |
|------|----------|------|------|
| 🥇 | 集成学习 | 49次(120%) | Stacking/Bagging/Boosting |
| 🥈 | 神经网络 | 44次(107%) | 含BP、CNN、RNN等 |
| 🥉 | K-Means | 39次(95%) | 聚类首选 |
| 4 | **模拟退火** | 37次(90%) | 优化问题首选！ |
| 5 | LSTM/GRU/RNN | 33次(80%) | 时序预测首选 |
| 6 | XGBoost | 32次(78%) | 表格数据首选 |
| 7 | 随机森林 | 29次(71%) | 高频基模型 |
| 8 | 决策树 | 29次(71%) | 可解释性强 |
| 9 | 遗传算法 | 24次(59%) | 多目标优化 |
| 10 | 线性规划 | 23次(56%) | 运筹优化基础 |
| 11 | PCA/主成分 | 18次(44%) | 降维首选 |
| 12 | 时间序列 | 16次(39%) | 含ARIMA/SARIMA |
| 13 | 支持向量机 | 16次(39%) | 小样本分类 |
| 14 | 注意力/Transformer | 13次(32%) | 新兴趋势 |
| 15 | DTW | 11次(27%) | 路径相似度 |

**深度学习 vs 传统机器学习：**
- 深度学习类（LSTM/GRU/NN/注意力）：90次
- 传统ML类（RF/XGB/LGB/决策树）：101次
- **传统机器学习略占优势**，但深度学习增长迅速

**任务类型分布：**
- 分类任务：39/41 (95%)
- 回归任务：28/41 (68%)
- 聚类任务：18/41 (44%)
- 时间序列：16/41 (39%)

### 赛道规律（历年稳定）

- **赛道A（预测/分类）**：聚类分析 → 深度学习/GBDT预测 → 路径相似度评估（DTW）
- **赛道B（优化/调度）**：多模型对比选最优 → 整数规划/LP → 元启发式优化

### 知识库文件索引

| 年份 | 摘要原文 | 知识库 |
|------|---------|--------|
| 2020 | `paper_analysis/2020/` | `full_knowledge_base_clean.json` |
| 2021 | `paper_analysis/2021/` | 同上 |
| 2022 | `paper_analysis/2022/` | 同上 |
| 2023 | `paper_analysis/2023/` | 同上 |
| 2024 | `paper_analysis/2024/` | 同上 |

---

## 五、建模数据包标准结构

当完成建模后，输出以下结构供 paper-skill 消费：

```
output/
├── summary.json              # 建模摘要
│   ├── problem_type          # 问题类型
│   ├── model_decisions       # 各问模型选择及理由
│   ├── data_findings         # 数据发现
│   └── innovation_points     # 创新点
│
├── results/
│   ├── comparison_table.csv  # 多模型对比表
│   ├── metrics.json          # 评估指标
│   ├── predictions.csv       # 预测结果（如有）
│   └── optimization.json     # 优化结果（如有）
│
├── code/
│   ├── preprocessing.py      # 预处理
│   ├── model_xxx.py          # 各问题模型
│   ├── evaluation.py         # 评估
│   └── main.py               # 主流程
│
└── figures/
    ├── data_exploration.png   # 数据探查
    ├── model_comparison.png  # 模型对比
    ├── error_analysis.png    # 误差分析
    ├── convergence.png       # 收敛曲线（如有）
    └── pareto_front.png       # 帕累托（如有）
```
