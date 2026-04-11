# MathorCup 模型技能

> **定位：** 与 `mathorcup-paper` 技能互补——paper 管写作输出，model 管算法内核。
> **侧重点：** 模型选型策略、算法实现代码、参数调优流程、方案对比框架。

---

## 触发条件

当用户提到以下关键词时自动加载本技能：
- MathorCup、数学建模、建模思路、模型选择
- 算法实现、代码求解、参数调优
- 模型对比、方案对比、baseline
- 预测模型、优化模型、调度模型

---

## ⭐ 核心交互流程（B+C 模式）

### 第一阶段：路线规划（用户确认后深入）

当用户给出赛题时，按以下流程工作：

```
Step 1: 读取赛题描述
Step 2: 判定问题类型 → 参考 references/problem_routing.md §1
Step 3: 分析数据特征（用户提供数据路径则分析，否则询问）
Step 4: 输出多方案建模路线图（Markdown格式）
Step 5: 等待用户选择/确认方案
Step 6: 按用户选择的方案逐项深入
```

### Step 4 输出格式（路线图）

```markdown
## 【建模路线规划】

### 一、问题类型判定
- 问题一：xxx → [预测/优化/评价/聚类]
- 问题二：xxx → [预测/优化/评价/聚类]
- 问题三：xxx → [预测/优化/评价/聚类]
- 问题四：xxx → [预测/优化/评价/聚类]

### 二、数据特征分析
- 时间粒度：xx
- 数据规模：xxx
- 关键特征：xxx
- 特殊点：[预知数据/多节点/类别特征]（如有）

### 三、技术路线图

```
[数据] → [数据探查] → [预处理] → [特征工程]
                                    ↓
                    [问题一模型] → [问题二模型] → [问题三/四]
                                    ↓                    ↓
                              [求解优化]          [鲁棒性分析]
                                    ↓                    ↓
                              [结果评估] ←──────── [建模数据包]
```

### 四、候选方案（供用户选择）

**方案A：[首选方案名称]**
- 适用场景：xxx
- 核心模型：xxx
- 优势：xxx
- 预计效果：xxx

**方案B：[备选方案名称]**
- 适用场景：xxx
- 核心模型：xxx
- 优势：xxx

**方案C：[保守方案]**
- 适用场景：xxx
- 核心模型：xxx
- 优势：xxx（稳定，但效果可能不如A）

### 五、需用户确认
1. 数据文件路径（用于分析数据特征）
2. 选择哪个方案？（A/B/C/自定义）
3. 有无特殊要求？（比如必须用某模型/必须某指标最优）
```

### 第二阶段：逐项深入（用户选择后）

| 阶段 | 内容 | 输出文件 |
|------|------|----------|
| 数据分析 | 探查数据特征、缺失、异常、相关性 | data_findings.json |
| 模型选型 | 推荐模型 + 对比方案 + 选型理由 | model_decisions.json |
| 代码实现 | 生成完整代码（含注释）| code/*.py |
| 参数调优 | 执行调优，记录最优参数 | best_params.json |
| 结果评估 | 多模型对比表、评估指标 | comparison_table.csv |
| 可视化 | 生成所有图表 | figures/*.png |
| 打包输出 | 汇总为建模数据包 | output/summary.json |

---

## ⭐ 与 paper-skill 的协作

### 接口：建模数据包

model-skill 完成建模后，将结果写入标准目录：

```
~/.qclaw/workspace/model_output/
├── summary.json              # 建模方案摘要
├── data_findings.json        # 数据分析结果
├── model_decisions.json      # 模型选型决策
├── best_params.json          # 最优参数
├── results/
│   ├── comparison_table.csv  # 多模型对比
│   ├── metrics.json          # 评估指标
│   ├── predictions.csv       # 预测结果
│   └── optimization.json     # 优化结果
├── code/
│   ├── preprocessing.py     # 预处理
│   ├── model_xxx.py          # 各问题模型
│   └── main.py               # 主流程
└── figures/
    ├── data_exploration.png
    ├── model_comparison.png
    ├── error_analysis.png
    └── convergence.png       # （如有）
```

### paper-skill 如何消费

| paper 章节 | 从建模数据包读什么 |
|------------|-----------------|
| 摘要 | summary.json → 创新点 + 结果 |
| 数据探查 | data_findings.json → 探查文字 |
| 模型选择 | model_decisions.json → 选型理由 |
| 实验结果 | comparison_table.csv → 对比表格 |
| 评估指标 | metrics.json → 指标文字 |
| 代码附录 | code/*.py → 格式化代码 |

---

## 参考文档索引

| 文档 | 内容 | 何时使用 |
|------|------|----------|
| `references/problem_routing.md` | 问题类型判定树 + 选型表 | 收到赛题时 |
| `references/preprocess_guide.md` | 数据预处理全流程 | 数据分析阶段 |
| `references/prediction_guide.md` | 预测类完整路线 | 问题类型=预测时 |
| `references/optimization_guide.md` | 优化类完整路线 | 问题类型=优化时 |
| `references/evaluation_guide.md` | 评价/聚类/分类路线 | 问题类型=评价/聚类时 |
| `references/param_tuning.md` | 各模型调参手册 | 调参阶段 |
| `references/code_templates.md` | 可运行代码模板 | 实现代码时 |
| `references/scoring_tips.md` | 建模评分要点 | 确认方案时 |

---

## 评分标准提醒（建模阶段必须做到）

| 评分项 | 必须做到 |
|--------|----------|
| 数据预处理 | ✅ 缺失值处理 ✅ 异常值处理 ✅ 可视化探查 |
| 多模型对比 | ✅ 至少3个模型 ✅ 含baseline ✅ 表格输出 |
| 调参有记录 | ✅ 调参策略说明 ✅ 最优参数记录 |
| 预知数据 | ✅ 残差预测法/特征构造法 ✅ 效果对比 |
| 消融实验 | ✅ 分模块对比 ✅ 量化贡献 |
| 鲁棒性分析 | ✅ 敏感性分析 或 ✅ 分位数预测区间 |

---

*技能版本：v1.0 正式版*
