# 优化类问题建模指南

> 适用问题类型：路径规划、资源调度、成本优化、分配问题
> 本指南包含：问题建模 → 算法选择 → 实现细节 → 鲁棒性分析
>
> ⚠️ **历年优秀论文统计**：模拟退火在 37/41 篇论文中出现（90%），
> 是优化问题的首选元启发式算法！线性规划（56%）和遗传算法（59%）紧随其后。

---

## 一、问题识别

### 1.1 关键词判定

| 关键词 | 问题类型 | 典型赛题 |
|--------|----------|----------|
| 调度、分配、安排、指派 | 资源调度/排程 | 车辆调度、人员排班 |
| 成本最小、时间最短、效率最高 | 单目标优化 | 物流成本、路径规划 |
| 路径规划、旅行商 | TSP/VRP | 配送路线优化 |
| 在满足...条件下 | 约束优化 | 含多种约束的调度问题 |
| 多目标、成本+时间+风险 | 多目标优化 | 综合决策 |

---

## 二、数学建模标准流程

### 2.1 四要素建模模板

```
┌─────────────────────────────────────────┐
│ Step 1: 定义决策变量                      │
│   · 连续变量：x_i ∈ [0, U_i]              │
│   · 整数变量：y_j ∈ {0,1,2,...}           │
│   · 0-1变量：z_k ∈ {0,1}                 │
│   · 向量/矩阵变量：X = [x_ij]             │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 2: 建立目标函数                      │
│   · 单目标：min/max f(x)                  │
│   · 多目标：f₁(x) + λf₂(x) 或帕累托前沿   │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 3: 列出约束条件                      │
│   · 资源约束：∑a_ij x_j ≤ B_i            │
│   · 需求约束：∑x_j ≥ D                   │
│   · 时间约束：t_k - t_i ≥ dur_ik          │
│   · 逻辑约束：z_k ≤ ∑x_ij                 │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 4: 问题复杂度分析                     │
│   · 规模估计（变量数/约束数）              │
│   · NP-hard判断（TSP/VRP→是）             │
│   · 求解策略选择依据                       │
└─────────────────────────────────────────┘
```

### 2.2 常用数学规划模板

**资源配置优化：**
```
min  ∑c_i x_i
s.t. ∑a_ij x_i ≤ B_j    (资源约束)
     x_i ≥ 0             (非负约束)
     x_i ≤ U_i           (能力上限)
```

**指派问题：**
```
min  ∑∑c_ij x_ij
s.t. ∑x_ij = 1  ∀j       (每任务恰好一人)
     ∑x_ij = 1  ∀i       (每人恰好一任务)
     x_ij ∈ {0,1}
```

**车辆路径问题（VRP）：**
```
min  ∑∑∑d_ij x_ijk
s.t. ∑y_ik = 1        ∀k (每个顾客被服务一次)
     ∑x_ijk = y_ik    ∀i,k (路径定义)
     ∑x_ijk - ∑x_jik = 0 ∀k,j (流平衡)
     ∑q_i y_ik ≤ Q    ∀k   (容量约束)
     x_ijk, y_ik ∈ {0,1}
```

---

## 三、算法选择框架

### 3.1 问题规模与算法对照

| 问题规模 | 变量数 | 推荐算法 |
|----------|--------|----------|
| 小规模 | < 1000 | 精确算法（分支定界、割平面、内点法）|
| 中等规模 | 1000-10万 | 启发式算法（贪心+局部搜索）|
| 大规模 | > 10万 | 元启发式算法（GA/SA/PSO/ACO）|
| 超大规模+时间紧 | — | 贪心 + 规则 + 并行计算 |

### 3.2 元启发式算法参数参考

**遗传算法（GA）：**

| 参数 | 常用值 | 说明 |
|------|--------|------|
| 种群大小 | 50-200 | 问题规模越大，种群越大 |
| 交叉概率 | 0.6-0.9 | 常用 0.8 |
| 变异概率 | 0.01-0.1 | 常用 0.05 |
| 最大迭代次数 | 200-1000 | 配合早停 |
| 精英保留比例 | 0.05-0.1 | 保留最优个体 |

**模拟退火（SA）：**

| 参数 | 常用值 | 说明 |
|------|--------|------|
| 初始温度 | 1000-10000 | 高温开始 |
| 冷却率 | 0.95-0.99 | 常用 0.97 |
| 邻域大小 | 问题相关 | 每个温度下探索次数 |
| 终止温度 | 0.001-1 | 温度降到此值停止 |

**粒子群优化（PSO）：**

| 参数 | 常用值 | 说明 |
|------|--------|------|
| 种群大小 | 20-50 | 常用 30 |
| 认知系数 c1 | 1.5-2.5 | 常用 2.0 |
| 社会系数 c2 | 1.5-2.5 | 常用 2.0 |
| 惯性权重 w | 0.4-0.9 | 常用 0.7 或线性递减 |

---

## 四、启发式算法实现框架

### 4.1 遗传算法完整实现

```python
import numpy as np
import random

class GeneticOptimizer:
    def __init__(self, n_vars, n_pop=100, n_iter=500, 
                 pc=0.8, pm=0.05, elite_ratio=0.1,
                 lb=None, ub=None):
        self.n_vars = n_vars
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率
        self.elite_ratio = elite_ratio
        self.lb = lb if lb is not None else np.zeros(n_vars)
        self.ub = ub if ub is not None else np.ones(n_vars)
        self.best_x = None
        self.best_f = np.inf
        self.history = []
    
    def initial_population(self):
        """初始解生成（贪心+随机混合）"""
        pop = []
        # 60%贪心解
        for _ in range(int(self.n_pop * 0.6)):
            ind = self._greedy_solution()
            pop.append(ind)
        # 40%随机解
        for _ in range(self.n_pop - len(pop)):
            ind = np.random.uniform(self.lb, self.ub)
            pop.append(ind)
        return np.array(pop)
    
    def _greedy_solution(self):
        """贪心初始解（根据问题定制）"""
        # 示例：物流配送贪心解：按距离最近优先分配
        x = np.zeros(self.n_vars)
        # TODO: 根据具体问题实现贪心规则
        return x
    
    def evaluate(self, pop):
        """评估种群"""
        fitness = np.array([self._fitness(ind) for ind in pop])
        return fitness
    
    def _fitness(self, x):
        """目标函数（根据问题定制）"""
        # TODO: 返回目标函数值
        return np.sum(x**2)  # 示例：Sphere函数
    
    def selection(self, pop, fitness):
        """锦标赛选择"""
        selected = []
        tournament_size = 3
        for _ in range(len(pop)):
            idx = random.sample(range(len(pop)), tournament_size)
            winner = idx[np.argmin(fitness[idx])]
            selected.append(pop[winner])
        return np.array(selected)
    
    def crossover(self, pop):
        """单点交叉"""
        new_pop = []
        for i in range(0, len(pop), 2):
            if i+1 >= len(pop):
                new_pop.append(pop[i])
                break
            p1, p2 = pop[i], pop[i+1]
            if random.random() < self.pc:
                pt = random.randint(1, self.n_vars-1)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
                new_pop.extend([c1, c2])
            else:
                new_pop.extend([p1, p2])
        return np.array(new_pop[:self.n_pop])
    
    def mutation(self, pop):
        """高斯变异"""
        for i in range(len(pop)):
            if random.random() < self.pm:
                sigma = 0.1 * (self.ub - self.lb)
                pop[i] = np.clip(
                    pop[i] + np.random.normal(0, sigma),
                    self.lb, self.ub
                )
        return pop
    
    def elitism(self, old_pop, new_pop, old_fitness, new_fitness):
        """精英保留"""
        n_elite = int(self.n_pop * self.elite_ratio)
        elite_idx = np.argsort(old_fitness)[:n_elite]
        elite = old_pop[elite_idx]
        elite_f = old_fitness[elite_idx]
        
        combined_pop = np.vstack([new_pop, elite])
        combined_f = np.concatenate([new_fitness, elite_f])
        
        top_idx = np.argsort(combined_f)[:self.n_pop]
        return combined_pop[top_idx], combined_f[top_idx]
    
    def solve(self):
        """主循环"""
        pop = self.initial_population()
        fitness = self.evaluate(pop)
        
        for gen in range(self.n_iter):
            # 选择
            selected = self.selection(pop, fitness)
            # 交叉
            crossed = self.crossover(selected)
            # 变异
            mutated = self.mutation(crossed)
            # 评估
            new_fitness = self.evaluate(mutated)
            # 精英保留
            pop, fitness = self.elitism(pop, mutated, fitness, new_fitness)
            
            # 记录
            gen_best_idx = np.argmin(fitness)
            gen_best_f = fitness[gen_best_idx]
            self.history.append(gen_best_f)
            
            if gen_best_f < self.best_f:
                self.best_f = gen_best_f
                self.best_x = pop[gen_best_idx]
            
            if gen % 50 == 0:
                print(f'Gen {gen}: Best = {self.best_f:.4f}')
            
            # 早停：连续50代无改进
            if len(self.history) > 50 and \
               max(self.history[-50:]) - min(self.history[-50:]) < 1e-6:
                print(f'早停于第 {gen} 代')
                break
        
        return self.best_x, self.best_f
    
    def plot_convergence(self):
        """收敛曲线"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.history, 'b-', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('GA Convergence Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/ga_convergence.png', dpi=150)
        plt.close()
```

### 4.2 邻域操作详解（VRP示例）

```python
class VRPOperators:
    """车辆路径问题的邻域操作"""
    
    @staticmethod
    def swap(route, i, j):
        """交换：交换两个位置"""
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route
    
    @staticmethod
    def insert(route, i, j):
        """插入：将位置i的元素插入到位置j之后"""
        new_route = route.copy()
        node = new_route.pop(i)
        new_route.insert(j if j > i else j, node)
        return new_route
    
    @staticmethod
    def reverse(route, i, j):
        """逆转（2-opt）：反转[i,j]区间"""
        new_route = route.copy()
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route
    
    @staticmethod
    def relocate(route1, route2, i, j):
        """转移：将route1中的节点i转移到route2的位置j"""
        new_route1 = route1.copy()
        new_route2 = route2.copy()
        node = new_route1.pop(i)
        new_route2.insert(j, node)
        return new_route1, new_route2
    
    @staticmethod
    def cross_exchange(route1, route2, i, j):
        """交叉交换（or-opt）：交换两个节点"""
        new_route1 = route1.copy()
        new_route2 = route2.copy()
        new_route1[i], new_route2[j] = new_route2[j], new_route1[i]
        return new_route1, new_route2
```

---

## 五、鲁棒性分析

### 5.1 敏感性分析

```python
def sensitivity_analysis(base_solution, base_params, 
                        param_name, param_range):
    """
    敏感性分析：关键参数扰动对结果的影响
    """
    results = []
    for val in param_range:
        # 修改参数
        params = base_params.copy()
        params[param_name] = val
        # 重新求解
        solution = solve_optimization(params)
        results.append({
            'param_value': val,
            'objective': solution['objective'],
            'solve_time': solution['time']
        })
    
    df = pd.DataFrame(results)
    
    # 绘制敏感性曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['param_value'], df['objective'], 'bo-')
    plt.xlabel(param_name)
    plt.ylabel('Objective Value')
    plt.title(f'Sensitivity: {param_name} vs Objective')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df['param_value'], df['solve_time'], 'rs-')
    plt.xlabel(param_name)
    plt.ylabel('Solve Time (s)')
    plt.title(f'Sensitivity: {param_name} vs Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/sensitivity_{param_name}.png', dpi=150)
    plt.close()
    
    return df

# 使用示例：分析车辆容量对总成本的影响
param_ranges = {
    'vehicle_capacity': [500, 750, 1000, 1250, 1500],
    'num_vehicles': [5, 6, 7, 8, 9, 10],
    'max_time': [180, 240, 300, 360]
}
for param, values in param_ranges.items():
    sensitivity_analysis(base_solution, base_params, param, values)
```

### 5.2 Monte Carlo 鲁棒性分析

```python
def monte_carlo_robustness(n_simulations=1000):
    """
    Monte Carlo鲁棒性分析：参数服从一定分布时解的稳定性
    """
    results = []
    for _ in range(n_simulations):
        # 随机扰动参数
        perturbed_params = {
            'demand_scale': base_params['demand_scale'] * np.random.uniform(0.8, 1.2),
            'cost_coef': base_params['cost_coef'] * np.random.uniform(0.9, 1.1),
            'time_limit': base_params['time_limit'] * np.random.uniform(0.95, 1.05),
        }
        solution = solve_optimization(perturbed_params)
        results.append(solution['objective'])
    
    results = np.array(results)
    
    print(f'Monte Carlo Results ({n_simulations} simulations):')
    print(f'  Mean: {results.mean():.4f}')
    print(f'  Std: {results.std():.4f}')
    print(f'  95% CI: [{np.percentile(results, 2.5):.4f}, {np.percentile(results, 97.5):.4f}]')
    print(f'  Robustness: {results.std() / results.mean() * 100:.1f}% CV')
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(results, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(results.mean(), color='red', label=f'Mean={results.mean():.2f}')
    plt.xlabel('Objective Value')
    plt.ylabel('Frequency')
    plt.title('Monte Carlo Robustness Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot(results)
    plt.ylabel('Objective Value')
    plt.title('Solution Robustness Boxplot')
    
    plt.tight_layout()
    plt.savefig('figures/robustness_analysis.png', dpi=150)
    plt.close()
```

---

## 六、多目标优化

### 6.1 帕累托前沿计算

```python
from scipy.optimize import minimize
import numpy as np

def weighted_sum_method(obj1_func, obj2_func, n_weights=20):
    """
    加权求和法计算帕累托前沿
    """
    pareto_front = []
    pareto_solutions = []
    
    for w1 in np.linspace(0, 1, n_weights):
        w2 = 1 - w1
        def combined(x):
            return w1 * obj1_func(x) + w2 * obj2_func(x)
        
        # 多次随机起点
        best_x, best_f = None, np.inf
        for _ in range(5):
            x0 = np.random.uniform(0, 10, n_vars)
            result = minimize(combined, x0, method='SLSQP')
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x
        
        pareto_front.append([obj1_func(best_x), obj2_func(best_x)])
        pareto_solutions.append(best_x)
    
    return np.array(pareto_front), pareto_solutions

def plot_pareto_front(pareto_front, all_points=None):
    """绘制帕累托前沿"""
    plt.figure(figsize=(10, 6))
    if all_points is not None:
        plt.scatter(all_points[:,0], all_points[:,1], 
                   c='gray', alpha=0.3, label='All Solutions')
    plt.scatter(pareto_front[:,0], pareto_front[:,1], 
               c='red', s=100, label='Pareto Front', zorder=5)
    plt.xlabel('Objective 1 (e.g., Cost)')
    plt.ylabel('Objective 2 (e.g., Time)')
    plt.title('Pareto Front')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/pareto_front.png', dpi=150)
    plt.close()
```

---

## 七、消融实验设计

### 7.1 消融实验框架

```python
def ablation_study():
    """
    消融实验：验证每个模块的贡献
    """
    results = []
    
    # 实验1：纯贪心baseline
    baseline = greedy_solve()
    results.append({'Method': 'Greedy Baseline', **evaluate(baseline)})
    
    # 实验2：贪心 + 局部搜索
    ls = local_search(greedy_solve())
    results.append({'Method': 'Greedy + LS', **evaluate(ls)})
    
    # 实验3：贪心 + 局部搜索 + 2-opt
    two_opt = two_opt_improve(ls)
    results.append({'Method': 'Greedy + LS + 2-opt', **evaluate(two_opt)})
    
    # 实验4：GA（完整算法）
    ga = GeneticOptimizer(n_vars).solve()
    results.append({'Method': 'GA (Full)', **evaluate(ga)})
    
    # 实验5：GA + 2-opt后处理
    ga_2opt = two_opt_improve(ga)
    results.append({'Method': 'GA + 2-opt', **evaluate(ga_2opt)})
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # 柱状图对比
    plt.figure(figsize=(12, 5))
    metrics = ['Cost', 'Time', 'Feasibility']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.bar(df['Method'], df[metric])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{metric} Comparison')
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/ablation_study.png', dpi=150)
    plt.close()
    
    return df
```

---

*本指南参考 param_tuning.md（调参手册）和 scoring_tips.md（评分要点）*
