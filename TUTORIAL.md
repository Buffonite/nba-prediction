# NBA 预测项目 — 学习教材

这份教材带你走一遍这个项目是**怎么一步步搭出来的**，以及跑完之后**输出的数字和图代表什么**。

读完之后，你应该能：
1. 自己解释每个文件的作用和设计原因（面试常考）
2. 看懂训练日志里的每一行
3. 解读 5 张评估图，知道模型好坏在哪

---

## 目录

- [Part 0: 全局思路](#part-0-全局思路)
- [Part 1: 数据获取（data_fetch.py）](#part-1-数据获取)
- [Part 2: 特征工程（preprocessing.py）⭐ 最重要](#part-2-特征工程)
- [Part 3: 神经网络模型（model.py）](#part-3-神经网络模型)
- [Part 4: 训练流程（train.py）](#part-4-训练流程)
- [Part 5: 评估与可视化（evaluate.py）](#part-5-评估与可视化)
- [Part 6: 如何阅读输出](#part-6-如何阅读输出)
- [Part 7: 面试常问问题](#part-7-面试常问问题)
- [Part 8: 高级特征 — ELO + 伤病](#part-8-高级特征--elo--伤病)
- [Part 9: 深度专题](#part-9-深度专题)
  - [9.1 反向传播 — 神经网络是怎么"学"的](#91-反向传播--神经网络是怎么学的)
  - [9.2 梯度消失 / 爆炸 — 为什么 ReLU 比 sigmoid 好](#92-梯度消失--爆炸--为什么-relu-比-sigmoid-好)
  - [9.3 过拟合诊断 — 看训练曲线判断模型状态](#93-过拟合诊断--看训练曲线判断模型状态)
  - [9.4 Adam 内部机制 — 一阶矩 / 二阶矩 / 偏差修正](#94-adam-内部机制--一阶矩--二阶矩--偏差修正)
  - [9.5 替代架构 — 残差连接、宽 vs 深](#95-替代架构--残差连接宽-vs-深)
- [Part 10: 季后赛 Bracket 模拟器](#part-10-季后赛-bracket-模拟器)
- [Part 11: Basketball Reference 备用数据源](#part-11-basketball-reference-备用数据源)

---

## Part 0: 全局思路

### 我们在解决什么问题？

> **给定一场 NBA 比赛的两支球队和当前情境，预测主队是否获胜。**

这是一个**二分类问题**（binary classification）：
- 输出是 0 到 1 之间的概率
- 大于 0.5 → 预测主队赢
- 小于 0.5 → 预测主队输

### 为什么这是个好的入门项目？

| 优点 | 说明 |
|---|---|
| 数据免费 | NBA 官方数据 API 不需要 key |
| 标签明确 | 比赛结果是确定的 0/1，没有歧义 |
| 有基准 | NBA 主队胜率历史约 59%，模型必须超过这个数才有用 |
| 体量适中 | 几千场比赛 ≈ 神经网络刚好能学的样本量 |

### 整体流程

```
[原始 API 数据] 
      ↓ data_fetch.py        (下载、整理成"每场一行"的表)
[每场比赛一行的表]
      ↓ preprocessing.py     (生成滚动特征：每队最近 N 场的统计)
[特征矩阵 X + 标签 y]
      ↓ train.py             (按时间切分 → 标准化 → 训练 NN + 基准)
[训练好的模型]
      ↓ evaluate.py          (指标表 + 5 张图)
[结果展示]
```

---

## Part 1: 数据获取

📄 文件：[src/data_fetch.py](src/data_fetch.py)

### 为什么用 nba_api？

`nba_api` 是一个 Python 库，直接和 stats.nba.com 通信。**免费、无需注册**。

```python
from nba_api.stats.endpoints import leaguegamefinder
df = leaguegamefinder.LeagueGameFinder(season_nullable="2022-23").get_data_frames()[0]
```

### 关键设计：从"每队一行"变成"每场一行"

NBA API 返回的原始数据是**每场比赛两行**（主队一行，客队一行）：

| GAME_ID | TEAM | MATCHUP | PTS | WL |
|---|---|---|---|---|
| 001 | LAL | LAL vs. GSW | 110 | W |
| 001 | GSW | GSW @ LAL | 105 | L |

我们需要的是**每场比赛一行**：

| GAME_ID | home_team | away_team | home_pts | away_pts | home_win |
|---|---|---|---|---|---|
| 001 | LAL | GSW | 110 | 105 | 1 |

`build_game_table()` 函数干的就是这个 pivot 操作。**关键技巧**：
- `MATCHUP` 字段里 `"vs."` 表示主场，`"@"` 表示客场 → 用来区分主客
- 用 `merge()` 按 `GAME_ID` 把两行合成一行

### 缓存策略

代码会先检查 `data/raw/games.csv` 是否存在 — 如果存在就直接读文件，不重新调用 API。这样：
- 第一次运行下载（要几分钟）
- 之后运行秒级启动

### 输出
- 文件：`data/raw/games.csv`
- 列数：约 9 列
- 行数：约 3,600 场（3 个赛季）

---

## Part 2: 特征工程

📄 文件：[src/preprocessing.py](src/preprocessing.py)

> ⭐ **这是整个项目最重要的一步。** 模型的好坏 80% 取决于特征质量。

### 核心问题：**不能用未来数据**

假设我们要预测 2024-01-15 湖人 vs 勇士。能用的信息：
- ✅ 两队 1 月 15 日**之前**所有比赛的统计
- ❌ 这场比赛本身的得分、命中率（这就是答案，用了等于作弊）
- ❌ 1 月 16 日之后的任何数据

这叫**防止数据泄露（data leakage）**。新手最常犯的错误就是这里。

### 解决方案：滚动平均 + shift(1)

```python
df.groupby("team_id")["pts_scored"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).mean()
)
```

这行代码的含义：
1. `.groupby("team_id")` — 按球队分组（每支队各算各的）
2. `.shift(1)` — **关键！** 把数据向后挪一格 → 当前比赛被排除
3. `.rolling(5)` — 取窗口大小为 5 的滚动窗口
4. `.mean()` — 求平均

**举例**：湖人队的得分序列是 `[110, 105, 98, 115, 102, 108]`
- 不用 shift：第 6 场的 `last5` = mean(105, 98, 115, 102, 108) ← **包含第6场之前的5场**，✅
- 用 shift(1)：完全等价，明确表达"不含当前场"

实际场景中 `.shift(1)` 在某些边界情况（比如做 `last1` 即上一场）就必不可少。养成习惯总是用 `.shift(1)`。

### 我们造的特征

对**主队**和**客队**都计算：

| 特征 | 含义 | 为什么重要 |
|---|---|---|
| `win_pct_last5` | 最近 5 场胜率 | 当前状态（最近表现） |
| `win_pct_last10` | 最近 10 场胜率 | 中期状态（更稳定） |
| `pts_scored_last5` | 最近 5 场场均得分 | 进攻强度 |
| `pts_allowed_last5` | 最近 5 场场均失分 | 防守强度 |
| `net_rating_last5` | 得分 - 失分 | **综合实力指标** |
| `rest_days` | 距离上一场的天数 | 体力 |
| `is_b2b` | 是否背靠背（连续两天比赛） | 累积疲劳 |

### 差值特征：让模型直接看到"谁更强"

```python
features["diff_net_rating_last5"] = features["home_net_rating_last5"] - features["away_net_rating_last5"]
```

为什么有用？
- 模型其实可以自己学到 `home - away` 的关系，但**显式提供更容易学**
- 这是机器学习的一个原则：**好特征 > 复杂模型**

### 最终特征矩阵

- 行数：约 3,500（去掉每队最早几场没有历史数据的）
- 列数：约 28（主队 7 + 客队 7 + 差值 6 × 2 个窗口）
- 标签：`home_win` (0 或 1)

---

## Part 3: 神经网络模型

📄 文件：[src/model.py](src/model.py)

> 这一节是整个项目最技术密集的部分。我会把每个组件分三段讲：
> **是什么 / 怎么工作 / 为什么选它**。读完之后你应该能在面试时自信地解释每一行代码的存在理由。

### 3.0 全局视角：神经网络在做什么

输入：一场比赛的 ~34 个数字（特征向量 x）
输出：一个 0~1 的数字（主队获胜概率 p）

中间发生的事情（用纯数学语言）：

```
p = sigmoid( W4 · ReLU( W3 · ReLU( W2 · ReLU( W1·x + b1 ) + b2 ) + b3 ) + b4 )
```

看起来吓人，其实就是**一连串矩阵乘法 + 非线性变换的嵌套**。我们要做的就是：找到一组让 p 尽可能接近真实标签 y 的 W 和 b。

### 3.1 为什么需要神经网络？

#### 逻辑回归的局限

逻辑回归学的是：
```
P(主队赢) = sigmoid(w₁·特征1 + w₂·特征2 + ... + wn·特征n + b)
```

每个特征单独贡献一个**线性**权重。它没法表达"组合效应"，比如：

> 主队净评分高 + 客队背靠背 → 赢的概率特别高（不只是两个效应相加，还有相乘的额外加成）

这种**特征之间的交互**（interaction）就是非线性。

#### 神经网络的本事

通过隐藏层 + 激活函数，神经网络能**自动**学到这些交互模式。第一层学简单组合，第二层学组合的组合，越深越抽象。

#### 但要诚实

对 NBA 预测这种**特征数少、样本数中等**的问题，神经网络通常**只比逻辑回归好一点点**（AUC +0.01~0.03）。我们用 NN 是因为：
1. 它能学到非线性，多少有提升
2. 这是 ML 学习项目，掌握 TensorFlow 用法本身就是收获
3. 同时跑 LR 基准做对比 — 量化神经网络的真实贡献

如果是图像、文本、语音 → 神经网络的优势会大得多。

---

### 3.2 输入层（Input）

```python
inputs = keras.Input(shape=(input_dim,), name="features")
```

**是什么**：定义输入张量的形状。`input_dim` 是特征数（约 34）。

**怎么工作**：每次喂入一个 batch（比如 64 场比赛），输入张量形状是 `(64, 34)`。

**为什么这样写**：Keras 函数式 API 要求显式声明输入。这让模型结构可视化更清楚、能多输入多输出。

---

### 3.3 第一隐藏层：`Dense(128)`

```python
x = keras.layers.Dense(128)(inputs)
```

#### 是什么
**全连接层**（Fully Connected / Dense）：每个输出神经元和**所有**输入神经元相连。

#### 怎么工作
数学上就是一个矩阵乘法 + 偏置：
```
输出[j] = Σ (W[i,j] × 输入[i]) + b[j]
```

具体到这一层：
- 输入：34 维向量
- 权重矩阵 W：形状 (34, 128) → **34 × 128 = 4,352 个权重参数**
- 偏置 b：128 维 → **128 个偏置参数**
- 输出：128 维向量

**用一个比喻**：
- 输入的 34 个特征像 34 种食材
- 每个神经元像一道菜的"配方"，决定每种食材放多少（权重）
- 128 个神经元 = 同时做 128 道菜

每道菜是不同特征的不同组合。模型在训练中学习"哪些组合有用"。

#### 为什么选 128
经验法则：第一层通常是输入维度的 **2-4 倍**，给模型足够"原料组合空间"。
- 太少（如 16）：可能学不到所有有用的组合
- 太多（如 1024）：参数过多，容易过拟合，训练慢

128 是中等数据集（几千样本）的常见甜点。

---

### 3.4 BatchNormalization

```python
x = keras.layers.BatchNormalization()(x)
```

#### 是什么
**批标准化**：把当前 batch 的输出强行变成均值 0、方差 1（再加两个可学习的缩放/平移参数）。

#### 怎么工作
对一个 batch 的 64 场比赛，假设第 j 个神经元在这 64 场上的输出是 `[2.1, -0.5, 1.8, ..., 3.2]`：
1. 算这 64 个数的均值 μ 和方差 σ²
2. 把每个值变成 `(x - μ) / √(σ² + ε)` → 现在均值 0、方差 1
3. 再用两个可学习的参数 γ（缩放）和 β（平移）调整：`γ·标准化值 + β`

**为什么还要 γ 和 β？** 强制均值 0 方差 1 太死板了，让模型自己学最合适的尺度。

#### 为什么用它
**问题：内部协变量偏移（Internal Covariate Shift）**
随着训练进行，前面层的权重在变 → 后面层接收到的输入分布也在变 → 就像射箭时靶子一直在动。

BatchNorm 把每层的输入强行拉回稳定分布，效果：
- ✅ **训练快很多**（可以用更大学习率）
- ✅ **对初始化不那么敏感**
- ✅ **轻微的正则化效果**（每个 batch 的均值方差略不同 → 给模型加了点噪音）

#### 放在哪里？
经典争议：BatchNorm 在激活函数**前**还是**后**？
- 原论文：前
- 实践常见：前
- 我们的代码：前

实测两种都能工作。

---

### 3.5 ReLU 激活函数

```python
x = keras.layers.Activation("relu")(x)
```

#### 是什么
**Rectified Linear Unit**：
```
ReLU(x) = max(0, x)
```
负数变 0，正数不变。

#### 怎么工作
逐元素操作。一个 128 维向量进去，一个 128 维向量出来，每个元素独立处理。

```
输入:  [2.1, -0.5, 1.8, -3.2, 0.7]
输出:  [2.1,  0.0, 1.8,  0.0, 0.7]
```

#### 为什么需要激活函数？
**核心定理**：没有非线性激活函数的话，**多层网络等价于一层**。

证明（线性代数）：
```
y = W2 · (W1 · x + b1) + b2
  = (W2·W1) · x + (W2·b1 + b2)
  = W' · x + b'
```

`W2·W1` 还是个矩阵 → 整个网络只是个线性变换 → **加多少层都没用**。

激活函数引入非线性，让网络真的能学复杂模式。

#### 为什么选 ReLU
ReLU 是 2010 年左右才流行的。之前主流是 sigmoid 和 tanh，但它们有**梯度消失问题**：在两端饱和（输入很大或很小时），梯度接近 0 → 反向传播更新不动权重 → 深层网络训不出来。

ReLU 的优点：
- ✅ **正区间梯度恒为 1**：不会消失
- ✅ **计算超快**：就是 max 操作
- ✅ **稀疏激活**：一半神经元输出 0 → 类似自动正则化

ReLU 的缺点：
- ⚠️ **死神经元问题**：如果一个神经元始终输出 0，梯度也是 0，永远不更新

变种（Leaky ReLU、ELU、GELU 等）解决死神经元问题，但 ReLU 在中等规模问题上够用了。

---

### 3.6 Dropout

```python
x = keras.layers.Dropout(0.3)(x)
```

#### 是什么
**随机失活**：训练时每次前向传播，**随机**把 30% 神经元的输出强制变 0。

#### 怎么工作
对 128 维输出，每次随机选 ~38 个（30%）变成 0，剩下 ~90 个保持不变（但乘以 1/0.7 = 1.43，保持总输出量级一致）。

**关键**：
- 训练时：随机失活 + 缩放
- 推理时：**不失活**，所有神经元正常工作
- Keras 自动处理这个区别，你不用管

#### 为什么用它（直觉）
**防止过拟合**。具体机制有两种解释：

**解释 1：模型不能依赖单一神经元**
没有 Dropout 时，某个神经元可能学到一个"作弊"特征（比如正好抓住了训练集里的一个特殊模式）→ 训练集表现好，测试集垮掉。

有 Dropout 时，任何神经元都可能在训练中"消失"，所以网络必须学**冗余**的、**鲁棒**的特征 → 即使一些神经元失活，整体还能工作。

**解释 2：隐式集成学习**
每次前向传播，因为随机失活不同的神经元 → 等于每次训练**一个不同的子网络**。整个训练过程相当于训练了**指数多个子网络的集合**，最后推理时一起用 → 类似集成学习的效果。

#### 为什么选 0.3
经验值：
- 隐藏层：0.2~0.5（我们用 0.3）
- 输入层（如果用）：0.1~0.2
- 最后一层（接近输出）：0.0~0.2（我们用 0.15）

太低 → 没什么效果
太高 → 训练不稳定，可能欠拟合

---

### 3.7 第二、三隐藏层：金字塔结构

```python
# 第二层
x = Dense(64)(x); x = BatchNorm()(x); x = ReLU()(x); x = Dropout(0.3)(x)
# 第三层
x = Dense(32)(x); x = ReLU()(x); x = Dropout(0.15)(x)
```

#### 为什么神经元数递减（128 → 64 → 32）？
这叫**金字塔架构**（pyramid / encoder 结构）。

**直觉**：
- 第一层：从原始特征里提取**很多低层组合**（128 个）
- 第二层：把低层组合**整合成高层概念**（64 个）
- 第三层：进一步**抽象总结**（32 个）
- 输出层：最终决策（1 个数字 = 主队获胜概率）

像漏斗一样，信息逐层浓缩。

**对比**：等宽（128-128-128）也行，但金字塔通常**参数更少、过拟合风险更低**，对中等数据集更友好。

#### 为什么第三层去掉 BatchNorm？
经验：靠近输出的层用 BatchNorm 反而可能伤害效果（限制了输出灵活性）。常见做法是**前 2-3 层用 BN，最后一层不用**。这是"好用就保留"型的工程经验，没有强理论基础。

#### 为什么最后一层 Dropout 减半（0.15）？
靠近输出的特征更"成型"，太多失活会破坏信息。所以渐弱使用 Dropout。

---

### 3.8 输出层：`Dense(1) + Sigmoid`

```python
output = keras.layers.Dense(1, activation="sigmoid", name="home_win_prob")(x)
```

#### 是什么
单个神经元 + sigmoid 激活，输出一个 0~1 之间的数。

#### 怎么工作
1. 32 维特征 → 1 个数字（叫 logit，可以是任意实数）
2. sigmoid(logit) = 1 / (1 + e^(-logit)) → 压缩到 (0, 1)

```
logit = -3 → sigmoid ≈ 0.05
logit =  0 → sigmoid  = 0.50
logit = +3 → sigmoid ≈ 0.95
```

#### 为什么用 sigmoid（不是别的）？
对于**二分类**，sigmoid 是标准选择：
- 输出可以解释为**概率**
- 与 binary cross-entropy 损失数学上"配对"，反向传播形式优雅
- 可微，到处可导

对于多分类（比如 5 种结果），会用 softmax。我们是 2 类问题 → sigmoid 足够。

---

### 3.9 损失函数：Binary Cross-Entropy

```python
loss="binary_crossentropy"
```

#### 是什么
衡量预测概率 p 和真实标签 y（0 或 1）的差距：
```
loss = -[y · log(p) + (1-y) · log(1-p)]
```

#### 怎么工作（举例）

| 真实 y | 预测 p | 损失 |
|---|---|---|
| 1 (主队真的赢了) | 0.99 | -log(0.99) ≈ **0.01** （很小，预测正确） |
| 1 | 0.50 | -log(0.50) ≈ **0.69** （中等） |
| 1 | 0.01 | -log(0.01) ≈ **4.61** （大，预测错得离谱） |
| 0 (主队输了) | 0.99 | -log(1-0.99) ≈ **4.61** （大，预测错得离谱） |
| 0 | 0.01 | -log(1-0.01) ≈ **0.01** （很小） |

**关键性质**：错得越离谱（p 越靠近错的那一端），损失**指数级**增大 → 模型有强烈动力修正大错。

#### 为什么不直接用准确率作为损失？
准确率（用阈值 0.5 判 0/1）**不可微** → 没法用梯度下降。

而且准确率忽略了"自信程度"：预测 0.51 和 0.99 都被算作"主队赢"，但前者明显信心不足。Cross-entropy 会奖励**有信心且正确**的预测、**惩罚有信心但错误**的预测。

#### 为什么不用 MSE（均方误差）？
回归问题用 MSE，分类问题用 cross-entropy。原因：
- 对分类问题，MSE 的梯度在错的离谱时反而**变小**（数学不友好）
- Cross-entropy 配 sigmoid 的导数特别简洁：`梯度 = (p - y)`

---

### 3.10 优化器：Adam

```python
optimizer=keras.optimizers.Adam(learning_rate=1e-3)
```

#### 是什么
**自适应矩估计**（Adaptive Moment Estimation）。是 SGD（随机梯度下降）的高级版本。

#### 怎么工作
SGD 的更新规则：
```
权重 -= 学习率 × 梯度
```

Adam 在此基础上加了两件事：
1. **动量**（momentum）：用过去梯度的滑动平均，平滑掉噪声
2. **自适应学习率**：每个参数有自己的学习率，根据该参数历史梯度的方差调整

简化公式：
```
m_t = β1 · m_{t-1} + (1-β1) · 梯度       ← 一阶矩（梯度的滑动平均）
v_t = β2 · v_{t-1} + (1-β2) · 梯度²      ← 二阶矩（梯度的滑动方差）
权重 -= 学习率 · m_t / (√v_t + ε)
```

效果：
- 频繁出现的方向 → 更新更稳（被 √v_t 缩放）
- 罕见方向 → 也能更新（一阶矩会累积）

#### 为什么选 Adam（不是 SGD）
**对中等规模问题，Adam 几乎是默认选择**：
- ✅ 不用太调超参，开箱即用
- ✅ 收敛快
- ✅ 对学习率不敏感

SGD 有时能找到**更好的最终解**（特别是大数据 + 长训练），但需要精细调参。
RMSprop 是 Adam 的简化版，效果接近。

#### 学习率 1e-3 = 0.001 怎么来的
Adam 论文推荐的默认值，**在 80% 的问题上效果都不错**。
- 太大（如 0.01）：训练发散
- 太小（如 1e-5）：训练慢得没法看

---

### 3.11 训练时监控的指标

```python
metrics=["accuracy", AUC(), Precision(), Recall()]
```

这些**不参与训练**（不算梯度），只是每个 epoch 末尾打印出来给你看的：
- **accuracy**：准确率，最直观
- **AUC**：和阈值无关的总体好坏 → 我们用它做早停
- **precision / recall**：理解模型偏向（更倾向预测主胜还是客胜）

注意：**loss 才是模型实际优化的目标**。指标只是"翻译给人看"的版本。

---

### 3.12 回顾：参数总数

让我们数一下整个网络有多少要学的参数：

| 层 | 输入 → 输出 | 权重 + 偏置 |
|---|---|---|
| Dense 1 | 34 → 128 | 34×128 + 128 = 4,480 |
| BN 1 | 128 → 128 | 128×4 = 512 |
| Dense 2 | 128 → 64 | 128×64 + 64 = 8,256 |
| BN 2 | 64 → 64 | 64×4 = 256 |
| Dense 3 | 64 → 32 | 64×32 + 32 = 2,080 |
| Dense 4 | 32 → 1 | 32×1 + 1 = 33 |
| **总计** | | **~15,600 参数** |

**经验法则**：训练样本数 ≥ 参数数 × 5~10 → 不容易过拟合。
- 我们：~3,500 训练样本 vs 15,600 参数 → **不太够**！
- 这就是为什么 Dropout、BatchNorm、EarlyStopping **必须用上**

如果数据更多，可以试更大的网络。如果想缩小，可以把 128-64-32 改成 64-32-16。

---

### 3.13 一句话总结架构选择的理由

> 输入 34 个特征 → 用金字塔结构（128→64→32）逐层抽象，每层用 BatchNorm 稳定训练、ReLU 引入非线性、Dropout 防过拟合，最后单神经元 + sigmoid 输出概率。Adam 优化器配 binary cross-entropy 损失。早停 + 监控验证集 AUC 防止过度训练。

这段话基本能回答面试官问的"你为什么这样设计网络"。

---

## Part 4: 训练流程

📄 文件：[src/train.py](src/train.py)

### 步骤 1：按时间切分数据

```
全部 3,500 场比赛，按日期排好序：

[──── 训练集 65% ────][── 验证集 15% ──][── 测试集 20% ──]
   2021-22 + 大部分 22-23      22-23 末尾         23-24
```

**为什么不能用 sklearn 默认的 `train_test_split` 随机切？**
- 体育比赛是时间序列，**球队实力会变**
- 随机切等于"用未来训练去预测过去"，指标会虚高
- 真实部署时，你只能用过去预测未来 → 切分必须模拟这一点

### 步骤 2：标准化（StandardScaler）

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 计算均值方差，并应用
X_val_scaled   = scaler.transform(X_val)        # 只应用训练集的统计
X_test_scaled  = scaler.transform(X_test)
```

**关键**：`fit()` 只在训练集上做。如果在全部数据上 fit，等于让模型偷看了测试集分布 → 又一种数据泄露。

标准化后每个特征均值 0、方差 1。神经网络对特征尺度很敏感，不做标准化会训练不稳定。

### 步骤 3：训练循环

```python
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[早停, 自动调整学习率, 保存最佳模型],
)
```

**一个 epoch = 全部训练数据过一遍**

每个 epoch 内部：
1. 数据被切成大小为 64 的 batch
2. 每个 batch：前向传播算 loss → 反向传播算梯度 → 更新权重
3. 一个 epoch 结束后，在验证集上算指标

### 步骤 4：早停（EarlyStopping）

```python
EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True)
```

- 监控验证集 AUC
- 如果连续 15 个 epoch 没有提升 → 停止训练
- 自动恢复到 AUC 最高那一轮的权重

**为什么需要？** 训练太多轮会**过拟合**：训练集表现一直变好，但验证集开始变差。早停能在这个拐点附近停下。

### 步骤 5：基准模型

```python
LogisticRegression().fit(X, y)
```

逻辑回归只学一个线性公式：`P(home_win) = sigmoid(w₁·特征1 + w₂·特征2 + ... + b)`

**为什么训练它？** 对比！
- 如果 NN ≈ LR → 神经网络没学到啥额外的东西，可能数据本身没那么多非线性
- 如果 NN > LR → NN 学到了非线性模式，物有所值

---

## Part 5: 评估与可视化

📄 文件：[src/evaluate.py](src/evaluate.py)

### 5 个核心指标

#### 1. Accuracy（准确率）
```
正确预测数 / 总预测数
```
- 直观，但有陷阱：如果数据不平衡（比如 95% 主队赢），全猜主队赢就有 95% 准确率，但模型啥都没学到
- NBA 主客胜率比较平衡（59/41），所以 accuracy 还算可信

#### 2. Precision（精确率）
```
真正例 / (真正例 + 假正例)
= 在我说"主队赢"的所有预测里，有多少真的赢了
```
- 在意"别误报"时看这个（比如垃圾邮件检测）

#### 3. Recall（召回率）
```
真正例 / (真正例 + 假反例)
= 在所有主队真的赢的比赛里，我抓到了多少
```
- 在意"别漏报"时看这个（比如疾病诊断）

#### 4. F1-Score
```
2 × (Precision × Recall) / (Precision + Recall)
```
- Precision 和 Recall 的调和平均
- 两者都重要时看 F1

#### 5. ROC-AUC（最重要！）
- AUC = ROC 曲线下的面积，值在 0.5 ~ 1.0
- **AUC = 0.5**：和瞎猜一样
- **AUC = 0.7**：还行
- **AUC = 0.8+**：很好
- **AUC = 1.0**：完美（不可能在真实数据上达到）

**直觉解释**：随机抽一场主队赢的比赛和一场主队输的比赛，模型给前者更高分的概率 = AUC。

---

### 5 张图怎么读

#### 📈 图 1：训练曲线 (`training_curves.png`)

两幅子图：左边 loss、右边 AUC，都画了**训练**和**验证**两条线。

| 现象 | 说明 |
|---|---|
| 训练 loss ↓ + 验证 loss ↓ | 健康，继续训练 |
| 训练 loss ↓ + 验证 loss ↑ | **过拟合**！应该早停 |
| 训练 loss 不下降 | 欠拟合，模型太简单或学习率不对 |

**理想形态**：两条线同步下降，验证线比训练线略高。

#### 📊 图 2：混淆矩阵 (`confusion_matrix.png`)

一个 2×2 的方阵：

|  | 预测客胜 | 预测主胜 |
|---|---|---|
| **实际客胜** | TN（真反例） | FP（假正例） |
| **实际主胜** | FN（假反例） | TP（真正例） |

对角线（左上 + 右下）= 预测正确的数量。**理想是对角线数字大、非对角线数字小**。

#### 📉 图 3：ROC 曲线 (`roc_curve.png`)

X 轴：假正例率（错把客胜判成主胜的比例）
Y 轴：真正例率（正确识别主胜的比例）

**怎么读**：
- 曲线越靠近左上角越好
- 对角虚线是"瞎猜"的水平
- 多模型对比时，曲线在上面的更好
- AUC 数字标在图例里

#### 📐 图 4：校准曲线 (`calibration.png`)

X 轴：模型预测的概率（比如"我觉得主队 70% 会赢"）
Y 轴：实际胜率（这些"我说 70%"的比赛中，主队真的赢的比例）

**理想**：完美沿对角线 — 模型说 70% 就真的赢 70%。

**实际意义**：
- 如果只关心**预测对错**，准确率/AUC 够用
- 如果要把概率拿去做决策（押注、做组合），**校准必须好**
- 神经网络配 sigmoid 通常校准还行；树模型经常校准很差

#### 📊 图 5：特征重要性 (`feature_importance.png`)

**注意**：这张图用的是**逻辑回归**的系数，不是神经网络的。
- NN 的特征重要性很难直接算（要用 SHAP、Permutation Importance 等技术）
- LR 的系数能直接告诉你"这个特征对预测影响有多大"
- 当 NN 和 LR 表现差不多时，LR 的系数能给你一个不错的近似

---

## Part 6: 如何阅读输出

跑完 `python main.py` 之后你会看到：

### 训练日志（每个 epoch 一行）
```
Epoch 23/100
44/44 [==============================] - 0s 4ms/step
  - loss: 0.5821 - accuracy: 0.6948 - auc: 0.7651
  - val_loss: 0.6234 - val_accuracy: 0.6512 - val_auc: 0.7012
```

每行包含：
- `loss` / `accuracy` / `auc`：**训练集**指标
- `val_loss` / `val_accuracy` / `val_auc`：**验证集**指标
- 我们关心的是 `val_*`（训练集会越来越好直到过拟合）

如果看到：
```
Epoch 38: ReduceLROnPlateau reducing learning rate to 0.0005
```
说明学习率自动减半了 — 模型卡住了，调小步长再试。

```
Epoch 45: early stopping
```
说明早停触发，训练结束。

### 最终指标表
```
══════════════════════════════════════════
Metric          Neural Net    Logistic Reg
──────────────────────────────────────────
accuracy           0.6452          0.6298
precision          0.6750          0.6603
recall             0.7710          0.7589
f1                 0.7198          0.7062
roc_auc            0.6850          0.6678
══════════════════════════════════════════
```

**怎么解读**：
- 看 `accuracy`：64.5% > 主队基础胜率 59% → 模型学到了东西 ✅
- 看 `roc_auc`：0.685 → 在"还行"和"很好"之间，对体育预测来说算合理
- NN > LR：说明数据里有非线性模式 ✅
- 但差距不大（0.015 AUC）：意味着这个问题大部分信号是线性的，不需要太复杂的模型

---

## Part 7: 面试常问问题

如果面试官看你的 GitHub 项目，他可能会问：

### Q1: "你怎么防止数据泄露？"
**A**: 三个层面：
1. 特征工程时用 `.shift(1)` 排除当前比赛
2. 训练/验证/测试集**按时间切**而不是随机切
3. `StandardScaler` 只在训练集上 `fit`

### Q2: "为什么用神经网络而不是更简单的模型？"
**A**: 项目里同时跑了逻辑回归基准做对比。NN 的 AUC 比 LR 高约 1.5 个百分点，说明数据里有非线性模式神经网络能捕捉。但坦白说差距不算大，这个量级的数据上 XGBoost 可能更合适 — 这也是 Future Work 里写的。

### Q3: "怎么处理过拟合？"
**A**: 三个机制：
1. Dropout（每层随机失活 30%）
2. BatchNormalization（隐含的正则化效果）
3. EarlyStopping（监控验证集 AUC，patience=15）

### Q4: "如何评估模型好坏？"
**A**: 不只看准确率。我们看 5 个指标：accuracy、precision、recall、F1、AUC。还看 5 张诊断图：训练曲线、ROC、混淆矩阵、校准、特征重要性。AUC 是首要指标因为它和阈值无关。

### Q5: "下一步怎么改进？"
**A**: README 里列了几条：
- 加入伤病数据（伤病对结果影响巨大）
- 加 ELO 评分作为更稳定的实力衡量
- 试试 LSTM 把球队最近表现作为序列建模
- 集成 NN + XGBoost

---

## Part 8: 高级特征 — ELO + 伤病

📄 文件：[src/elo.py](src/elo.py) ｜ [src/injuries.py](src/injuries.py)

这两个特征在体育预测里**通常是单一最强信号**。让我们一个个讲。

---

### 8.1 ELO 评分系统

#### 起源
ELO 是匈牙利物理学家 Arpad Elo 在 1960 年代发明的，最初用于国际象棋排名。后来被广泛用于围棋、足球、电竞、NBA。

#### 核心思想
每支球队有一个**单一数字**代表当前实力（NBA 通常起点 1500）。每场比赛结束后，根据**预期 vs 实际结果**调整：

```
新评分 = 旧评分 + K × (实际结果 − 预期结果)
```

- `实际结果`：赢了 = 1，输了 = 0
- `预期结果`：根据当前评分差算出的获胜概率
- `K`：调整速度（NBA 一般用 20）

#### 预期胜率公式

```
预期 = 1 / (1 + 10^((对手评分 − 自己评分) / 400))
```

**400 点的差距 = 10 倍获胜概率**：
- 评分相等 → 预期 50%
- 高 100 点 → 预期 ~64%
- 高 200 点 → 预期 ~76%
- 高 400 点 → 预期 ~91%

#### 举个例子

| 项 | 数值 |
|---|---|
| 湖人 ELO | 1550 |
| 勇士 ELO | 1500 |
| 主场 = 湖人，主场加成 +100 → 有效 ELO | 1650 |
| 湖人预期胜率 | `1/(1+10^((1500-1650)/400))` ≈ 70% |

如果湖人真赢了 → 实际 1，预期 0.7 → 评分增加 `20 × (1−0.7) = +6`
如果湖人输了 → 实际 0，预期 0.7 → 评分减少 `20 × (0−0.7) = −14`

→ **冷门（弱队赢强队）涨分多，正常胜负调整小**。这就是 ELO 的精髓。

#### Margin-of-Victory 调整（赢多少分也算）

简单 ELO 不区分赢 1 分还是赢 30 分。FiveThirtyEight 改进版加了一个**赢分多少**的乘数：

```
mov = (|分差| + 3)^0.8 / (7.5 + 0.006 × 赢家ELO优势)
```

- 赢 30 分 → mov ≈ 1.8 → 评分变化乘 1.8
- 赢 1 分  → mov ≈ 0.6 → 评分变化只有 0.6 倍

#### 赛季衔接

每个赛季结束，所有球队评分**向 1500 回归 25%**：

```
新赛季评分 = 0.75 × 上赛季末评分 + 0.25 × 1500
```

为什么？球队休赛期会变（选秀、自由球员、伤病）→ 上赛季评分不能完全代表新赛季。

#### 在我们项目里

```python
from src.elo import compute_elo_features
games = compute_elo_features(games)
# 新增列: home_elo_pre, away_elo_pre, elo_diff
```

注意：**永远只用 pre-game ELO 作特征**。如果用 post-game ELO，你已经知道了胜负 → 数据泄露。

#### 为什么 ELO 这么强？

它在一个数字里编码了：
- ✅ 球队历史所有比赛结果
- ✅ 对手强度（赢强队加分多）
- ✅ 赢分差（blowout 加分多）
- ✅ 主客场（通过有效 ELO 体现）

这就是为什么单一 `elo_diff` 特征通常能贡献 2-3 个百分点的 AUC 提升。

---

### 8.2 伤病特征：星球员可用性

#### 为什么伤病关键
NBA 球员实力**严重不均衡**。詹姆斯+戴维斯 vs 詹姆斯一个人 → 完全不同的两支队。如果模型不知道明星球员是否上场，它会把"湖人"当成一个均匀实体来看。

#### 现实问题：没有干净的伤病 API
- 官方伤病报告分散在新闻和球队公告里
- 第三方爬虫不稳定
- nba_api 没有专门的 injury 端点

#### 我们的代理方案：星球员出场情况

**思路**：
1. 找出每队每赛季**总出场时间最多的 5 个球员** → 这就是"明星"
2. 检查每场比赛**这些明星实际上场了几个**
3. 这个数字（0-5）就是球员可用性指标

代码实现在 `src/injuries.py`：

```python
# 第一步：识别明星
totals = player_logs.groupby(["SEASON", "TEAM_ID", "PLAYER_ID"])["MIN"].sum()
stars = totals.sort_values(ascending=False).groupby(["SEASON", "TEAM_ID"]).head(5)

# 第二步：每场比赛数明星出场数
played = (player_logs["MIN"] > 0).astype(int)
star_apps = player_logs.merge(stars, ...).groupby(["GAME_ID", "TEAM_ID"])["played"].sum()
```

#### 数据获取的优化

如果一场场调用 API（`boxscoretraditionalv2`），3000 场比赛要 3000 次请求 → 几小时。

**关键技巧**：用 `LeagueGameLog(player_or_team_abbreviation="P")` → **一个赛季一次请求**就能拿到所有球员所有比赛的数据。

#### 数据泄露的轻微问题
我们用**整赛季**的数据来识别明星，但游戏 G 在赛季中部 → 严格说是用了未来信息（"明星身份"是赛季末才能确定的）。

**为什么还能用？**
- "谁是球队主力"是相对稳定的信息（不会一夜变天）
- 实际部署时可以用上赛季的明星名单代替
- 这是工程上的合理简化，**面试时主动提出来 = 加分项**（说明你知道这个 trade-off）

#### 在我们项目里

```python
features = build_features(games, star_avail=star_avail)
# 新增列: home_stars_avail, away_stars_avail, stars_avail_diff
```

`stars_avail_diff` 大概是这样：
- `+2` → 主队 5 个明星都在，客队只有 3 个 → 主队大概率会赢
- `0`  → 双方阵容齐整
- `-3` → 主队伤了 3 个明星，客队齐整 → 客队优势

---

### 8.3 加完之后的预期效果

| 配置 | AUC（典型值） |
|---|---|
| 只有滚动统计 | 0.685 |
| + ELO | 0.705-0.715 |
| + ELO + 伤病代理 | 0.715-0.730 |

**注意**：这些是经验估计，你的实际数字会因数据和随机种子而异。重点是看**相对提升**，不是绝对值。

### 8.4 用法

代码已经默认开启。如果想关掉对比效果，改 `config.py`：

```python
USE_ELO_FEATURES    = False
USE_INJURY_FEATURES = False
```

跑两次比较 AUC：
```bash
# 跑一次开着，记下 AUC
python main.py

# 关掉 ELO+伤病，再跑一次
# (改 config.py 后)
python main.py --skip-fetch
```

差值就是这两个特征的贡献。

---

### 8.5 面试加分回答

如果面试官问"你怎么改进了基础模型"：

> 我先用滚动统计跑了一个基线，AUC 大概 0.68。然后加了两组高级特征：第一是 ELO 评分，按 FiveThirtyEight 的方法实现，包括主场加成、赢分差乘数、跨赛季衰减；第二是基于 box score 的星球员可用性代理，用每队每赛季出场时间前 5 的球员作为"星球员"。这两组特征加起来大概把 AUC 从 0.68 提到 0.72。我也意识到星球员识别用了赛季末数据有轻微泄露，更严格的版本会用上赛季的名单。

这段话展示：
- ✅ 实验思维（基线 → 改进 → 量化提升）
- ✅ 借鉴行业最佳实践（FiveThirtyEight）
- ✅ 工程权衡意识（知道有泄露但选择简化）
- ✅ 自我批判（主动指出问题）

---

## Part 9: 深度专题

> 这一部分是给学进阶内容的人看的。理解这 5 个主题之后，你不只是会用 TensorFlow，而是真的**懂**神经网络在做什么。面试时问到细节也能从容应对。

---

### 9.1 反向传播 — 神经网络是怎么"学"的

#### 大问题：训练究竟在干嘛？

我们说"模型从数据学习"。具体是怎么学？

答案：**通过反向传播（backpropagation）和梯度下降（gradient descent），不停调整每个权重 W 和偏置 b，让损失（loss）越来越小**。

#### 整个训练循环（一个 batch 的完整步骤）

```
[前向传播 forward pass]   把数据喂进网络，得到预测值
        ↓
[计算损失 compute loss]    比较预测和真实标签，得到一个数 (loss)
        ↓
[反向传播 backward pass]   从 loss 反推每个参数应该往哪个方向调
        ↓
[更新参数 update weights]  用梯度下降更新所有 W 和 b
        ↓
[重复]                    下一个 batch
```

#### 用一个超小的例子讲清楚

假设我们只有**一层网络、一个权重**：

```
预测 ŷ = w · x
损失 L = (ŷ - y)²       ← MSE
```

具体数值：
- 输入 x = 2
- 真实标签 y = 6
- 当前权重 w = 1
- 预测 ŷ = 1 × 2 = 2
- 损失 L = (2 - 6)² = **16**（错得很多）

#### 求梯度（这是反向传播的本质）

**梯度 = 损失对每个参数的偏导数**。直觉：损失对这个参数有多敏感？

用链式法则：
```
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂w
      = 2(ŷ - y) × x
      = 2(2 - 6) × 2
      = -16
```

**梯度 = -16 是什么意思？**
- 负的 → 增大 w 会让 loss 减小
- 绝对值大 → 这个参数对 loss 影响大

#### 梯度下降更新

```
w_new = w_old - 学习率 × 梯度
      = 1 - 0.1 × (-16)
      = 1 + 1.6
      = 2.6
```

新的预测：ŷ = 2.6 × 2 = 5.2，比原来的 2 离 6 更近了！损失从 16 降到 (5.2-6)² = 0.64。

**这就是学习的全部：算梯度 → 沿反方向走一小步 → 重复**。

#### 多层网络的链式法则

我们实际的网络有多层。从损失到第一层权重的梯度，要把每一层的"局部梯度"乘起来：

```
∂L/∂W₁ = ∂L/∂ŷ  ×  ∂ŷ/∂h₃  ×  ∂h₃/∂h₂  ×  ∂h₂/∂h₁  ×  ∂h₁/∂W₁
         (输出层)    (第3层)    (第2层)    (第1层)     (输入层)
```

每个箭头反向传一次梯度。这就是为什么叫"反向"传播 — **梯度从损失沿网络反向流回每一层**。

#### TensorFlow 自动做这件事

```python
model.fit(X, y)   # 这一行内部其实在做：
```

```python
for batch in batches:
    with tf.GradientTape() as tape:        # 记录所有计算
        predictions = model(batch_X)        # 前向传播
        loss = loss_fn(batch_y, predictions)
    grads = tape.gradient(loss, model.trainable_variables)  # 自动求所有梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 更新
```

`GradientTape` 会自动用链式法则计算所有梯度。**这就是深度学习框架的核心价值** — 它替你自动做微积分。

#### 关键概念：什么是 epoch / batch / iteration

| 术语 | 含义 | 我们项目里的数字 |
|---|---|---|
| **样本 (sample)** | 一个数据点 | 一场比赛 |
| **batch** | 一次同时处理的样本数 | 64 场 |
| **iteration** | 一次梯度更新（处理一个 batch） | ~44 次 / epoch |
| **epoch** | 全部数据过一遍 | ~3000 / 64 ≈ 44 iterations |

我们设 `epochs=100`，意味着同样的数据会被反复使用 100 次。每次的随机 batch 顺序不同，dropout 失活的神经元也不同 → 每个 epoch 都能学到一点新东西。

#### 为什么用 batch（不是一次喂全部）？

| 方式 | 速度 | 内存 | 梯度噪声 | 收敛 |
|---|---|---|---|---|
| 全部数据（Batch GD）| 慢 | 内存炸 | 无噪声 | 容易陷入局部最优 |
| 单样本（SGD） | 快 | 小 | 巨大噪声 | 不稳定 |
| 小批量（Mini-batch）| **中** | **可控** | **适度噪声** | **既快又稳** ✅ |

`batch_size=64` 是甜点。**适度的噪声反而是好事** — 帮助模型跳出局部最优。

---

### 9.2 梯度消失 / 爆炸 — 为什么 ReLU 比 sigmoid 好

#### 问题来源：链式法则的乘法

回忆刚才那个公式：
```
∂L/∂W₁ = (导数₁) × (导数₂) × (导数₃) × ... × (导数ₙ)
```

如果每个"导数ᵢ" < 1 → 它们的乘积**指数级缩小**到 0
如果每个"导数ᵢ" > 1 → 它们的乘积**指数级放大**到无穷

这就是**梯度消失**和**梯度爆炸**。

#### Sigmoid 的致命缺陷

Sigmoid 函数：
```
σ(x) = 1 / (1 + e^(-x))
```

它的导数：
```
σ'(x) = σ(x) · (1 - σ(x))
```

**最大值是多少？** 当 σ(x) = 0.5 时（即 x = 0），导数 = 0.5 × 0.5 = **0.25**。

也就是说，**sigmoid 的导数永远不会超过 0.25**。

#### 灾难性后果（10 层网络）

假设网络有 10 层 sigmoid，最佳情况下每层导数都是 0.25：
```
最终梯度 ≈ 0.25¹⁰ ≈ 0.0000009
```

接近 0！→ 第一层的权重**基本不更新** → 深层网络根本训不出来。

更糟：sigmoid 在两端**饱和**（输入很大或很小时导数接近 0），实际情况比上面还惨。

#### ReLU 的优势

ReLU 的导数：
```
ReLU'(x) = 1   if x > 0
           0   if x ≤ 0
```

**关键**：在正区间，导数恒为 1！

10 层 ReLU 的梯度乘积：
```
最终梯度 ≈ 1¹⁰ = 1
```

完全不衰减！梯度能畅通无阻地传到第一层。

#### 直观对比

| 激活函数 | 导数最大值 | 100 层后梯度 | 结论 |
|---|---|---|---|
| Sigmoid | 0.25 | ≈ 0 | ❌ 深层不可用 |
| Tanh | 1.0（但常 < 1） | 衰减 | ⚠️ 中等 |
| ReLU | 1.0（正区间稳定） | 1 | ✅ 深层友好 |

#### 但 ReLU 不完美：死神经元问题

ReLU 在 x ≤ 0 时导数为 0。如果一个神经元的输入**总是负的**（比如初始化不好，或学习率太大把权重推到了一个不好的位置），它将：
- 输出永远是 0
- 梯度永远是 0
- **权重永远不会更新** → "死了"

#### 改进版 ReLU

| 变种 | 公式 | 解决死神经元 |
|---|---|---|
| **Leaky ReLU** | `max(0.01x, x)` | ✅ 负区间小斜率 |
| **PReLU** | `max(αx, x)`，α 可学 | ✅ 自适应 |
| **ELU** | `x` (正) / `α(eˣ-1)` (负) | ✅ 负区间平滑 |
| **GELU** | 复杂 sigmoid 类 | ✅ Transformer 用的多 |

我们项目用普通 ReLU 就够了，因为：
- 网络只有 3 层（不深）
- 配合 BatchNorm 进一步缓解死神经元

#### 梯度爆炸（反方向问题）

如果权重初始化太大或激活函数导数 > 1（比如 RNN 中的某些情况）→ 梯度乘积**指数增长** → 数值溢出（NaN）。

**应对方法**：
- **梯度裁剪**：`tf.clip_by_norm(grads, 1.0)` 强行限制梯度大小
- **权重正则化**：L2 正则惩罚大权重
- **更好的初始化**（见下文）

#### 重要：权重初始化

如果所有权重初始化为 0 → 所有神经元输出一样 → 没法学不同特征 → "对称性问题"。

如果初始化太大 → 梯度爆炸；太小 → 梯度消失。

**He 初始化**（针对 ReLU 设计）：
```
W ~ Normal(0, √(2/n_in))
```
其中 n_in 是该层输入维度。

Keras 的 `Dense` 默认就是合理的初始化（Glorot/He），你不用手动设置。但面试可能会问，**记住"He 初始化是 ReLU 的标配"**。

---

### 9.3 过拟合诊断 — 看训练曲线判断模型状态

#### 4 种典型曲线形态

跑完 `python main.py` 后看 `outputs/plots/training_curves.png`。曲线形状告诉你模型状态。

##### 形态 1：欠拟合（Underfitting）
```
loss
 │
 │  ━━━━━━━━━━━━  train（高且不下降）
 │  ━━━━━━━━━━━━  val
 │
 └──────────────► epoch
```

**症状**：
- 训练 loss 一直很高、不下降
- 验证 loss 也很高
- 准确率不超过基础胜率（59%）

**原因**：模型太弱、学不到模式
- 网络太浅 / 太窄
- 学习率太小
- 特征不够好
- 训练 epoch 太少

**解决**：
- 加层 / 加神经元
- 提高学习率
- 加更多有用特征
- 多训练几轮

##### 形态 2：完美拟合（Good Fit）✅
```
loss
 │\
 │ \
 │  \\___          train
 │    \\___        val（略高于 train，且稳定）
 │      \\\\___
 │       
 └──────────────► epoch
```

**症状**：
- 训练 loss 下降并稳定
- 验证 loss 跟着下降并稳定
- 验证 loss **略高于**训练 loss（很正常）

**这是我们想看到的**。说明模型学到了东西，又没过分依赖训练集细节。

##### 形态 3：过拟合（Overfitting）⚠️
```
loss
 │\
 │ \
 │  \___          val（先降后升）
 │     \____/      
 │      \____    
 │       \___    train（一直下降）
 │       
 └──────────────► epoch
```

**症状**：
- 训练 loss 一直下降（甚至接近 0）
- 验证 loss **先下降后回升** ← 关键信号
- 训练准确率比验证高很多（如 95% vs 65%）

**原因**：模型背下了训练集的噪声 / 偶然模式，没学到通用规律
- 网络太大（参数太多）
- 数据太少
- 训练太久
- 缺正则化

**解决**：
- 加 Dropout / L2 正则
- 减小网络
- 加数据 / 数据增强
- **EarlyStopping**（在 val 开始上升时停下）⭐

我们的代码默认开启了 EarlyStopping，所以即使发生过拟合也会自动停在最佳点。

##### 形态 4：训练崩了（Broken）🚨
```
loss
 │
 │  /\/\/\        震荡
 │ /  \  /\
 │/    \/  \      或 NaN（直接消失）
 │
 └──────────────► epoch
```

**症状**：
- loss 震荡剧烈
- loss 突然变 NaN
- 完全没在学

**原因**：
- 学习率太大（步子迈太大跳过最优）
- 数据有问题（NaN、无穷值）
- 梯度爆炸
- 标签和特征不对应（bug）

**解决**：
- 把学习率降一个数量级（1e-3 → 1e-4）
- 检查数据：`X.isna().sum()` / `np.isinf(X).any()`
- 加梯度裁剪
- 仔细检查数据预处理代码

#### 训练 vs 验证 gap 的解读

定义：**gap = 验证 loss - 训练 loss**

| Gap 大小 | 解读 | 行动 |
|---|---|---|
| Gap ≈ 0 或负 | 欠拟合或异常 | 加大模型 / 检查 bug |
| Gap 小（< 10%） | 健康 ✅ | 不用动 |
| Gap 中（10-30%） | 轻微过拟合 | 加 Dropout / 减小网络 |
| Gap 大（> 30%） | 严重过拟合 | 数据太少或模型太大 |

#### 偏差-方差权衡（Bias-Variance Tradeoff）

这是机器学习的核心理论：

```
总误差 = 偏差² + 方差 + 不可约噪声
```

- **高偏差（Bias）= 欠拟合**：模型太简单，对训练集都学不好
- **高方差（Variance）= 过拟合**：模型太复杂，对训练集敏感、对新数据差

**调参就是在这两者之间找平衡**。我们的"金字塔 + Dropout + EarlyStopping"组合就是控制方差的常见配方。

#### 在我们项目里诊断

跑完之后看：

```
最后一个 epoch:
  loss: 0.55   accuracy: 0.69
  val_loss: 0.62   val_accuracy: 0.65
```

- gap = 0.62 - 0.55 = 0.07 → ~13% → 轻微过拟合，可接受
- 如果 gap 超过 0.20 → 增加 Dropout 到 0.4-0.5

---

### 9.4 Adam 内部机制 — 一阶矩 / 二阶矩 / 偏差修正

#### 从 SGD 出发，逐步加配件

##### 第 0 层：朴素 SGD
```
w = w - lr · g    (g 是梯度)
```

**问题**：
- 如果不同方向的梯度尺度差很多 → 同一个学习率两边都不合适
- 如果梯度有噪声 → 更新方向震荡

##### 第 1 层：加动量（Momentum）

想象一个球从山上滚下来：
- 朴素 SGD = 每步看当前坡度走
- 动量 = 还有惯性，会沿之前方向继续滚一会

```
v = β₁ · v + (1 - β₁) · g     ← 动量（梯度的滑动平均）
w = w - lr · v
```

`β₁ = 0.9`（典型值）意味着：
- 90% 是过去的梯度方向
- 10% 是当前梯度方向

**效果**：
- 抵抗梯度噪声（平均掉）
- 在峡谷地形里加速（梯度方向稳定时累积）

`v` 叫做**一阶矩**（first moment），因为它是梯度本身的期望。

##### 第 2 层：加自适应学习率（RMSprop 思想）

不同参数应该用不同学习率：
- 经常出现大梯度的参数 → 减小步长
- 很少出现梯度的参数 → 加大步长

```
s = β₂ · s + (1 - β₂) · g²    ← 二阶矩（梯度平方的滑动平均）
w = w - lr · g / (√s + ε)
```

`β₂ = 0.999`（更慢的滑动平均）

**`s` 是什么？** 它衡量这个参数最近梯度的"波动大小"（实际上是未中心化的方差）。
- s 大 → 这个参数梯度波动大 → 用小步走
- s 小 → 这个参数梯度稳定且小 → 用大步走

`s` 叫做**二阶矩**（second moment），因为它是梯度平方的期望。

##### 第 3 层：合并 — Adam！

Adam = Momentum + RMSprop：

```
m = β₁ · m + (1 - β₁) · g       ← 一阶矩
v = β₂ · v + (1 - β₂) · g²      ← 二阶矩
w = w - lr · m / (√v + ε)
```

#### 偏差修正（Bias Correction）

**问题**：m 和 v 初始化为 0。在训练初期它们会**偏向 0**（因为还没累积够梯度）。

**修正**：
```
m̂ = m / (1 - β₁ᵗ)    ← t 是当前 step 数
v̂ = v / (1 - β₂ᵗ)
```

举例（β₁ = 0.9）：
- t=1: 修正系数 = 1 / (1 - 0.9¹) = 10 → m 被放大 10 倍
- t=10: 1 / (1 - 0.9¹⁰) = 1 / 0.65 ≈ 1.5 → 修正变小
- t=∞: 1 / (1 - 0) = 1 → 不再修正

**效果**：训练早期更新步子不会因为 m、v 太小而偏小，让训练一开始就有合理速度。

#### 完整 Adam 算法

```python
# 初始化
m, v = 0, 0
t = 0

# 每个 batch
for batch in data:
    t += 1
    g = compute_gradient(loss, w)
    
    m = β₁ · m + (1 - β₁) · g
    v = β₂ · v + (1 - β₂) · g²
    
    m_hat = m / (1 - β₁^t)        # 偏差修正
    v_hat = v / (1 - β₂^t)
    
    w = w - lr · m_hat / (√v_hat + ε)
```

#### 默认超参（一般不用调）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `lr` | 1e-3 | 全局学习率 |
| `β₁` | 0.9 | 一阶矩衰减 |
| `β₂` | 0.999 | 二阶矩衰减 |
| `ε` | 1e-7 | 防止除零 |

#### Adam vs SGD：什么时候选哪个？

| 场景 | 推荐 |
|---|---|
| 中等数据集 + 复杂任务 | **Adam** |
| 你不想调超参 | **Adam** |
| 项目原型 / 快速实验 | **Adam** |
| 大数据集 + 想榨干性能 | **SGD + Momentum**（精调可能更好） |
| 计算机视觉的 SOTA | 通常 SGD + Momentum |

我们 NBA 项目用 Adam — 数据量小，不值得花时间精调 SGD。

#### 变种：AdamW（更现代的选择）

Adam 把 L2 正则和动量"耦合"了，导致正则化效果不如预期。AdamW 修正了这个问题：
```python
optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
```

如果你想再优化，把 Adam 换成 AdamW 通常能再涨一点。

---

### 9.5 替代架构 — 残差连接、宽 vs 深

#### 为什么不能无脑加深？

直觉：加更多层 → 表达能力更强 → 效果更好？

**实际上**：超过一定深度（比如 20+ 层），训练会变**更差**，不是过拟合，而是**训不动**。

这叫**退化问题（degradation problem）**。即使理论上深层网络至少能做到浅层网络的效果（多余层学恒等映射），实际优化做不到。

#### ResNet 的革命：残差连接（Skip Connection）

何凯明 2015 年的 ResNet：在普通层基础上加一个**捷径**：

```
传统：    x → [Layer] → y
ResNet：  x → [Layer] → +  → y
          └────────────┘
            （捷径直连）
```

数学上：
```
传统：    y = F(x)
ResNet：  y = F(x) + x       ← 加上原始输入
```

#### 为什么这么简单的改动这么有效？

**1. 梯度可以直接流回前面**

反向传播时：
```
∂L/∂x = ∂L/∂y · ∂y/∂x
      = ∂L/∂y · (∂F/∂x + 1)    ← 多了个 +1！
```

那个 `+1` 保证梯度至少能传过去 → 大大缓解梯度消失。

**2. 学起来更容易**

如果某一层最优解就是恒等变换（不变），传统网络要学到 F(x) ≈ x（难）。ResNet 只要学 F(x) ≈ 0（容易，权重置零即可）。

#### 在我们项目里加残差连接

虽然 3 层网络不需要 ResNet（深度不够），但作为练习，可以这么改 `model.py`：

```python
def build_nn_resnet(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    
    # 第一块
    x = keras.layers.Dense(64)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    
    # 残差块 1
    shortcut = x
    y = keras.layers.Dense(64)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)
    y = keras.layers.Dense(64)(y)
    y = keras.layers.BatchNormalization()(y)
    x = keras.layers.Add()([shortcut, y])    # ← 残差连接
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # 残差块 2（同上）
    # ...
    
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, output)
```

注意：残差连接要求**输入和输出维度相同**（或用 1x1 卷积/Dense 调整）。

#### 宽（Wide）vs 深（Deep）

固定参数预算的情况下，怎么分配更好？

| 架构 | 参数 | 优点 | 缺点 |
|---|---|---|---|
| **宽**：1 层 1024 神经元 | ~35K | 表达浅层组合好 | 缺乏抽象层次 |
| **深**：5 层 64 神经元 | ~17K | 抽象层次丰富 | 难训练 |
| **金字塔**：128→64→32 | ~15K | 平衡 | 我们用的 ✅ |

**经验法则**：
- 数据有清晰的层次结构（图像、语言）→ 深网络
- 数据是表格特征 → **宽 + 浅**通常够用
- NBA 这种几十特征的表格数据 → 不需要 ResNet，3 层金字塔正好

#### 更适合表格数据的高级架构

如果你想超越普通 NN：

##### TabNet
专门为表格设计的网络，用注意力机制选特征：
```python
# pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
```

##### 树模型 + NN 集成
**XGBoost** 在表格数据上经常打败神经网络。终极方案是把 XGBoost 和 NN 的预测平均：

```python
final_prob = 0.6 * xgb_prob + 0.4 * nn_prob
```

这种集成（ensemble）通常比任一单一模型好 1-3% AUC。Kaggle 比赛冠军方案 80% 都是集成。

##### Wide & Deep
Google 提出的架构，给推荐系统用：
- **Wide 部分**：直接的特征交叉（线性，记忆能力）
- **Deep 部分**：嵌入 + DNN（泛化能力）
- 输出：两部分加权求和

#### 给你的建议

作为作品集，当前 3 层 NN + LR 基准就够了。如果要展示更强的工程能力，**最简单的扩展是加 XGBoost 做集成**：

```python
# 在 train.py 里加：
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, eval_metric="auc",
)
xgb_model.fit(X_train, y_train)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# 集成
ensemble_prob = 0.5 * nn_prob + 0.5 * xgb_prob
```

XGBoost 在 NBA 这种表格数据上经常 AUC 比 NN 高 0.01-0.02。集成后再涨一点。

面试时这能说："我用了 NN + XGBoost 集成，比单一神经网络 AUC 提升约 0.02" → 加分项。

---

## Part 10: 季后赛 Bracket 模拟器

📄 文件：[src/playoffs.py](src/playoffs.py) ｜ [playoffs.py](playoffs.py) (CLI)

> 单场预测做完，怎么从"单场 64% 主场胜率"推到"NYK 53% 夺冠"？这一节讲清楚。

### 10.1 问题：从单场到系列再到 bracket

单场预测告诉你**一场比赛**的胜率。但季后赛是：
- **七场四胜**（best-of-7）
- 一支队伍要赢**四轮**才夺冠
- 每轮的对手都不一样

模型不能直接吐出"夺冠概率"，因为：
- 它没见过完整 bracket 的训练数据
- 季后赛对手依赖于前几轮谁赢谁输

→ **解法：用单场概率 + 蒙特卡洛模拟整个 bracket**。

### 10.2 第一层：单场 → 系列概率

NBA 季后赛系列采用 **2-2-1-1-1 主场模式**：
- 高种子主场：Game 1, 2, 5, 7
- 低种子主场：Game 3, 4, 6

代码里这就是：
```python
SERIES_HOME_PATTERN = [True, True, False, False, True, False, True]
# True = 高种子主场
```

#### 系列模拟算法

```python
def series_win_prob(higher, lower, predictor, n_sims=5000):
    # 拿到两个单场胜率（主客可能不对称）
    p_higher_at_home = predictor.predict(higher, lower)   # 高种子主场胜率
    p_higher_away    = 1 - predictor.predict(lower, higher)  # 高种子客场胜率
    
    wins = 0
    for _ in range(n_sims):
        higher_score, lower_score = 0, 0
        for game_idx, is_higher_home in enumerate(SERIES_HOME_PATTERN):
            p = p_higher_at_home if is_higher_home else p_higher_away
            if random() < p:
                higher_score += 1
            else:
                lower_score += 1
            if higher_score == 4 or lower_score == 4:
                break  # 系列结束
        if higher_score == 4:
            wins += 1
    return wins / n_sims
```

**关键设计**：
1. **同一对手的主/客场单场概率分开预测** — 因为模型本来就是按 home/away 的视角，不能假设对称
2. **逐场抽样直到 4 胜** — 模拟"提前结束"的情况（4-0、4-1、4-2、4-3 都可能）

#### 直觉：为什么 best-of-7 放大优势

```
单场胜率 p = 55%
系列胜率 ≈ ?
```

直觉答案"系列胜率 ≈ 55%"是错的。best-of-7 让小优势变大：
- p = 50% → 系列 50%
- p = 55% → 系列 ≈ 62%
- p = 60% → 系列 ≈ 73%
- p = 65% → 系列 ≈ 82%

这是因为多打几场，方差被平均掉，"稍微强一点"的队会更稳定地胜出。NBA 主队胜率历史 ~57%，所以哪怕只有一点点优势，季后赛也会比常规赛"明显"。

### 10.3 第二层：系列 → 完整 bracket

bracket 有 16 队、4 轮、15 个系列。完整模拟一遍：

```
Round 1 (4 系列 × 2 会区 = 8 系列)
   ↓
Round 2 (2 系列 × 2 会区 = 4 系列)
   ↓
Conference Finals (1 系列 × 2 会区 = 2 系列)
   ↓
NBA Finals (1 系列)
   ↓
Champion
```

#### 蒙特卡洛的关键：每次模拟是 ONE 种可能的 bracket 走向

```python
for sim_idx in range(N_simulations):
    # 模拟 R1
    east_r2_teams = []
    for higher, lower in east_r1_matchups:
        p = series_win_prob(higher, lower)
        winner = higher if random() < p else lower
        east_r2_teams.append(winner)
    
    # 模拟 R2: 谁打谁取决于 R1 结果
    east_r3_teams = pair_and_play(east_r2_teams, ...)
    
    # 模拟会区决赛
    east_champ = conf_final(east_r3_teams)
    west_champ = conf_final(west_r3_teams)
    
    # 模拟总决赛
    champion = nba_finals(east_champ, west_champ)
    
    counts[champion] += 1

# 最终概率 = 计数 / N
```

跑 10,000 次，每队夺冠次数 / 10,000 = 夺冠概率。

#### 性能优化：缓存单场概率

每次模拟里调用 `predictor.predict(A, B)` 是慢的（神经网络前向传播）。但**同一对球队的单场胜率永远相同**（特征不变）。

→ 加缓存：第一次算出后存下来，后面直接查表。

```python
class MatchupPredictor:
    def __init__(self):
        self.model = load_model()
        self._cache = {}  # (home, away) → prob
    
    def predict(self, home, away):
        key = (home, away)
        if key not in self._cache:
            self._cache[key] = self._compute(home, away)
        return self._cache[key]
```

效果：10,000 次模拟里只算 ~30 个唯一对决（每个对决两次方向），总单场预测次数从理论 ~150,000 降到约 60，**快了 2500 倍**。

### 10.4 第三层：从中间状态继续（`--from-round 2`）

季后赛进行到一半，**已经发生的事不该重新模拟**。比如：
- NYK 已经 4-0 横扫 PHI → 应该直接进东决
- DET vs CLE 2-2 → 只剩 3 场抢

#### `SeriesState` 数据类

```python
@dataclass
class SeriesState:
    higher_seed: str
    lower_seed: str
    higher_wins: int = 0
    lower_wins: int = 0
    
    @property
    def is_over(self):
        return self.higher_wins == 4 or self.lower_wins == 4
    
    @property
    def games_played(self):
        return self.higher_wins + self.lower_wins
```

#### `simulate_remaining`：从当前比分模拟剩下的比赛

```python
def simulate_remaining(state, predictor, n_sims):
    if state.is_over:
        return 1.0 if state.winner == state.higher_seed else 0.0
    
    p_home, p_away = get_probabilities(state)
    wins = 0
    for _ in range(n_sims):
        h, l = state.higher_wins, state.lower_wins  # 从当前比分开始
        for game_idx in range(state.games_played, 7):  # 从下一场开始
            higher_is_home = SERIES_HOME_PATTERN[game_idx]
            p = p_home if higher_is_home else p_away
            if random() < p:
                h += 1
            else:
                l += 1
            if h == 4 or l == 4:
                break
        if h == 4:
            wins += 1
    return wins / n_sims
```

关键点：
- `range(state.games_played, 7)` — 跳过已经打完的场次
- `SERIES_HOME_PATTERN[game_idx]` — 看当前是第几场，决定谁是主场

#### CLI 用法

```bash
python playoffs.py --from-round 2 \
  --r2-east "NYK:PHI:4-0" "DET:CLE:2-2" \
  --r2-west "OKC:LAL:4-0" "SAS:MIN:2-2"
```

格式：`HIGHER:LOWER:W-L`，比如 `NYK:PHI:4-0` 意为"NYK 4-0 击败 PHI"。

### 10.5 总决赛的特殊处理：中性主场

R1-R3 的主场基于种子。但总决赛的"主场"取决于常规赛战绩，跨会区比较。

实现里用了一个折中：**取双向预测的平均**：

```python
p_east_at_home = series_prob(east_champ, west_champ)   # 东冠主场
p_west_at_home = series_prob(west_champ, east_champ)   # 西冠主场
p_east_wins = (p_east_at_home + (1 - p_west_at_home)) / 2
```

这样能消除主场偏置，得到中性比较。

### 10.6 输出概率的解读

跑完 `playoffs.py --from-round 2`，会看到：

```
👑 NBA Champion
   1. NYK   53.1%
   2. OKC   26.1%
   3. SAS   17.5%
```

#### 这些数字怎么算的？

```
NYK 夺冠 = NYK 赢 R2 × NYK 赢东决 × NYK 赢总决赛
       = 100% × 87.9% × (87.9% 中又赢的比例)
       = 53.1%
```

每个百分点都来自 10,000 次蒙特卡洛抽样里的实际出现频率。

#### 概率必须自洽

```
所有队伍的夺冠概率之和 = 100%
53.1% + 26.1% + 17.5% + 2.4% + 0.7% + 0.2% = 100.0% ✓
```

如果不自洽（比如加起来 98%），说明代码有 bug。

### 10.7 模拟器的局限

1. **模型只看常规赛** → 不知道"季后赛节奏"（更慢、防守更紧、轮换更短）
2. **伤病假设** → 当前模拟假设所有队 5/5 明星都在
3. **历史样本** → 上次 SAS 进总决赛是很多年前，模型对马刺的预测可能保守

要克服这些，下一步：
- 加入实时伤病数据（手动 / 抓 Rotoworld）
- 用季后赛比赛单独训练（或用迁移学习）
- 加入对手特定的"防守效率"特征

---

## Part 11: Basketball Reference 备用数据源

📄 文件：[src/data_fetch_bref.py](src/data_fetch_bref.py)

> NBA 官方 API (`stats.nba.com`) 在国内访问不稳定。这一节讲我们怎么用 Basketball Reference 作为备用。

### 11.1 为什么需要备用数据源

`nba_api` 这个库直接连接 `stats.nba.com`，这个域名在中国大陆经常：
- 连接被强制断开（`ConnectionResetError`）
- 超时（`ReadTimeout`）
- 速率限制更严

`basketball-reference.com` 在美国其他基础设施上，**通常可访问**。它是 NBA 数据爱好者的"圣经"网站，从 1946 年开始的所有比赛数据都有。

### 11.2 BR 的数据组织方式

URL 模式：
```
https://www.basketball-reference.com/leagues/NBA_{end_year}_games-{month}.html
```

例子：
- `NBA_2024_games-october.html` → 2023-24 赛季 10 月的比赛
- `NBA_2026_games-may.html` → 2025-26 赛季 5 月（含季后赛）

注意：BR 用**赛季结束年**命名。2023-24 赛季 → `NBA_2024`。

### 11.3 抓取技巧：`pandas.read_html`

不需要写复杂爬虫。pandas 内置了 HTML 表格解析：

```python
import pandas as pd
tables = pd.read_html("https://www.basketball-reference.com/leagues/NBA_2024_games-october.html")
df = tables[0]  # 第一个表格就是赛程
```

返回的列：
```
['Date', 'Start (ET)', 'Visitor/Neutral', 'PTS', 'Home/Neutral', 'PTS.1',
 'Arena', 'Notes', ...]
```

关键列：
- `Date`：比赛日期
- `Visitor/Neutral`：客队全称（如 `"Los Angeles Lakers"`）
- `PTS`：客队得分
- `Home/Neutral`：主队全称
- `PTS.1`：主队得分（pandas 自动给重复列名加 `.1`）

### 11.4 数据格式统一

BR 返回的格式跟 `nba_api` 不一样。我们要做"适配"，让下游代码不需要知道数据来自哪。

```python
def _to_game_table(raw):
    nba_teams = nba_teams_module.get_teams()
    name_to_id = {t["full_name"]: t["id"] for t in nba_teams}
    
    df = raw.copy()
    df["GAME_DATE"]    = pd.to_datetime(df["Date"], errors="coerce")
    df["home_pts"]     = df["PTS.1"].astype(int)
    df["away_pts"]     = df["PTS"].astype(int)
    df["home_team_id"] = df["Home/Neutral"].map(name_to_id)
    df["away_team_id"] = df["Visitor/Neutral"].map(name_to_id)
    df["home_win"]     = (df["home_pts"] > df["away_pts"]).astype(int)
    df["GAME_ID"]      = [f"BR{i:06d}" for i in range(len(df))]
    return df[["GAME_ID", "GAME_DATE", ...]]  # 跟 nba_api 输出列对齐
```

下游的特征工程 / 训练 / 预测代码**完全不变**，因为数据格式一致。

### 11.5 礼貌延迟

爬数据要尊重对方服务器。BR 文档说"1 请求 / 3 秒"是上限。我们设：

```python
BR_DELAY_SECONDS = 3.0

for month in months:
    df = _fetch_month_table(end_year, month)
    time.sleep(BR_DELAY_SECONDS)
```

4 个赛季 × 9 个月（含季后赛 5/6 月） = 36 次请求 × 3 秒 ≈ **2 分钟拉完整套数据**。

### 11.6 关键设计：抓季后赛月份

之前代码只抓 `october-april` 7 个常规赛月。但 NBA 季后赛在 5/6 月，**那些比赛也很重要**：
- 影响 ELO 评分（顶级球队季后赛胜场强化它们的当前 ELO）
- 让 `predict.py` 看到的"最近 10 场"包含季后赛实战

所以代码扩展为：

```python
REGULAR_SEASON_MONTHS = ["october", ..., "april"]
PLAYOFF_MONTHS = ["may", "june"]
ALL_MONTHS = REGULAR_SEASON_MONTHS + PLAYOFF_MONTHS
```

加完后，从 2026-04-30 数据扩展到 **2026-05-09**（季后赛 R1 完整 + R2 进行中）。

### 11.7 切换数据源

`main.py` 加了 `--source` 开关：

```bash
python main.py                  # 默认走 nba_api（如能访问）
python main.py --source bref    # 切换到 Basketball Reference
```

在 `main.py` 里：
```python
if args.source == "bref":
    from src.data_fetch_bref import fetch_all_seasons_bref
    games = fetch_all_seasons_bref()
else:
    # 原来的 nba_api 路径
    ...
```

### 11.8 BR 模式下的功能差异

| 功能 | nba_api | bref |
|---|:---:|:---:|
| 比赛数据 | ✅ | ✅ |
| 球员场次数据 | ✅ | ❌ |
| 当日赛程查询 | ✅ | ❌ |
| 伤病代理特征 | ✅ | ⚠️ 自动关闭 |
| 速度 | 快（API） | 慢（爬网页）|

代码会在 BR 模式下**自动把 USE_INJURY_FEATURES 设为 False**，因为 BR 没有干净的球员出场数据。其他功能完全等价。

### 11.9 你可以学到的教训

1. **永远准备 fallback** — 不依赖单一数据源
2. **数据格式抽象** — 让下游代码看到统一接口
3. **礼貌爬虫** — 加延迟，看 robots.txt，不要 DDoS 别人
4. **数据完整性 > 数据新鲜度** — 拿到 4 个赛季完整数据比拿到上周比赛但缺周二更有价值

---

## 学习路径建议

如果你想深入理解某一块：

| 想学 | 推荐资源 |
|---|---|
| 神经网络数学 | 3Blue1Brown 的 YouTube 系列（有中文字幕） |
| TensorFlow/Keras | 官方教程 [tensorflow.org/tutorials](https://www.tensorflow.org/tutorials) |
| 特征工程 | Kaggle 课程 "Feature Engineering"（免费） |
| 评估指标 | scikit-learn 文档的 model_evaluation 章节 |
| 时间序列预测 | "Forecasting: Principles and Practice"（在线免费书） |

---

## 给作品集面试用的一句话总结

> 这是一个端到端的 NBA 比赛预测项目，从 nba_api 抓 3 个赛季的数据、做防泄露的滚动特征工程、用 TensorFlow 训练一个三层神经网络做二分类，并用 5 个指标和 5 张图对比逻辑回归基准。最终神经网络达到约 64% 准确率和 0.685 AUC，超过主队基础胜率约 5 个百分点。

记住这段话，面试时直接说。
