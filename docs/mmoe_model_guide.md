# Multi-gate Mixture-of-Experts (MMoE) Model 指南

## 📋 概述

本项目实现了基于论文 [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) 的 MMoE 模型。MMoE 是一个强大的多任务学习架构，特别适用于推荐系统场景。

## 🏗️ 架构设计

### 核心组件

1. **专家网络 (Expert Networks)**
   - 多个共享的专家网络，每个专家捕获输入的不同方面
   - 支持可配置的隐藏层维度和 dropout

2. **门控网络 (Gate Networks)**
   - 每个任务一个门控网络
   - 动态计算专家权重，为不同任务分配不同的专家组合

3. **塔网络 (Tower Networks)**
   - 任务特定的网络，基于专家混合输出做最终预测
   - 每个任务都有独立的塔网络

### 模型类层次结构

```
MMoEModel (基础模型)
├── ExpertNetwork (专家网络)
├── GateNetwork (门控网络)
└── TowerNetwork (塔网络)

MMoERecommendationModel (推荐系统专用模型)
└── 继承自 MMoEModel，添加用户/物品嵌入层
```

## 🚀 快速开始

### 基础 MMoE 模型使用

```python
from model.mmoe import MMoEModel
import torch

# 创建基础 MMoE 模型
model = MMoEModel(
    input_dim=128,           # 输入特征维度
    num_experts=3,           # 专家网络数量
    num_tasks=2,             # 任务数量
    expert_hidden_dims=[64, 32],    # 专家网络隐藏层
    tower_hidden_dims=[32, 16]      # 塔网络隐藏层
)

# 前向传播
x = torch.randn(32, 128)  # (batch_size, input_dim)
predictions = model(x)    # 返回任务预测列表

# 获取专家权重（用于分析）
predictions, expert_weights = model(x, return_expert_weights=True)
```

### 推荐系统专用模型

```python
from model.mmoe import MMoERecommendationModel

# 创建推荐系统 MMoE 模型
rec_model = MMoERecommendationModel(
    user_cardinality=10000,         # 用户数量
    item_cardinality=50000,         # 物品数量
    embedding_dim=64,               # 嵌入维度
    num_dense_features=8,           # 稠密特征数量
    num_experts=4,                  # 专家数量
    num_tasks=2                     # 任务数量（如CTR和评分预测）
)

# 前向传播
user_ids = torch.randint(0, 10000, (32,))
item_ids = torch.randint(0, 50000, (32,))
dense_features = torch.randn(32, 8)

predictions = rec_model(user_ids, item_ids, dense_features)
```

## 📊 MovieLens 多任务学习示例

### 运行训练示例

```bash
# 安装依赖
pip install torch pandas numpy

# 运行 MMoE 训练示例
python movie_len_mmoe_example.py
```

### 示例说明

我们的示例在 MovieLens 数据集上同时预测两个任务：

1. **CTR 预测** (二分类)：预测用户是否会点击/喜欢某个电影
2. **评分预测** (回归)：预测用户对电影的具体评分

### 关键特性

- **多任务损失函数**：使用不确定性加权平衡不同任务的损失
- **专家权重分析**：可视化不同任务如何利用专家网络
- **完整训练循环**：包含训练、验证和评估

## 🔧 模型配置

### 超参数说明

| 参数 | 描述 | 推荐值 |
|------|------|--------|
| `num_experts` | 专家网络数量 | 3-8 |
| `expert_hidden_dims` | 专家网络隐藏层维度 | [128, 64] |
| `gate_hidden_dims` | 门控网络隐藏层（None为线性门控） | None 或 [32] |
| `tower_hidden_dims` | 塔网络隐藏层维度 | [64, 32] |
| `dropout_rate` | Dropout 比率 | 0.1-0.3 |

### 任务配置

```python
# 多任务输出配置
task_output_dims = [1, 1]  # 每个任务的输出维度

# 创建模型
model = MMoEModel(
    input_dim=input_dim,
    num_tasks=len(task_output_dims),
    task_output_dims=task_output_dims,
    # ... 其他参数
)
```

## 📈 训练最佳实践

### 1. 多任务损失设计

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # 学习任务特定的权重
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, predictions, targets):
        # 不确定性加权
        weighted_losses = []
        for i in range(len(predictions)):
            precision = torch.exp(-self.log_vars[i])
            loss = task_loss_function(predictions[i], targets[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)
```

### 2. 学习率调度

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
```

### 3. 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 🔍 模型分析

### 专家利用分析

```python
def analyze_expert_weights(model, dataloader):
    model.eval()
    expert_weights_per_task = []
    
    with torch.no_grad():
        for batch in dataloader:
            _, expert_weights = model(batch, return_expert_weights=True)
            expert_weights_per_task.append(expert_weights)
    
    # 分析每个任务的专家权重分布
    for task_id, weights in enumerate(expert_weights_per_task):
        avg_weights = torch.cat(weights, dim=0).mean(dim=0)
        print(f"Task {task_id} expert weights: {avg_weights}")
```

### 性能评估

```python
def evaluate_multitask_model(model, dataloader, tasks):
    model.eval()
    task_metrics = {}
    
    with torch.no_grad():
        for batch in dataloader:
            predictions = model(batch)
            
            for task_id, (pred, target) in enumerate(zip(predictions, targets)):
                if tasks[task_id] == 'classification':
                    accuracy = compute_accuracy(pred, target)
                    task_metrics[f'task_{task_id}_acc'] = accuracy
                elif tasks[task_id] == 'regression':
                    mse = compute_mse(pred, target)
                    task_metrics[f'task_{task_id}_mse'] = mse
    
    return task_metrics
```

## 🎯 应用场景

### 推荐系统

- **CTR 预测 + 转化率预测**
- **点击预测 + 停留时间预测**
- **评分预测 + 购买预测**

### 其他多任务场景

- **文本分类 + 情感分析**
- **图像分类 + 目标检测**
- **语音识别 + 说话人识别**

## 🔧 自定义扩展

### 添加新的专家网络类型

```python
class ConvolutionalExpert(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.conv_layers(x)
```

### 自定义门控机制

```python
class AttentionGate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)
        self.gate_projection = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        attended_x, _ = self.attention(x, x, x)
        gate_weights = F.softmax(self.gate_projection(attended_x), dim=-1)
        return gate_weights
```

## 📚 参考资料

1. [原始论文](https://dl.acm.org/doi/10.1145/3219819.3220007): Ma, J., et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts." KDD 2018.

2. **相关工作**:
   - PLE (Progressive Layered Extraction)
   - ESMM (Entire Space Multi-Task Model)
   - MMOE 在工业推荐系统中的应用

3. **实现参考**:
   - TensorFlow 版本: [官方实现](https://github.com/drawbridge/keras-mmoe)
   - PyTorch 社区实现

## 🛠️ 故障排除

### 常见问题

1. **专家权重不平衡**
   - 检查学习率设置
   - 考虑添加专家平衡损失

2. **某个任务性能差**
   - 调整任务权重
   - 检查数据标签质量

3. **训练不稳定**
   - 降低学习率
   - 增加梯度裁剪
   - 检查批次大小

### 性能优化

1. **内存优化**
   - 使用混合精度训练
   - 适当的批次大小
   - 梯度累积

2. **计算优化**
   - 专家网络参数共享
   - 动态专家选择

## 🎉 总结

MMoE 模型为多任务学习提供了一个灵活且强大的架构。通过共享专家网络和任务特定的门控机制，它能够有效地建模任务间的关系，在推荐系统等场景中取得优异的性能。

本实现提供了完整的训练流程、分析工具和扩展接口，可以轻松适配到不同的多任务学习场景中。 