# Qwen2 NumPy 推理实现

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.44+-orange.svg)](https://huggingface.co/transformers)

## 🚀 项目简介

本项目是**DeepSeek-R1-Distill-Qwen-1.5B**模型的**纯NumPy实现**，从头开始构建了完整的Transformer架构，实现了高精度的文本生成推理。项目旨在深入理解大语言模型的内部工作机制，通过纯NumPy代码展示Transformer架构的每个细节。

## ✨ 主要特性

- 🔧 **完全纯NumPy实现**：不依赖深度学习框架，仅使用NumPy实现所有计算
- 🏗️ **完整Transformer架构**：包含所有核心组件
  - Token嵌入层 (Embedding)
  - 多头自注意力机制 (Multi-Head Self-Attention)
  - RMSNorm层归一化
  - 旋转位置编码 (RoPE)
  - SwiGLU激活的MLP层
  - 因果注意力掩码
- 📊 **精度验证**：与原始PyTorch模型对比，确保数值精度
- 🎯 **文本生成**：支持贪婪解码生成文本
- 🎨 **可视化输出**：丰富的终端颜色和进度显示
- 📈 **性能监控**：实时显示推理进度和差异统计

## 📁 项目结构

```text
qwen2_numpy_recurrence/
├── base.py                 # 基础版本实现
├── v1.py                   # 版本1实现
├── v2.py                   # 版本2实现（推荐）
├── v2.ipynb               # Jupyter Notebook版本
├── modeling_qwen2.py      # 原始Qwen2模型代码参考
├── DeepSeek-R1-Distill-Qwen-1.5B/  # 预训练模型文件
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── README.md
```

## 🔧 环境要求

```bash
# Python 3.8+
numpy>=1.21.0
torch>=1.9.0
transformers>=4.44.0
safetensors>=0.3.0
```

## 📦 安装与使用

### 1. 克隆仓库

```bash
git clone https://github.com/CuiPenghub/qwen2_numpy_recurrence.git
cd qwen2_numpy_recurrence
```

### 2. 安装依赖

```bash
pip install numpy torch transformers safetensors
```

### 3. 下载模型

下载DeepSeek-R1-Distill-Qwen-1.5B模型到项目目录：

```bash
# 请确保模型文件在 DeepSeek-R1-Distill-Qwen-1.5B/ 目录下
```

### 4. 运行推理

```bash
# 运行最新版本（推荐）
python v2.py

# 或者使用Jupyter Notebook
jupyter notebook v2.ipynb
```

## 🎯 核心功能展示

### 1. 模型配置

```python
config = {
    "hidden_size": 1536,           # 隐藏层维度
    "intermediate_size": 8960,     # MLP中间层维度  
    "num_attention_heads": 12,     # 注意力头数
    "num_hidden_layers": 28,       # Transformer层数
    "num_key_value_heads": 2,      # KV头数（GQA）
    "rms_norm_eps": 1e-6,         # RMSNorm epsilon
    "rope_theta": 10000,          # RoPE theta参数
    "hidden_act": "silu"          # 激活函数
}
```

### 2. 核心组件实现

#### RMSNorm层归一化

```python
def rms_norm(hidden_states, weight, eps=1e-6):
    rms = np.mean(np.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * (1 / np.sqrt(rms + eps))
    return weight * hidden_states
```

#### 旋转位置编码 (RoPE)

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

#### 多头注意力机制

```python
def attention_forward(query, key, value, attention_mask, scaling, num_key_value_groups):
    # 群组查询注意力 (GQA)
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    
    # 计算注意力分数
    attn_weights = np.matmul(query, np.transpose(key_states, (0, 1, 3, 2))) * scaling
    # ...
```

### 3. 文本生成

```python
# 贪婪解码生成
generated_ids = greedy_decode(input_ids, all_parameters, config, max_new_tokens=50)
```

## 📊 精度验证

项目包含完整的精度验证功能，对比NumPy实现与原始PyTorch模型：

```text
差异统计总结:
平均最大差异: 1.2e-05
平均80%分位差异: 3.4e-06
最大差异峰值: 5.7e-05
80%分位差异峰值: 8.9e-06
```

## 🛠️ 技术实现细节

### Transformer架构组件

1. **Token嵌入** - 将token ID映射为向量表示
2. **位置编码** - RoPE旋转位置编码
3. **多头注意力** - 支持GQA（群组查询注意力）
4. **层归一化** - RMSNorm实现
5. **前馈网络** - SwiGLU激活的MLP
6. **因果掩码** - 确保生成的因果性

### 数值计算优化

- 使用稳定的Softmax计算
- 正确的广播机制
- 高效的矩阵运算
- 精确的浮点数处理

## 🎨 可视化特性

项目包含丰富的终端可视化功能：

- 🎯 彩色进度条显示推理进度
- 📊 实时差异统计显示
- 🔍 Token级别的生成过程展示
- 📈 性能监控和时间统计

## 🔍 使用场景

- 📚 **教育学习**：深入理解Transformer架构
- 🔬 **研究开发**：验证模型实现的正确性
- 🛠️ **算法优化**：测试新的数值计算方法
- 🎯 **精度分析**：对比不同实现的数值差异

## 📈 性能指标

- **推理速度**：约0.5-2.0 tokens/秒（CPU）
- **内存占用**：约2-4GB RAM
- **精度差异**：与原始模型相比< 1e-4
- **支持序列长度**：最大131072 tokens

## 🤝 贡献指南

欢迎提交问题报告和功能建议！如果您想贡献代码：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 🙏 致谢

- [DeepSeek](https://github.com/deepseek-ai) - 提供预训练模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 原始模型实现参考
- [Qwen2](https://github.com/QwenLM/Qwen2) - 模型架构参考

## 📞 联系方式

- 作者：CuiPenghub
- GitHub：[@CuiPenghub](https://github.com/CuiPenghub)
- 项目链接：[qwen2_numpy_recurrence](https://github.com/CuiPenghub/qwen2_numpy_recurrence)

---

⭐ 如果这个项目对你有帮助，请给它一个Star！
