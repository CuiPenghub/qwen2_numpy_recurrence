import numpy as np
import torch
from transformers import AutoTokenizer
import safetensors
import os
import time
import sys

# 添加终端颜色和格式化工具
class TerminalFormatter:
    # ANSI颜色代码
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
    }
    
    # 样式代码
    STYLES = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        'reverse': '\033[7m',
        'hidden': '\033[8m',
    }
    
    # 重置代码
    RESET = '\033[0m'
    
    @staticmethod
    def colorize(text, color=None, style=None):
        """为文本添加颜色和样式"""
        result = ""
        if color and color in TerminalFormatter.COLORS:
            result += TerminalFormatter.COLORS[color]
        if style and style in TerminalFormatter.STYLES:
            result += TerminalFormatter.STYLES[style]
        result += text
        result += TerminalFormatter.RESET
        return result
    
    @staticmethod
    def header(text, width=80):
        """创建带有边框的标题"""
        border = "═" * width
        return f"\n{TerminalFormatter.colorize(border, 'bright_blue')}\n{TerminalFormatter.colorize('▶ ' + text.center(width-4), 'bright_cyan', 'bold')}\n{TerminalFormatter.colorize(border, 'bright_blue')}\n"
    
    @staticmethod
    def section(text):
        """创建小节标题"""
        return f"\n{TerminalFormatter.colorize('┌─ ' + text + ' ', 'bright_green', 'bold')}\n"
    
    @staticmethod
    def info(text):
        """信息文本"""
        return TerminalFormatter.colorize(f"ℹ️ {text}", 'bright_white')
    
    @staticmethod
    def success(text):
        """成功文本"""
        return TerminalFormatter.colorize(f"✓ {text}", 'bright_green')
    
    @staticmethod
    def warning(text):
        """警告文本"""
        return TerminalFormatter.colorize(f"⚠️ {text}", 'bright_yellow')
    
    @staticmethod
    def error(text):
        """错误文本"""
        return TerminalFormatter.colorize(f"✗ {text}", 'bright_red')
    
    @staticmethod
    def progress_bar(current, total, prefix='', suffix='', length=30):
        """创建进度条"""
        filled_length = int(length * current // total)
        bar = '█' * filled_length + '░' * (length - filled_length)
        percentage = f"{100 * current / total:.1f}%"
        return f"\r{prefix} {TerminalFormatter.colorize(bar, 'bright_blue')} {TerminalFormatter.colorize(percentage, 'bright_yellow')} {suffix}"
    
    @staticmethod
    def token_display(token_text):
        """美化token显示"""
        # 替换空格为可见字符，使空格可见
        visible_text = token_text.replace(' ', '␣')
        return TerminalFormatter.colorize(f"[{visible_text}]", 'bright_magenta')

# 辅助函数：比较两个数组的差异
def judge_two_arrays(a, b):
    diff = np.abs(a - b)
    max_diff = np.max(diff)
    p80 = np.percentile(diff, 80)
    return max_diff, p80

# 比较numpy实现和原始库函数的推理结果
def compare_inference_results(input_ids, all_parameters, config):
    print(TerminalFormatter.section("比较推理结果"))
    
    # 使用numpy实现进行推理
    print(TerminalFormatter.info("执行numpy实现推理..."))
    numpy_logits = inference(input_ids, all_parameters, config)
    
    # 使用原始库函数进行推理
    print(TerminalFormatter.info("执行原始库函数推理..."))
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path)
    with torch.no_grad():
        torch_outputs = model(input_ids=torch.from_numpy(input_ids))
        torch_logits = torch_outputs.logits.numpy()
    
    # 比较结果
    print(TerminalFormatter.info("比较推理结果..."))
    max_diff, p80_diff = judge_two_arrays(numpy_logits, torch_logits)
    
    print(TerminalFormatter.colorize("差异统计:", 'bright_cyan', 'bold'))
    print(TerminalFormatter.colorize(f"最大差异: {max_diff:.6f}", 'bright_white'))
    print(TerminalFormatter.colorize(f"80%分位数差异: {p80_diff:.6f}", 'bright_white'))
    
    return max_diff, p80_diff

model_path = "e:\\Pro\\DeepSeek-R1-Distill-Qwen-1.5B"

print(TerminalFormatter.header("DeepSeek-R1-Distill-Qwen-1.5B 推理演示", 80))

# 加载tokenizer
print(TerminalFormatter.section("加载Tokenizer"))
print(TerminalFormatter.info("正在加载tokenizer..."))
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(TerminalFormatter.success("Tokenizer加载完成！"))

# 准备输入
print(TerminalFormatter.section("准备输入"))
chat = [
    {"role": "user", "content": "介绍一下你自己"}
]

# 应用聊天模板
print(TerminalFormatter.info("应用聊天模板..."))
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(TerminalFormatter.colorize("格式化后的聊天内容:", 'bright_cyan'))
print(TerminalFormatter.colorize(formatted_chat, 'bright_white'))

# Tokenize输入
print(TerminalFormatter.info("Tokenize输入..."))
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
print(TerminalFormatter.colorize("Tokenized inputs:", 'bright_cyan'))
print(TerminalFormatter.colorize(str(inputs), 'bright_white'))

# 转换为NumPy数组
input_ids = inputs['input_ids'].numpy()
print(TerminalFormatter.colorize(f"Input shape: {input_ids.shape}", 'bright_cyan'))

# 加载模型参数
print(TerminalFormatter.section("加载模型参数"))
print(TerminalFormatter.info("正在加载模型参数..."))
model_file = os.path.join(model_path, "model.safetensors")
all_parameters = safetensors.torch.load_file(model_file)
print(TerminalFormatter.success("模型参数加载完成！"))

# 打印模型参数信息
print(TerminalFormatter.colorize("\n模型参数信息:", 'bright_cyan', 'bold'))
print(TerminalFormatter.colorize("─" * 50, 'bright_blue'))
for key in list(all_parameters.keys())[:10]:  # 只打印前10个参数
    print(TerminalFormatter.colorize(f"{key}: {all_parameters[key].shape}", 'bright_white'))
print(TerminalFormatter.colorize("─" * 50, 'bright_blue'))
print(TerminalFormatter.colorize(f"总参数数量: {len(all_parameters.keys())}", 'bright_yellow', 'bold'))

# 1. 嵌入层 (Embedding)
print(TerminalFormatter.section("1. 嵌入层 (Embedding)"))
embed_tokens_weight = all_parameters['model.embed_tokens.weight'].float().numpy()
hidden_states = embed_tokens_weight[input_ids]
print(TerminalFormatter.colorize(f"Embedding输出形状: {hidden_states.shape}", 'bright_cyan'))

# 2. 开始实现Transformer层
config = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_attention_heads": 12,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "hidden_act": "silu"
}

# 打印模型配置
print(TerminalFormatter.section("模型配置"))
for key, value in config.items():
    print(TerminalFormatter.colorize(f"{key}: {value}", 'bright_white'))

# 实现RMSNorm
def rms_norm(hidden_states, weight, eps=1e-6):
    # 计算均方根
    rms = np.mean(np.square(hidden_states), axis=-1, keepdims=True)
    # 归一化
    hidden_states = hidden_states * (1 / np.sqrt(rms + eps))
    # 应用权重
    return weight * hidden_states

# 实现SiLU激活函数
def silu(x):
    return x * 1/(1 + np.exp(-x))

# 实现旋转位置编码
def get_rotary_embedding(hidden_states, position_ids, config):
    # 获取维度信息
    batch_size, seq_length, _ = hidden_states.shape
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    
    # 计算旋转位置编码的频率
    theta = config["rope_theta"]
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim))
    
    # 计算位置编码
    t = position_ids.reshape(-1, 1)
    freqs = t * inv_freq.reshape(1, -1)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos = np.cos(emb)
    sin = np.sin(emb)
    
    # 调整形状以匹配注意力头的维度
    return cos.reshape(-1, seq_length, head_dim//2*2), sin.reshape(-1, seq_length, head_dim//2*2)

# 实现旋转位置编码的应用
def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return np.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # 确保cos和sin的最后一个维度与head_dim匹配
    head_dim = q.shape[-1]
    if cos.shape[-1] != head_dim:
        # 如果维度不匹配，可能需要填充或截断
        if cos.shape[-1] < head_dim:
            # 填充
            pad_width = head_dim - cos.shape[-1]
            cos = np.pad(cos, ((0,0), (0,0), (0,pad_width)))
            sin = np.pad(sin, ((0,0), (0,0), (0,pad_width)))
        else:
            # 截断
            cos = cos[..., :head_dim]
            sin = sin[..., :head_dim]
    
    # 扩展维度以便广播
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    
    # 应用旋转位置编码
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# 实现重复KV头
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # 扩展并重塑
    hidden_states = np.repeat(hidden_states[:, :, np.newaxis, :, :], n_rep, axis=2)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# 实现注意力机制
def attention_forward(query, key, value, attention_mask, scaling, num_key_value_groups):
    # 重复KV头
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    
    # 计算注意力分数
    attn_weights = np.matmul(query, np.transpose(key_states, (0, 1, 3, 2))) * scaling
    
    # 应用注意力掩码
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    # Softmax
    attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))
    attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-6)
    
    # 应用注意力
    attn_output = np.matmul(attn_weights, value_states)
    attn_output = np.transpose(attn_output, (0, 2, 1, 3))
    
    return attn_output

# 实现MLP
def mlp_forward(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight):
    # SwiGLU激活
    gate_proj = np.matmul(hidden_states, gate_proj_weight.T)
    up_proj = np.matmul(hidden_states, up_proj_weight.T)
    
    gate_act = silu(gate_proj)
    intermediate = gate_act * up_proj
    
    # 下投影
    down_proj = np.matmul(intermediate, down_proj_weight.T)
    
    return down_proj

# 实现单个Transformer层
def transformer_layer_forward(hidden_states, layer_idx, all_parameters, config, position_ids=None, attention_mask=None):
    # 获取层参数前缀
    prefix = f"model.layers.{layer_idx}."
    
    # 输入层归一化
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, 
        all_parameters[prefix + "input_layernorm.weight"].float().numpy(),
        config["rms_norm_eps"]
    )
    
    # 自注意力
    batch_size, seq_length, _ = hidden_states.shape
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    num_key_value_groups = config["num_attention_heads"] // config["num_key_value_heads"]
    
    # 投影到Q, K, V
    q_proj_weight = all_parameters[prefix + "self_attn.q_proj.weight"].float().numpy()
    k_proj_weight = all_parameters[prefix + "self_attn.k_proj.weight"].float().numpy()
    v_proj_weight = all_parameters[prefix + "self_attn.v_proj.weight"].float().numpy()
    o_proj_weight = all_parameters[prefix + "self_attn.o_proj.weight"].float().numpy()
    
    # 可能存在的偏置
    q_proj_bias = all_parameters.get(prefix + "self_attn.q_proj.bias", None)
    k_proj_bias = all_parameters.get(prefix + "self_attn.k_proj.bias", None)
    v_proj_bias = all_parameters.get(prefix + "self_attn.v_proj.bias", None)
    
    if q_proj_bias is not None:
        q_proj_bias = q_proj_bias.float().numpy()
        k_proj_bias = k_proj_bias.float().numpy()
        v_proj_bias = v_proj_bias.float().numpy()
    
    # 计算Q, K, V
    query_states = np.matmul(hidden_states, q_proj_weight.T)
    key_states = np.matmul(hidden_states, k_proj_weight.T)
    value_states = np.matmul(hidden_states, v_proj_weight.T)
    
    if q_proj_bias is not None:
        query_states = query_states + q_proj_bias
        key_states = key_states + k_proj_bias
        value_states = value_states + v_proj_bias
    
    # 重塑为多头形状
    query_states = query_states.reshape(batch_size, seq_length, config["num_attention_heads"], head_dim)
    query_states = np.transpose(query_states, (0, 2, 1, 3))
    
    key_states = key_states.reshape(batch_size, seq_length, config["num_key_value_heads"], head_dim)
    key_states = np.transpose(key_states, (0, 2, 1, 3))
    
    value_states = value_states.reshape(batch_size, seq_length, config["num_key_value_heads"], head_dim)
    value_states = np.transpose(value_states, (0, 2, 1, 3))
    
    # 应用旋转位置编码
    if position_ids is None:
        position_ids = np.arange(seq_length).reshape(1, -1)
    
    cos, sin = get_rotary_embedding(hidden_states, position_ids, config)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # 计算注意力
    scaling = 1.0 / np.sqrt(head_dim)
    attn_output = attention_forward(
        query_states, key_states, value_states, attention_mask, scaling, num_key_value_groups
    )
    
    # 重塑并投影回原始维度
    attn_output = attn_output.reshape(batch_size, seq_length, config["hidden_size"])
    attn_output = np.matmul(attn_output, o_proj_weight.T)
    
    # 残差连接
    hidden_states = residual + attn_output
    
    # 后注意力层归一化
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, 
        all_parameters[prefix + "post_attention_layernorm.weight"].float().numpy(),
        config["rms_norm_eps"]
    )
    
    # MLP
    gate_proj_weight = all_parameters[prefix + "mlp.gate_proj.weight"].float().numpy()
    up_proj_weight = all_parameters[prefix + "mlp.up_proj.weight"].float().numpy()
    down_proj_weight = all_parameters[prefix + "mlp.down_proj.weight"].float().numpy()
    
    mlp_output = mlp_forward(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight)
    
    # 残差连接
    hidden_states = residual + mlp_output
    
    return hidden_states

# 创建因果注意力掩码
def create_causal_mask(seq_length):
    # 创建下三角矩阵（包括对角线）
    mask = np.tril(np.ones((seq_length, seq_length)))
    # 转换为注意力掩码格式，将0转换为大的负值
    mask = np.where(mask == 0, -1e9, 0)
    # 扩展维度以适应注意力计算
    return mask.reshape(1, 1, seq_length, seq_length)

# 主推理函数
def inference(input_ids, all_parameters, config):
    print(TerminalFormatter.section("开始推理过程"))
    
    # 1. 嵌入层
    print(TerminalFormatter.info("执行嵌入层处理..."))
    embed_tokens_weight = all_parameters['model.embed_tokens.weight'].float().numpy()
    hidden_states = embed_tokens_weight[input_ids]
    
    # 获取序列长度并创建位置ID
    batch_size, seq_length = input_ids.shape
    position_ids = np.arange(seq_length).reshape(1, -1)
    
    # 创建注意力掩码
    attention_mask = create_causal_mask(seq_length)
    print(TerminalFormatter.success("嵌入层处理完成！"))
    
    # 2. 通过所有Transformer层
    print(TerminalFormatter.section("Transformer层处理"))
    total_layers = config["num_hidden_layers"]
    
    for layer_idx in range(total_layers):
        # 使用进度条显示处理进度
        progress = TerminalFormatter.progress_bar(
            layer_idx + 1, total_layers, 
            prefix=f"处理层 {layer_idx+1}/{total_layers}", 
            suffix="", length=40
        )
        print(progress, end="")
        sys.stdout.flush()
        
        # 处理当前层
        hidden_states = transformer_layer_forward(
            hidden_states, layer_idx, all_parameters, config, position_ids, attention_mask
        )
        
        # 每5层或最后一层打印详细信息
        if (layer_idx + 1) % 5 == 0 or layer_idx == total_layers - 1:
            print()  # 换行
            print(TerminalFormatter.success(f"完成层 {layer_idx+1}/{total_layers}"))
    
    print()  # 确保进度条后有换行
    print(TerminalFormatter.success("所有Transformer层处理完成！"))
    
    # 3. 最终层归一化
    print(TerminalFormatter.info("执行最终层归一化..."))
    norm_weight = all_parameters["model.norm.weight"].float().numpy()
    hidden_states = rms_norm(
        hidden_states, 
        norm_weight,
        config["rms_norm_eps"]
    )
    
    # 4. 输出层（语言模型头）
    print(TerminalFormatter.info("计算最终logits..."))
    lm_head_weight = all_parameters["lm_head.weight"].float().numpy()
    logits = np.matmul(hidden_states, lm_head_weight.T)
    print(TerminalFormatter.success(f"推理完成！输出logits形状: {logits.shape}"))
    
    return logits

# 贪婪解码函数
def greedy_decode(input_ids, all_parameters, config, max_new_tokens=512):
    print(TerminalFormatter.header("贪婪解码生成", 80))
    
    # 初始输入
    current_ids = input_ids.copy()
    generated_ids = []
    generated_text = ""
    
    # 用于记录所有token的差异
    all_max_diffs = []
    all_p80_diffs = []
    
    print(TerminalFormatter.info(f"开始贪婪解码，最多生成 {max_new_tokens} 个token..."))
    print(TerminalFormatter.colorize("─" * 80, 'bright_blue'))
    
    # 生成新token
    start_time = time.time()
    for i in range(max_new_tokens):
        # 显示生成进度
        progress = TerminalFormatter.progress_bar(
            i + 1, max_new_tokens,
            prefix=f"生成进度", 
            suffix=f"Token {i+1}/{max_new_tokens}", 
            length=30
        )
        print(progress, end="")
        sys.stdout.flush()
        
        # 执行推理并比较结果
        max_diff, p80_diff = compare_inference_results(current_ids, all_parameters, config)
        all_max_diffs.append(max_diff)
        all_p80_diffs.append(p80_diff)
        
        # 获取numpy实现的logits
        logits = inference(current_ids, all_parameters, config)
        next_token_logits = logits[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)
        
        # 添加到生成的ID列表
        generated_ids.append(next_token_id[0])
        
        # 解码当前token并显示
        token_text = tokenizer.decode([next_token_id[0]])
        generated_text += token_text
        
        # 每个token都显示
        print()  # 换行
        print(TerminalFormatter.token_display(token_text))
        print(TerminalFormatter.colorize(f"当前token差异 - 最大: {max_diff:.6f}, 80%分位: {p80_diff:.6f}", 'bright_yellow'))
        
        # 如果生成了多个token，每10个显示一次累积文本
        if (i + 1) % 10 == 0:
            print("\n" + TerminalFormatter.colorize("当前生成文本:", 'bright_cyan', 'bold'))
            print(TerminalFormatter.colorize(generated_text, 'bright_white'))
            print(TerminalFormatter.colorize("─" * 40, 'bright_blue'))
        
        # 检查是否生成了EOS token
        if next_token_id[0] == config.get("eos_token_id", 151643):
            print(TerminalFormatter.warning("检测到EOS token，停止生成。"))
            break
        
        # 更新当前输入
        current_ids = np.concatenate([current_ids, next_token_id.reshape(1, 1)], axis=1)
    
    # 计算生成速度和差异统计
    end_time = time.time()
    generation_time = end_time - start_time
    tokens_per_second = len(generated_ids) / generation_time
    
    # 计算总体差异统计
    avg_max_diff = np.mean(all_max_diffs)
    avg_p80_diff = np.mean(all_p80_diffs)
    max_max_diff = np.max(all_max_diffs)
    max_p80_diff = np.max(all_p80_diffs)
    
    print("\n" + TerminalFormatter.colorize("─" * 80, 'bright_blue'))
    print(TerminalFormatter.success(f"贪婪解码完成，共生成 {len(generated_ids)} 个tokens。"))
    print(TerminalFormatter.info(f"生成速度: {tokens_per_second:.2f} tokens/秒"))
    
    # 显示差异统计总结
    print("\n" + TerminalFormatter.colorize("差异统计总结:", 'bright_cyan', 'bold'))
    print(TerminalFormatter.colorize(f"平均最大差异: {avg_max_diff:.6f}", 'bright_white'))
    print(TerminalFormatter.colorize(f"平均80%分位差异: {avg_p80_diff:.6f}", 'bright_white'))
    print(TerminalFormatter.colorize(f"最大差异峰值: {max_max_diff:.6f}", 'bright_white'))
    print(TerminalFormatter.colorize(f"80%分位差异峰值: {max_p80_diff:.6f}", 'bright_white'))
    
    # 显示完整生成文本
    print("\n" + TerminalFormatter.colorize("完整生成文本:", 'bright_cyan', 'bold'))
    print(TerminalFormatter.colorize(generated_text, 'bright_white'))
    
    return np.array(generated_ids)


print(TerminalFormatter.header("完整模型推理演示", 80))
print(TerminalFormatter.section("使用完整模型生成文本"))

# 创建生成配置
generation_config = {
    "max_new_tokens": 5,  # 减少生成token数量以加快演示
    "temperature": 0.6,    # 从generation_config.json中获取
    "top_p": 0.95,        # 从generation_config.json中获取
    "eos_token_id": 151643  # 从config.json中获取
}

# 打印生成配置
print(TerminalFormatter.colorize("生成配置:", 'bright_cyan', 'bold'))
for key, value in generation_config.items():
    print(TerminalFormatter.colorize(f"  {key}: {value}", 'bright_white'))

# 执行完整的贪婪解码生成
print(TerminalFormatter.header("执行完整贪婪解码", 80))
print(TerminalFormatter.info("使用所有28层执行贪婪解码..."))

# 使用完整配置进行贪婪解码
generated_ids = greedy_decode(input_ids, all_parameters, config, generation_config['max_new_tokens'])

# 将生成的token ID转换为文本
generated_text = tokenizer.decode(generated_ids)
# print("\n" + TerminalFormatter.colorize("生成的文本:", 'bright_cyan', 'bold'))
# print(TerminalFormatter.colorize(generated_text, 'bright_white'))

# 将原始输入和生成的文本组合
full_input_text = tokenizer.decode(input_ids[0])
full_response = full_input_text + generated_text
print("\n" + TerminalFormatter.colorize("完整对话:", 'bright_cyan', 'bold'))
print(TerminalFormatter.colorize(full_response, 'bright_white'))

print("\n" + TerminalFormatter.success("贪婪解码成功完成！"))