from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pdb
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
from PIL import Image
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

SYSTEM_PROMPT  = '''You are a helpful assistant.'''

# 热力图生成相关参数（核心可视化配置）
def generate_heatmap_on_image(feature_matrix, image_path, save_path, alpha=0.5, colormap='jet', 
                             normalize=True, resize_to_original=True, method='resize'):
    """
    根据特征矩阵在图像上生成热力图并保存
    
    参数:
    - feature_matrix: 特征矩阵 (numpy array 或 torch.Tensor), 形状为 [H, W] 或 [C, H, W] 或 [H, 1, W]
    - image_path: 原始图像路径
    - save_path: 热力图保存路径
    - alpha: 热力图透明度 (0-1)
    - colormap: 颜色映射 ('jet', 'viridis', 'hot', 'coolwarm' 等)
    - normalize: 是否对特征矩阵进行归一化
    - resize_to_original: 是否将热力图调整到原始图像尺寸
    - method: 调整尺寸的方法 ('resize', 'interpolate')
    """
    
    try:
        # 1. 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_h, original_w = original_image.shape[:2]
        
        print(f"原始图像尺寸: {original_w}x{original_h}")
        
        # 2. 处理特征矩阵
        if isinstance(feature_matrix, torch.Tensor):
            feature_matrix = feature_matrix.detach().cpu().numpy()
        
        print(f"输入特征矩阵形状: {feature_matrix.shape}")
        
        # 确保是2D矩阵
        if feature_matrix.ndim == 3:
            # 处理 [H, 1, W] 形状的情况
            if feature_matrix.shape[1] == 1:  # 形状为 [H, 1, W]
                feature_matrix = feature_matrix[:, 0, :]  # 直接去除中间的维度
                print(f"转换后特征矩阵形状: {feature_matrix.shape}")
            # 处理 [C, H, W] 形状的情况
            elif feature_matrix.shape[0] in [1, 3]:  # 可能是 [C, H, W]
                feature_matrix = np.mean(feature_matrix, axis=0)  # 取通道平均
                print(f"转换后特征矩阵形状: {feature_matrix.shape}")
            else:
                feature_matrix = np.mean(feature_matrix, axis=0)  # 取第一个维度平均
                print(f"转换后特征矩阵形状: {feature_matrix.shape}")
        
        feature_h, feature_w = feature_matrix.shape
        print(f"特征矩阵尺寸: {feature_w}x{feature_h}")

        mean_data = np.mean(feature_matrix, axis=0, keepdims=True)
        feature_matrix = np.resize(mean_data, (23, 23))
        # feature_matrix = np.resize(mean_data, (18, 27))

        
        # 3. 归一化特征矩阵
        if normalize:
            feature_matrix = (feature_matrix - feature_matrix.min()) / (feature_matrix.max() - feature_matrix.min() + 1e-8)
        
        # 4. 调整热力图尺寸到原始图像尺寸
        if resize_to_original and (feature_w != original_w or feature_h != original_h):
            if method == 'resize':
                heatmap = cv2.resize(feature_matrix, (original_w, original_h))
            else:  # interpolate
                from scipy import ndimage
                zoom_factor = (original_h / feature_h, original_w / feature_w)
                heatmap = ndimage.zoom(feature_matrix, zoom_factor, order=1)
        else:
            heatmap = feature_matrix
        
        print(f"热力图尺寸: {heatmap.shape[1]}x{heatmap.shape[0]}")
        
        # 5. 应用颜色映射
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # 取RGB通道，忽略alpha
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # 6. 调整原始图像尺寸（如果需要）
        if heatmap_colored.shape[:2] != original_image.shape[:2]:
            original_image = cv2.resize(original_image, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
        
        # 7. 融合图像和热力图
        blended = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        # 8. 保存结果
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, blended_bgr)
        
        print(f"热力图已保存到: {save_path}")
        
        return blended, heatmap
        
    except Exception as e:
        print(f"生成热力图时出错: {e}")
        return None, None



# 简化版注意力趋势图
def quick_time_avg_and_visualize_attention(attentions, answer_end_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list=None, output_text=None, processor=None):
    os.makedirs(save_folder, exist_ok=True)
    
    # 图像 Token 的起止位置，由pos和num_image_token计算，定义模型注意力中图像区域的范围
    img_token_start = pos
    img_token_end = pos + num_image_token
    # 查询文本 Token 起始位置（图像 Token 结束 + 1），可调整偏移量
    query_start_pos = img_token_end + 1
    # 生成文本 Token 起始位置（查询文本结束 + 5），硬编码偏移量+5可按需修改
    gen_start_pos = query_end_pos + 5


    steps = answer_end_idx
    img_attentions = np.zeros(steps)
    txt_attentions = np.zeros(steps)
    gen_attentions = np.zeros(steps)

    for step in range(1, steps + 1):
        step_idx = step - 1  
        current_attentions = attentions[step]
        num_layers = len(current_attentions)

        layer_img_sum = 0.0
        layer_txt_sum = 0.0
        layer_gen_sum = 0.0

        for layer_idx in range(num_layers):
            att_matrix = current_attentions[layer_idx][0]
            image_path = "/root/autodl-fs/DVGBench/images_28/all/Test_PostEarthquake_042.png"   # 测试图像路径，可替换为自定义图像路径
            save_path = f"/root/Look-Back/vis_demo/results/Qwen2.5-VL-7B-back/Test_NonEvent_056/heatmap_{step}_{layer_idx}.png"  
            # pos + num_image_token
            # print(step, step_idx, layer_idx)
            # break
            if layer_idx == num_layers - 1:
                generate_heatmap_on_image(att_matrix[:,:, img_token_start:img_token_end].to(torch.float32), image_path, save_path, alpha=0.5, colormap='jet', normalize=True, resize_to_original=True, method='resize')
            # att_matrix.shape: (28, 1, 589)
            # att_matrix.shape.mean(dim=0): (1, 1, 589)
            # att_matrix.shape.mean(dim=0).squeeze(0): (1, 589)
            att_avg = att_matrix.mean(dim=0).squeeze(0).to(torch.float32).detach().cpu().numpy()
            
            
            total_attention = att_avg[img_token_start:].sum()
            if total_attention == 0:
                continue  

            img_att = att_avg[img_token_start:img_token_end].sum() / total_attention
            txt_att = att_avg[query_start_pos:gen_start_pos].sum() / total_attention
            gen_att = att_avg[gen_start_pos:].sum() / total_attention
            layer_img_sum += img_att
            layer_txt_sum += txt_att
            layer_gen_sum += gen_att

        if num_layers > 0:
            img_attentions[step_idx] = layer_img_sum / num_layers
            txt_attentions[step_idx] = layer_txt_sum / num_layers
            gen_attentions[step_idx] = layer_gen_sum / num_layers

    back_start_positions = []
    back_end_positions = []
    if output_text and processor and decoded_tokens_text_list:


        for i, token in enumerate(decoded_tokens_text_list[:steps]):
            if token == '<' or token.endswith('<'):

                if (i + 2 < len(decoded_tokens_text_list) and 
                    decoded_tokens_text_list[i + 1] == 'explicit' and 
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_start_positions.append(i + 1) 

            elif token == '</' or token.endswith('</'):
                if (i + 2 < len(decoded_tokens_text_list) and
                    decoded_tokens_text_list[i + 1] == 'explicit' and
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_end_positions.append(i + 1)


    plt.figure(figsize=(12, 6))
    plt.plot(range(1, steps+1), img_attentions, label="Image Token", color="blue", alpha=0.8)
    plt.plot(range(1, steps+1), txt_attentions, label="Query Text Token", color="orange", alpha=0.8)
    plt.plot(range(1, steps+1), gen_attentions, label="Generated Token", color="green", alpha=0.8)
    
    for pos_mark in back_start_positions:
        if pos_mark <= steps:
            plt.axvline(x=pos_mark, color='red', linestyle='-', alpha=0.8, linewidth=2, label='<explicit>' if pos_mark == back_start_positions[0] else '')
    
    for pos_mark in back_end_positions:
        if pos_mark <= steps:
            plt.axvline(x=pos_mark, color='purple', linestyle='-', alpha=0.8, linewidth=2, label='</explicit>' if pos_mark == back_end_positions[0] else '')
    
    for start_pos, end_pos in zip(back_start_positions, back_end_positions):
        if start_pos <= steps and end_pos <= steps:
            plt.axvspan(start_pos, end_pos, alpha=0.2, color='yellow', label='<explicit> region' if start_pos == back_start_positions[0] else '')
    
    plt.xlabel("Generation Step")
    plt.ylabel("Attention Ratio")
    plt.title(f"Attention Distribution Over Generation Steps")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6) # 网格样式
    
    save_name = f"{mode}_step_attention"
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Step-wise attention plot saved to {save_path}")


# 带 Token 标签的注意力趋势图
def time_avg_and_visualize_attention(attentions, answer_end_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list, output_text=None, processor=None):
    os.makedirs(save_folder, exist_ok=True)

    img_token_start = pos
    img_token_end = pos + num_image_token
    query_start_pos = img_token_end + 1
    gen_start_pos = query_end_pos + 5

    steps = answer_end_idx
    img_attentions = np.zeros(steps)
    txt_attentions = np.zeros(steps)
    gen_attentions = np.zeros(steps)

    for step in range(1, steps + 1):
        step_idx = step - 1  
        current_attentions = attentions[step] 
        num_layers = len(current_attentions)

        layer_img_sum = 0.0
        layer_txt_sum = 0.0
        layer_gen_sum = 0.0


        for layer_idx in range(num_layers):
            att_matrix = current_attentions[layer_idx][0]
            att_avg = att_matrix.mean(dim=0).squeeze(0).to(torch.float32).detach().cpu().numpy()

            total_attention = att_avg[img_token_start:].sum()
            if total_attention == 0:
                continue 

            img_att = att_avg[img_token_start:img_token_end].sum() / total_attention
            txt_att = att_avg[query_start_pos:gen_start_pos].sum() / total_attention
            gen_att = att_avg[gen_start_pos:].sum() / total_attention
            layer_img_sum += img_att
            layer_txt_sum += txt_att
            layer_gen_sum += gen_att

        if num_layers > 0:
            img_attentions[step_idx] = layer_img_sum / num_layers
            txt_attentions[step_idx] = layer_txt_sum / num_layers
            gen_attentions[step_idx] = layer_gen_sum / num_layers

    token_labels = decoded_tokens_text_list[:steps]
    
    back_start_positions = []
    back_end_positions = []
    if output_text and processor:
        
        for i, token in enumerate(decoded_tokens_text_list[:steps]):
            if token == '<' or token.endswith('<'):
                if (i + 2 < len(decoded_tokens_text_list) and 
                    decoded_tokens_text_list[i + 1] == 'explicit' and 
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_start_positions.append(i + 1) 
            elif token == '</' or token.endswith('</'):
                if (i + 2 < len(decoded_tokens_text_list) and
                    decoded_tokens_text_list[i + 1] == 'explicit' and
                    decoded_tokens_text_list[i + 2].startswith('>')):
                    back_end_positions.append(i + 1) 
        


    max_labels = 40   # 最大显示 Token 数（控制 X 轴最多显示的 Token 标签数，超过则采样显示）
    if steps <= max_labels:
        step_size = 1
        xtick_positions = list(range(1, steps + 1))
        xtick_labels = token_labels
    else:
        step_size = steps // max_labels
        xtick_positions = list(range(1, steps + 1, step_size))
        if xtick_positions[-1] != steps:
            xtick_positions.append(steps)
        xtick_labels = [token_labels[i - 1] for i in xtick_positions]

    plt.figure(figsize=(15, 8))  # 图表尺寸（控制图表宽高，可修改数值调整尺寸）
    
    x = range(1, steps + 1)
    # 线条样式 / 颜色（图像 / 文本 / 生成 Token 的注意力曲线颜色、透明度）
    plt.plot(x, img_attentions, label="Image Token", color="blue", alpha=0.8)  
    plt.plot(x, txt_attentions, label="Query Text Token", color="orange", alpha=0.8)
    plt.plot(x, gen_attentions, label="Generated Token", color="green", alpha=0.8)

    for pos in back_start_positions:
        if pos <= steps:
            # 特殊 Token 标记（<explicit>/</explicit>的竖线、背景色，可修改颜色 / 线型）
            plt.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2, label='<explicit>' if pos == back_start_positions[0] else '')
    
    for pos in back_end_positions:
        if pos <= steps:
            plt.axvline(x=pos, color='purple', linestyle='-', alpha=0.8, linewidth=2, label='</explicit>' if pos == back_end_positions[0] else '')

    for start_pos, end_pos in zip(back_start_positions, back_end_positions):
        if start_pos <= steps and end_pos <= steps:
            plt.axvspan(start_pos, end_pos, alpha=0.2, color='yellow', label='<explicit> region' if start_pos == back_start_positions[0] else '') 
    
    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right', fontsize=10)     # X 轴标签旋转角度（rotation控制 Token 标签旋转角度，避免重叠）
    for tick in xtick_positions:
        plt.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel("Generated Tokens", fontsize=12)
    plt.ylabel("Attention Ratio", fontsize=12)
    plt.title(f"Attention Distribution Over Generation Steps", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_name = f"{mode}_step_attention_with_tokens.png"
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    if steps > max_labels:
        plt.figure(figsize=(steps / 5, 6))   # 详细版图表尺寸（全 Token 详细图表的尺寸，按生成步数自适应）
        plt.plot(x, img_attentions, label="Image Token", color="blue", alpha=0.8)
        plt.plot(x, txt_attentions, label="Query Text Token", color="orange", alpha=0.8)
        plt.plot(x, gen_attentions, label="Generated Token", color="green", alpha=0.8)

        for pos in back_start_positions:
            if pos <= steps:
                plt.axvline(x=pos, color='red', linestyle='-', alpha=0.8, linewidth=2, label='<explicit>' if pos == back_start_positions[0] else '')
        
        for pos in back_end_positions:
            if pos <= steps:
                plt.axvline(x=pos, color='purple', linestyle='-', alpha=0.8, linewidth=2, label='</explicit>' if pos == back_end_positions[0] else '')

        for start_pos, end_pos in zip(back_start_positions, back_end_positions):
            if start_pos <= steps and end_pos <= steps:
                plt.axvspan(start_pos, end_pos, alpha=0.2, color='yellow', label='<explicit> region' if start_pos == back_start_positions[0] else '')

        plt.xticks(range(1, steps + 1), token_labels, rotation=90, ha='center', fontsize=8)

        plt.xlabel("Generated Tokens")
        plt.ylabel("Attention Ratio")
        plt.title(f"Detailed Attention Distribution (All Tokens)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_name_detailed = f"Step_attention_all_tokens.png"
        save_path_detailed = os.path.join(save_folder, save_name_detailed)
        plt.savefig(save_path_detailed, bbox_inches='tight')
        plt.close()

        print(f"Detailed attention plot with all tokens saved to {save_path_detailed}")

def process_inputs_and_generate_output(processor, messages, model, device):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    image_inputs_aux = processor.image_processor(images=image_inputs)
    output_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:] / 2
    output_shape = output_shape.astype(int)
    num_image_token = (output_shape[0] * output_shape[1]).item()

    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
    pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)
    input_list = inputs.input_ids.tolist()[0]
    sys_end_pos = input_list.index(151645)
    
    try:
        query_end_pos = input_list.index(151645, sys_end_pos + 1)
    except ValueError:
        print("query_end_pos occurrence of token 151645 not found")

    outputs = model.generate(**inputs, max_new_tokens=2048, output_attentions=True, return_dict_in_generate=True, output_hidden_states=True)
    attentions = outputs.attentions
    generated_ids = outputs.sequences

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    
    print("Generated Text:", output_text)
    decoded_tokens = processor.tokenizer.encode(output_text[0], add_special_tokens=True)
    # print(decoded_tokens, decoded_tokens.index)
    try:
        End_idx = decoded_tokens.index(151645)
    except:
        End_idx = decoded_tokens.index(151643)
    
    # End_idx = decoded_tokens.index('<|endoftext|>')
    
    decoded_tokens_text_list = processor.tokenizer.convert_ids_to_tokens(decoded_tokens)
    return output_text, End_idx, num_image_token, pos, query_end_pos, attentions, decoded_tokens_text_list, output_shape

# def select_prompt_template(mode, question):
#     """
#     Function to prepend the instruction to the base text.
#     """
#     if mode == "back":
#         instruct = """
# You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags, and use <back> </back> to verify your reasoning against the image. The final answer MUST BE put in \boxed{}, respectively, i.e., <think> reasoning process here </think> <back> verification process here </back> <think> continue reasoning </think> \\boxed{final answer}.
#         """
    
#     elif mode == "cot":
#         instruct = """
# You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}, respectively, i.e., <think> reasoning process here </think> \\boxed{final answer}.
#         """

#     elif mode == "normal":
#         instruct = ""
#     return question + " " + instruct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attention visualization with custom parameters")
    parser.add_argument("--mode", type=str, default="back", choices=["back", "cot", "normal"], help="Reasoning mode")
    parser.add_argument("--image_path", type=str, default="./data/images/demo11.png", help="Path to the image file")
    parser.add_argument("--question", type=str, default="Subtract 0 blue spheres. How many objects are left?", help="Question to ask")
    parser.add_argument("--model_path", type=str, default="/home/liuyuyang_2/yangshuo/attention_in_rl/MLLMs/Reflect_4", help="Path to the model directory")
    parser.add_argument("--save_folder", type=str, default="./back_qwen_output_attention_maps_new/pretrain", help="Base folder to save output files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, or auto for automatic detection)")
    args = parser.parse_args()
    mode = args.mode
    image_path = args.image_path
    demo = os.path.splitext(os.path.basename(image_path))[0]
    question = args.question
    model_path = args.model_path
    save_folder = os.path.join(args.save_folder, demo)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", attn_implementation="eager"
    )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    # input_text = select_prompt_template(mode, question)
    input_text = question
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": input_text},
            ],
        }
    ]
    
    output_text, End_idx, num_image_token, pos, query_end_pos, attentions, decoded_tokens_text_list, output_shape = process_inputs_and_generate_output(processor, messages, model, device)
    quick_time_avg_and_visualize_attention(attentions, End_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list, output_text[0], processor)
    time_avg_and_visualize_attention(attentions, End_idx, pos, num_image_token, query_end_pos, save_folder, mode, decoded_tokens_text_list, output_text[0], processor)
