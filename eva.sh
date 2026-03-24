# 核心配置区
IMAGE_DIR="/root/autodl-fs/DVGBench/images_28/all" # 图片存储根目录：指定待测试图片所在的文件夹路径
IMAGE_NAME="Test_PostEarthquake_042.png"  # 待测试图片名称：明确本次推理使用的目标图片（可根据需求替换）


# 推理指令配置 
# 定义推理提问指令（QUESTION变量），并按指定格式输出结果
QUESTION="<image>Find <ref>A car that is turning left</ref> in the image. Output the thinking process in <think> </think>,  a brief description of the referred object (use words such as color, size, and relative position) in <explicit> </explicit>, and the final answer [x1,y1,x2,y2] in <answer> </answer> tags." 

python vis_demo/att_trend_viz-heatmap-i2e.py \
  --image_path "${IMAGE_DIR}/${IMAGE_NAME}" \
  --question "$QUESTION" \
  --mode back \
  --model_path "/root/autodl-fs/output/GRPO_Qwen2-VL-7B-Instruct/swift_dvg_train_28_cot2_12/v0-20250915-192007/checkpoint-663-merged" \
  --save_folder "./vis_demo/results/Qwen2.5-VL-7B-back" \
  --device "cuda:0"


# QUESTION="<image>Find <ref>正在行驶的车</ref> in the image. Output the thinking process in <think> </think>, and the final answer [x1,y1,x2,y2] in <answer> </answer> tags." 

# python vis_demo/att_trend_viz-heatmap-i2e.py \
#   --image_path "${IMAGE_DIR}/${IMAGE_NAME}" \
#   --question "$QUESTION" \
#   --mode back \
#   --model_path "/root/autodl-fs/output/GRPO_Qwen2-VL-7B-Instruct/swift_dvg_train_28_cot2_11/v0-20250915-130837/checkpoint-663-merged" \
#   --save_folder "./vis_demo/results/Qwen2.5-VL-7B-back" \
#   --device "cuda:0"





