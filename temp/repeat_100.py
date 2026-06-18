import os
from PIL import Image

# ========== 配置项，按需修改 ==========
input_image_path = "05.02.25#out1#mnr_input0012_block.bmp"  # 原始图片的路径
output_dir = "./05.02.25#out1#mnr_input0012_block"          # 输出文件夹
start_num = 0                    # 起始编号
end_num = 100                    # 结束编号（包含）
# =====================================

# 自动创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 读入原始图片并获取原格式
origin_img = Image.open(input_image_path)
_, file_ext = os.path.splitext(input_image_path)

# 循环生成 0000 ~ 0100 命名的副本
for idx in range(start_num, end_num + 1):
    file_name = f"frame{idx:04d}{file_ext}"
    save_path = os.path.join(output_dir, file_name)
    origin_img.save(save_path)
    print(f"已生成: {file_name}")

print(f"\n完成！共生成 {end_num - start_num + 1} 张图片，输出目录：{output_dir}")