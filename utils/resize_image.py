import os

from PIL import Image


def resize_image(input_image_path, output_image_path, scale_factor):
    # 打开原始图像
    original_image = Image.open(input_image_path)

    # 获取原始图像的尺寸
    width, height = original_image.size

    # 计算缩小后的尺寸
    new_width = width // scale_factor
    new_height = height // scale_factor

    # 缩小图像
    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)

    # 保存缩小后的图像
    resized_image.save(output_image_path)

    print(f"Original size: {width}x{height}")
    print(f"Resized size: {new_width}x{new_height}")



sfile = "../desk-feature"
dfile = "../desk-feature-s"
input_image = os.listdir(sfile)
scale_factor = 5
for image in input_image:
    input_image_path = os.path.join(sfile, image)
    output_image_path = os.path.join(dfile, image)
    # 调用函数进行图像缩放
    resize_image(input_image_path, output_image_path, scale_factor)






