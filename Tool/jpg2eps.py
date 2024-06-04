from PIL import Image

# 打开JPEG图像
jpeg_image_path = 'C:/Users/LQH/Desktop/HaoyiFan.jpg'
jpeg_image = Image.open(jpeg_image_path)

# 创建EPS图像（可调整DPI）
eps_image_path = 'C:/Users/LQH/Desktop/HaoyiFan.eps'
jpeg_image.save(eps_image_path, "EPS", dpi=(300, 300))

print("JPEG image converted to EPS format.")