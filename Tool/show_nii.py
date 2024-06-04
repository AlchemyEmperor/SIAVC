
import nibabel as nib
import matplotlib.pyplot as plt

def visualize_nii_file_and_save(nii_file_path, output_png_path):
    try:
        # 加载 NIfTI 文件
        img = nib.load(nii_file_path)
        data = img.get_fdata()

        # 获取图像的尺寸和层数
        width, height, depth = data.shape

        # 取中心层进行可视化
        mid_slice = depth // 2

        # 可视化图像
        plt.imshow(data[:, :, mid_slice], cmap='gray')
        plt.axis('off')

        # 保存为 PNG 格式
        plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0)

        # 显示图像（可选）
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")

# 使用示例
if __name__ == "__main__":
    nii_file_path = "E:\\Datatest\\BraTS2021\\Training\\NII\\BraTS2021_00000\\BraTS2021_00000_t1.nii.gz"         # 替换为你的 NIfTI 文件路径
    output_png_path = "E:\\Datatest\\BraTS2021\\Training\\IMG"    # 替换为你想要保存的 PNG 文件路径
    visualize_nii_file_and_save(nii_file_path, output_png_path)