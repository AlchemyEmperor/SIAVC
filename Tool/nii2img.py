import cv2
import nibabel as nib
import os #遍历文件夹
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

def nii_to_images(filepath,flag="image"):

    f1names = os.listdir(filepath)
    for f1 in f1names:
        f1_path = os.path.join(oldfilepath, f1)
        filenames = os.listdir(f1_path)  # 读取nii文件
        slice_trans = []
        for f in filenames:
            # 开始读取nii文件
            img_path = os.path.join(filepath, f1_path, f)
            img = nib.load(img_path)  # 读取nii
            img_fdata = img.get_fdata()
            fname = f.replace('.nii', '')  # 去掉nii的后缀名
            img_f_path = os.path.join(newfilepath, f1)
            if not os.path.exists(img_f_path):
                os.mkdir(img_f_path)  # 新建文件夹
            fname = os.path.splitext(fname)[0]
            img_f_path = os.path.join(img_f_path, fname)
            # 创建nii对应图像文件
            visualize_nii_file_and_save(img_path, img_f_path)




if __name__ == '__main__':
    oldfilepath = 'E:\\Datatest\\BraTS2021\\Training\\NII'  #nii文件所在的文件夹路径
    newfilepath = 'E:\\Datatest\\BraTS2021\\Training\\IMG'  #转化后的png文件存放的文件路径
    nii_to_images(oldfilepath,"label")
