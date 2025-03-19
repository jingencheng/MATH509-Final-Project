import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from rsc.rbf_regression import RBFRegression
import time

class ImageInpainting:
    """图像修复类 - RBF算法版本"""
    def __init__(self, model_type='RBF'):
        """初始化图像修复类
        参数:
        - model_type: 使用的模型类型，目前仅支持'RBF'
        """
        self.model_type = model_type.upper()
        if self.model_type != 'RBF':
            raise ValueError(f"当前版本仅支持RBF模型")

    def inpaint(self, image_name, fill_rgb, spacing, width, l2_coef=0.5, patch_size=25, tolerance=0.1):
        """执行图像修复
        参数:
        - image_name (str): 需要修复的图像文件名
        - fill_rgb (list): 需要填充的RGB颜色值，范围0-255
        - spacing (int): 采样点之间的间距，范围1-9
        - width (float): 采样范围的宽度，范围1-2*spacing
        - l2_coef (float): L2正则化系数
        - patch_size (int): 图像块大小
        - tolerance (float): 颜色相似度阈值
        """
        # 参数验证
        assert os.path.isfile(image_name), f"图像文件 {image_name} 不存在"
        assert len(fill_rgb) == 3 and all([0 <= element <= 255 for element in fill_rgb]), f"fill_rgb必须是3个0-255之间的整数，当前值: {fill_rgb}"
        assert 1 <= spacing <= 9, f"spacing必须在1-9之间，当前值: {spacing}"
        assert 1 <= width <= 2 * spacing, f"width必须在1-{2 * spacing}之间，当前值: {width}"
        assert 1 <= patch_size, f"patch_size必须大于等于1，当前值: {patch_size}"
        
        # 初始化参数
        CENTER_SPACING = spacing
        PATCH_SIZE = patch_size
        TOL = tolerance
        fill_rgb = fill_rgb.astype(float) / 255  # 归一化RGB值

        # 读取图像
        im = mpimg.imread(image_name)
        im = im.astype(float) / 255  # 归一化图像
        im_rec = im
        
        # 遍历图像块进行修复
        for i in range(CENTER_SPACING+1, im.shape[1]-(PATCH_SIZE+CENTER_SPACING)+1, PATCH_SIZE):
            for j in range(CENTER_SPACING+1, im.shape[0]-(PATCH_SIZE+CENTER_SPACING)+1, PATCH_SIZE):
                # 创建采样点网格
                [XX,YY] = np.meshgrid(
                    list(range(i-CENTER_SPACING, i+PATCH_SIZE+CENTER_SPACING+1, CENTER_SPACING)),
                    list(range(j-CENTER_SPACING, j+PATCH_SIZE+CENTER_SPACING+1, CENTER_SPACING))
                )

                # 构建采样点和宽度
                centers = np.array((XX.flatten(), YY.flatten()), dtype=float).T
                num_centers = centers.shape[0]
                widths = np.ones(shape=(num_centers, 1), dtype=float) * width

                # 为每个颜色通道创建RBF模型
                red_model = RBFRegression(centers=centers, widths=widths)
                green_model = RBFRegression(centers=centers, widths=widths)
                blue_model = RBFRegression(centers=centers, widths=widths)
                
                # 创建需要填充区域的坐标网格
                [XX,YY] = np.meshgrid(
                    list(range(i, i+PATCH_SIZE+1)),
                    list(range(j, j+PATCH_SIZE+1))
                )
                Pfill = np.array([XX.reshape(-1,order='F'), YY.reshape(-1,order='F')])
                patch_fill = im[j-1:j+PATCH_SIZE, i-1:i+PATCH_SIZE]
                
                # 找出需要填充的像素
                ref = patch_fill - fill_rgb
                ref = np.power(ref, 2)
                ref = np.sum(ref, 2)
                index_fill = np.argwhere(ref <= TOL)
                idx_fill = np.sort(index_fill[:,1]*ref.shape[0]+index_fill[:,0])
                
                # 创建训练数据坐标网格
                [XX,YY] = np.meshgrid(
                    list(range(i-CENTER_SPACING, i+PATCH_SIZE+CENTER_SPACING+1)),
                    list(range(j-CENTER_SPACING, j+PATCH_SIZE+CENTER_SPACING+1))
                )
                P = np.array([XX.reshape(-1,order='F'), YY.reshape(-1,order='F')])
                
                patch = im[j-CENTER_SPACING-1:j+PATCH_SIZE+CENTER_SPACING, 
                          i-CENTER_SPACING-1:i+PATCH_SIZE+CENTER_SPACING]

                # 找出可用于训练的像素
                ref = patch - fill_rgb
                ref = np.power(ref, 2)
                ref = np.sum(ref, 2)
                index_data = np.argwhere(ref > TOL)
                idx_data = np.sort(index_data[:,1]*ref.shape[0]+index_data[:,0])
                
                # 如果存在需要填充的像素，进行修复
                if (idx_fill.size > 0):
                    print('正在修复选定颜色的区域')
                    if(idx_data.size <= num_centers):
                        print('训练数据不足，直接复制图像块\n')
                        patch_rec = patch_fill
                    else:
                        # 获取有效的训练数据位置
                        PP = P[:,idx_data]

                        # 对每个颜色通道进行模型拟合
                        # 红色通道
                        patch_R = patch[:,:,0]
                        z_R = patch_R.reshape(patch_R.size,1, order='F')
                        z_R = z_R[idx_data]
                        red_model.fit_with_l2_regularization(PP.T, z_R, l2_coef)
                        
                        # 绿色通道
                        patch_G = patch[:,:,1]
                        z_G = patch_G.reshape(patch_G.size,1, order='F')
                        z_G = z_G[idx_data]
                        green_model.fit_with_l2_regularization(PP.T, z_G, l2_coef)
                        
                        # 蓝色通道
                        patch_B = patch[:,:,2]
                        z_B = patch_B.reshape(patch_B.size,1, order='F')
                        z_B = z_B[idx_data]
                        blue_model.fit_with_l2_regularization(PP.T, z_B, l2_coef)
                        
                        # 打印数据形状并退出程序
                        # print(f'PP.T: {PP.T.shape}, \nz_R: {z_R.shape}, \nz_G: {z_G.shape}, \nz_B: {z_B.shape}')
                        # import sys; sys.exit()  # 直接退出程序
                        # 在填充位置重建像素值
                        PP = Pfill[:,idx_fill].T
                        fill_R = red_model.predict(PP)
                        fill_G = green_model.predict(PP)
                        fill_B = blue_model.predict(PP)
                        
                        # 组装修复后的图像块
                        patch_rec = patch_fill
                        pr_R = patch_rec[:,:,0]
                        pr_G = patch_rec[:,:,1]
                        pr_B = patch_rec[:,:,2]
                        pr_R[index_fill[:,0],index_fill[:,1]] = np.squeeze(np.asarray(fill_R))
                        pr_G[index_fill[:,0],index_fill[:,1]] = np.squeeze(np.asarray(fill_G))
                        pr_B[index_fill[:,0],index_fill[:,1]] = np.squeeze(np.asarray(fill_B))
                        patch_rec[:,:,0] = pr_R
                        patch_rec[:,:,1] = pr_G
                        patch_rec[:,:,2] = pr_B
                else:
                    print('复制图像块 %d--%d'%(i,j))
                    patch_rec = patch_fill
                    
                # 更新修复后的图像
                im_rec[j-1:j+PATCH_SIZE,i-1:i+PATCH_SIZE] = patch_rec
            
        return np.round(im_rec,4)

    def calculate_color_differences(self, image_name, fill_rgb):
        """计算图像中所有像素点与目标颜色的差异值
        参数:
        - image_name (str): 图像文件名
        - fill_rgb (list): 目标RGB颜色值，范围0-255
        返回:
        - diff_values (ndarray): 所有像素点的颜色差异值
        """
        # 读取图像
        im = mpimg.imread(image_name)
        im = im.astype(float) / 255  # 归一化图像
        fill_rgb = fill_rgb.astype(float) / 255  # 归一化目标颜色
        
        # 计算颜色差异
        ref = im - fill_rgb
        ref = np.power(ref, 2)
        diff_values = np.sum(ref, 2)
        
        return diff_values

    def calculate_mse(self, original, reconstructed):
        """计算重建图像与原图的均方误差
        参数:
        - original: 原始图像
        - reconstructed: 重建图像
        返回:
        - mse: 均方误差值
        """
        # 确保图像是numpy数组
        if not isinstance(original, np.ndarray):
            original = np.array(original)
        if not isinstance(reconstructed, np.ndarray):
            reconstructed = np.array(reconstructed)
        
        # 计算MSE
        mse = np.mean((original - reconstructed) ** 2)
        return mse

    def plot_top_pixel_frequencies(self, image_name, top_n=10):
        """计算并显示RGB像素值频率的top N排序
        参数:
        - image_name: 图像文件名
        - top_n: 显示前N个最频繁的像素值组合
        """
        # 读取图像
        im = mpimg.imread(image_name)
        im = im.astype(float) / 255  # 归一化图像
        
        # 将RGB值四舍五入到3位小数，以便统计频率
        im_rounded = np.round(im, 3)
        
        # 将RGB值转换为字符串形式，以便作为唯一标识
        pixel_strings = [f'({r:.3f}, {g:.3f}, {b:.3f})' 
                        for r, g, b in zip(im_rounded[:,:,0].flatten(), 
                                         im_rounded[:,:,1].flatten(), 
                                         im_rounded[:,:,2].flatten())]
        
        # 统计每个RGB组合的频率
        unique_values, counts = np.unique(pixel_strings, return_counts=True)
        
        # 按频率降序排序
        sorted_indices = np.argsort(counts)[::-1]
        top_values = unique_values[sorted_indices[:top_n]]
        top_counts = counts[sorted_indices[:top_n]]
        
        # 创建新的图形
        plt.figure(figsize=(15, 6))
        
        # 绘制条形图
        plt.bar(range(top_n), top_counts)
        plt.title(f'Top {top_n} Most Frequent RGB Values', fontsize=12)
        plt.xlabel('Rank', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        
        # 添加数值标签
        for i, (value, count) in enumerate(zip(top_values, top_counts)):
            plt.text(i, count, f'{value}\n({count})', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.xticks(range(top_n))
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def inpaint_multiple_times(self, image_name, fill_rgb, spacing, width, l2_coef=0.5, patch_size=25, tolerance=[0.1]):
        """多次执行图像修复"""
        im_rec = None
        for i, tol in enumerate(tolerance):
            print(f'第{i+1}次修复 (tolerance={tol})...')
            im_rec = self.inpaint(image_name, fill_rgb, spacing, width, l2_coef, patch_size, tolerance=tol)
            # 更新image_name为上一次修复的结果
            if i < len(tolerance) - 1:  # 如果不是最后一次，保存临时结果
                temp_name = f'temp_inpaint_{i}.tif'
                # 确保像素值在0-1范围内，并只保存RGB通道
                im_rec_clipped = np.clip(im_rec, 0, 1)
                if im_rec_clipped.shape[-1] == 4:  # 如果是RGBA格式
                    im_rec_clipped = im_rec_clipped[..., :3]  # 只保留RGB通道
                # 将numpy数组转换为PIL图像并保存为tif格式
                im_pil = Image.fromarray((im_rec_clipped * 255).astype(np.uint8))
                im_pil.save(temp_name, format='TIFF')
                image_name = temp_name
        return im_rec

if __name__ == "__main__":
    # 设置图像路径
    image_name = "/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/processed/unsplash_eJlGGwboW3s.tif"
    oimage = Image.open("/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/original/unsplash_eJlGGwboW3s.tif")
    image = Image.open(image_name)

    # 设置修复参数
    fill_rgb = np.array([255, 0, 0])  # 需要填充的红色
    spacing = 3  # 采样点间距 3
    width = 2    # 采样范围宽度 2
    l2_coef = 0.5  # L2正则化系数 0.5
    # tolerance = 0.1 ~ 1.0 up to 3
    tolerance = 0.5
    
    # 使用RBF方法进行图像修复
    inpainter = ImageInpainting(model_type='RBF')
    
    # 进行多次修复
    im_rec1 = inpainter.inpaint_multiple_times(image_name, fill_rgb, spacing, width, l2_coef, tolerance=[0.2, 0.3, 0.4])
    # 使用tolerance=0.5进行一次修复
    # im_rec2 = inpainter.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=tolerance)

    inpainter.plot_top_pixel_frequencies(image_name)
    # 显示修复结果对比
    plt.figure(figsize=(20, 8))  # 调整图形尺寸为1行3列
    
    # 原始图像
    plt.subplot(1, 3, 1)  
    plt.imshow(oimage)
    plt.title('Original Image', fontsize=12)  
    plt.axis('off')

    # 损坏图像
    plt.subplot(1, 3, 2)  
    plt.imshow(image)
    plt.title(f'Damaged Image', fontsize=12)  
    plt.axis('off')

    # 宽松tolerance的修复结果
    plt.subplot(1, 3, 3) 
    plt.imshow(im_rec1)
    plt.title(f'tol=0.2 times=2', fontsize=12) 
    plt.axis('off')

    plt.tight_layout(pad=3.0)  # 增加子图之间的间距
    plt.show()
    

