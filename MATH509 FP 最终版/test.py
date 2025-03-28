import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from rsc.rbf_regression import RBFRegression
import time
from scipy.interpolate import griddata

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
        self.plot_count = 0  # 用于跟踪已绘制的次数

    def inpaint(self, image_name, fill_rgb, spacing, width, l2_coef=0.5, patch_size=25, tolerance=0.1, test_mode=False):
        """执行图像修复
        参数:
        - image_name (str): 需要修复的图像文件名
        - fill_rgb (list): 需要填充的RGB颜色值，范围0-255
        - spacing (int): 采样点之间的间距，范围1-9
        - width (float): 采样范围的宽度，范围1-2*spacing
        - l2_coef (float): L2正则化系数
        - patch_size (int): 图像块大小
        - tolerance (float): 颜色相似度阈值
        - test_mode (bool): 是否为测试模式，用于可视化训练数据
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
                        
                        # 绿色通道
                        patch_G = patch[:,:,1]
                        z_G = patch_G.reshape(patch_G.size,1, order='F')
                        z_G = z_G[idx_data]
                        
                        # 蓝色通道
                        patch_B = patch[:,:,2]
                        z_B = patch_B.reshape(patch_B.size,1, order='F')
                        z_B = z_B[idx_data]
                        
                        # 如果是测试模式，每10次修复绘制一次训练数据
                        if test_mode and self.plot_count % 1 == 0:
                            self._plot_training_data(PP, z_R, z_G, z_B, i, j)
                        
                        # 拟合模型
                        red_model.fit_with_l2_regularization(PP.T, z_R, l2_coef)
                        green_model.fit_with_l2_regularization(PP.T, z_G, l2_coef)
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
                        
                        if test_mode:
                            self.plot_count += 1
                else:
                    print('复制图像块 %d--%d'%(i,j))
                    patch_rec = patch_fill
                    
                # 更新修复后的图像
                im_rec[j-1:j+PATCH_SIZE,i-1:i+PATCH_SIZE] = patch_rec
            
        return np.round(im_rec,4)



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

    def inpaint_test(self, image_name, fill_rgb, spacing, width, l2_coef=0.5, patch_size=25, tolerance=0.1):
        """执行图像修复并可视化训练数据
        参数与inpaint()相同
        """
        self.plot_count = 0  # 重置计数器
        return self.inpaint(image_name, fill_rgb, spacing, width, l2_coef, patch_size, tolerance, test_mode=True)

    def _plot_training_data(self, PP, z_R, z_G, z_B, i, j):
        """绘制训练数据
        参数:
        - PP: 训练数据坐标
        - z_R, z_G, z_B: RGB通道的训练数据值
        - i, j: 当前处理的图像块位置
        """
        plt.figure(figsize=(8, 8))
        
        # 将坐标转换为图像格式
        x_min, x_max = PP[0].min(), PP[0].max()
        y_min, y_max = PP[1].min(), PP[1].max()
        
        # 创建网格
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # 将RGB值归一化到0-1范围
        rgb_values = np.column_stack([z_R, z_G, z_B])
        rgb_values = np.clip(rgb_values, 0, 1)
        
        # 创建RGB图像
        rgb_image = np.zeros((100, 100, 3))
        
        # 使用最近邻插值填充RGB值
        for k in range(3):
            rgb_image[:,:,k] = griddata(
                (PP[0], PP[1]), 
                rgb_values[:,k], 
                (X, Y), 
                method='nearest',
                fill_value=None
            )
        
        # 显示RGB图像
        plt.imshow(rgb_image, extent=[x_min, x_max, y_min, y_max])
        plt.title(f'RGB Training Data\nBlock ({i}, {j})')
        
        plt.tight_layout()
        plt.show()

    def calculate_pixel_differences(self, image_name):
        """计算每个像素点与其周围像素点的差异
        参数:
        - image_name (str): 图像文件名
        返回:
        - diff_map (ndarray): 每个像素点的差异值
        """
        # 读取图像
        im = mpimg.imread(image_name)
        im = im.astype(float) / 255  # 归一化图像
        
        height, width = im.shape[:2]
        diff_map = np.zeros((height, width))
        
        # 定义8个方向的偏移量（上、下、左、右、左上、右上、左下、右下）
        directions = [(-1,0), (1,0), (0,-1), (0,1), 
                     (-1,-1), (-1,1), (1,-1), (1,1)]
        
        # 对每个像素点计算与周围像素的差异
        for i in range(1, height-1):
            for j in range(1, width-1):
                pixel_diffs = []
                for di, dj in directions:
                    # 计算当前像素与周围像素的RGB差异
                    diff = im[i,j] - im[i+di,j+dj]
                    # 计算差异的平方和
                    diff_squared = np.sum(diff**2)
                    pixel_diffs.append(diff_squared)
                
                # 计算平均差异
                diff_map[i,j] = np.mean(pixel_diffs) / 3  # 除以3是因为RGB三个通道
        
        # 创建新的图形
        plt.figure(figsize=(30, 8))
        
        # 显示差异图
        plt.subplot(1, 3, 1)
        plt.imshow(diff_map, cmap='hot')
        plt.colorbar(label='Average Pixel Difference')
        plt.title('Average Pixel Differences with Neighbors', fontsize=12)
        plt.axis('off')
        
        # 显示差异值大于阈值的像素点的RGB分布
        plt.subplot(1, 3, 2)
        # 设置差异阈值
        diff_threshold = np.mean(diff_map) + np.std(diff_map)
        # 找出差异值大于阈值的像素点
        high_diff_mask = diff_map > diff_threshold
        high_diff_pixels = im[high_diff_mask]
        
        # 计算RGB值的频率
        rgb_values = high_diff_pixels * 255  # 转换回0-255范围
        rgb_values = rgb_values.astype(int)  # 转换为整数
        
        # 创建RGB值的标签
        rgb_labels = [f'({r}, {g}, {b})' for r, g, b in rgb_values]
        
        # 统计每个RGB组合的频率
        unique_rgb, counts = np.unique(rgb_labels, return_counts=True)
        
        # 按频率降序排序
        sorted_indices = np.argsort(counts)[::-1]
        top_n = min(20, len(unique_rgb))  # 显示前20个最频繁的RGB值
        top_rgb = unique_rgb[sorted_indices[:top_n]]
        top_counts = counts[sorted_indices[:top_n]]
        
        # 创建颜色块
        colors = []
        for rgb_str in top_rgb:
            # 从字符串中提取RGB值
            rgb = eval(rgb_str)  # 将字符串转换为元组
            colors.append([c/255 for c in rgb])  # 归一化到0-1范围
        
        # 绘制条形图
        bars = plt.bar(range(top_n), top_counts)
        
        plt.title(f'RGB Distribution of High-Diff Pixels\n(threshold={diff_threshold:.4f})', fontsize=12)
        plt.xlabel('RGB Values', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        
        # 添加RGB值标签
        plt.xticks(range(top_n), top_rgb, rotation=45, ha='right')
        
        # 添加统计信息
        total_high_diff = np.sum(high_diff_mask)
        plt.text(0.02, 0.98, f'Total High-Diff Pixels: {total_high_diff}\nThreshold: {diff_threshold:.4f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        
        # 显示20个RGB值对应的真实颜色
        plt.subplot(1, 3, 3)
        # 创建颜色展示图像 (改为更宽的布局)
        color_display = np.zeros((50, 200, 3))  # 改为200像素宽
        for i, color in enumerate(colors):
            # 每个颜色占据10像素宽度
            color_display[:, i*10:(i+1)*10] = color
        
        plt.imshow(color_display)
        plt.title('Color Display of Top 20 RGB Values', fontsize=12)
        plt.axis('off')
        
        # 添加RGB值标签（竖直显示）
        for i, rgb_str in enumerate(top_rgb):
            # 将RGB值字符串按行分割
            rgb_lines = rgb_str.split(',')
            for j, line in enumerate(rgb_lines):
                plt.text(i*10 + 5, 15 + j*8, line.strip(), 
                        ha='center', 
                        va='center',
                        color='white' if np.mean(colors[i]) < 0.5 else 'black',
                        fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return diff_map

if __name__ == "__main__":
    # 设置图像路径
    image_name = "/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/processed/unsplash_7oLemmP3XVk.tif"
    oimage = Image.open("/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/original/unsplash_7oLemmP3XVk.tif")
    image = Image.open(image_name)

    # 设置修复参数
    spacing = 3  # 采样点间距 3
    width = 2    # 采样范围宽度 2
    l2_coef = 0.5  # L2正则化系数 0.5
    
    # 使用RBF方法进行图像修复
    inpainter = ImageInpainting(model_type='RBF')
    from main import ImageInpainting
    inpainter_alter = ImageInpainting(model_type='KMS')
    
    # 像素差异 绘图 ###########################################
    diff_map = inpainter.calculate_pixel_differences(image_name)
    #########################################################


    # 获取用户输入的fill_rgb值
    print("\n请输入需要填充的RGB值（范围0-255，用空格分隔）：")
    while True:
        try:
            r, g, b = map(int, input().split())
            if all(0 <= x <= 255 for x in [r, g, b]):
                fill_rgb = np.array([r, g, b])
                break
            else:
                print("RGB值必须在0-255范围内，请重新输入：")
        except ValueError:
            print("请输入三个整数，用空格分隔：")
    
    # 修复
    im_rec1 = inpainter_alter.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.05)

    # 找出异常点的RGB
    inpainter.plot_top_pixel_frequencies(image_name)
    
    # 显示修复结果对比
    plt.figure(figsize=(20, 8))
    plt.imshow(im_rec1)
    plt.axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()
    

