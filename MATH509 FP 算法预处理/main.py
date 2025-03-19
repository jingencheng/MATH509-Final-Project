import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from rsc.rbf_regression import RBFRegression
from rsc.gmm_weighted_mean import GMMWeightedMean
from rsc.k_means import KMeansWeightedMean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import time

class Timer:
    """简单的计时器类"""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def stop(self):
        """停止计时并返回持续时间"""
        self.end_time = time.time()
        return self.end_time - self.start_time

class ImageInpainting:
    """图像修复类
    支持多种修复算法：RBF, KNN, GMM, KMS, MLP
    """
    def __init__(self, model_type='RBF', **kwargs):
        """初始化图像修复类
        参数:
        - model_type: 使用的模型类型，可选 'RBF', 'KNN', 'GMM', 'KMS', 'MLP'
        - kwargs: 模型特定的参数
            - RBF: 无额外参数
            - KNN: n_neighbors (默认10)
            - GMM: n_components (默认3)
            - KMS: n_clusters (默认3)
            - MLP: hidden_layer_sizes (默认(64, 64))
        """
        self.model_type = model_type.upper()
        self.timer = Timer()  # 添加计时器
        
        # 根据模型类型初始化对应的模型
        if self.model_type == 'RBF':
            self.create_model = lambda centers, widths: RBFRegression(centers=centers, widths=widths)
        elif self.model_type == 'KNN':
            n_neighbors = kwargs.get('n_neighbors', 10)
            self.create_model = lambda: KNeighborsRegressor(n_neighbors=n_neighbors)
        elif self.model_type == 'GMM':
            n_components = kwargs.get('n_components', 3)
            self.create_model = lambda: GMMWeightedMean(n_components=n_components, random_state=0)
        elif self.model_type == 'KMS':
            n_clusters = kwargs.get('n_clusters', 3)
            self.create_model = lambda: KMeansWeightedMean(n_clusters=n_clusters, random_state=0)
        elif self.model_type == 'MLP':
            hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (64, 64))
            self.create_model = lambda: MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                                   activation='relu', 
                                                   solver='lbfgs', 
                                                   max_iter=500)
        # elif self.model_type == '':
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def inpaint(self, image_name, fill_rgb, spacing, width, l2_coef=0.5, tolerance=0.1, patch_size=25):
        """执行图像修复
        参数:
        - image_name (str): 需要修复的图像文件名
        - fill_rgb (list): 需要填充的RGB颜色值，范围0-255
        - spacing (int): 采样点之间的间距，范围1-9
        - width (float): 采样范围的宽度，范围1-2*spacing
        - l2_coef (float): L2正则化系数（仅用于RBF）
        - patch_size (int): 图像块大小
        - tolerance (float): 颜色相似度阈值
        """
        # 开始计时
        self.timer.start()
        
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

                # 为每个颜色通道创建模型
                if self.model_type == 'RBF':
                    red_model = self.create_model(centers, widths)
                    green_model = self.create_model(centers, widths)
                    blue_model = self.create_model(centers, widths)
                else:
                    red_model = self.create_model()
                    green_model = self.create_model()
                    blue_model = self.create_model()
                
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
                        print('*训练数据不足，直接复制图像块*')
                        patch_rec = patch_fill
                    else:
                        # 获取有效的训练数据位置
                        PP = P[:,idx_data]

                        # 对每个颜色通道进行模型拟合
                        # 红色通道
                        patch_R = patch[:,:,0]
                        z_R = patch_R.reshape(patch_R.size,1, order='F')
                        z_R = z_R[idx_data]
                        if self.model_type == 'RBF':
                            red_model.fit_with_l2_regularization(PP.T, z_R, l2_coef)
                        elif self.model_type == 'MLP':
                            red_model.fit(PP.T, z_R.ravel())
                        else:
                            red_model.fit(PP.T, z_R)
                        
                        # 绿色通道
                        patch_G = patch[:,:,1]
                        z_G = patch_G.reshape(patch_G.size,1, order='F')
                        z_G = z_G[idx_data]
                        if self.model_type == 'RBF':
                            green_model.fit_with_l2_regularization(PP.T, z_G, l2_coef)
                        elif self.model_type == 'MLP':
                            green_model.fit(PP.T, z_G.ravel())
                        else:
                            green_model.fit(PP.T, z_G)
                        
                        # 蓝色通道
                        patch_B = patch[:,:,2]
                        z_B = patch_B.reshape(patch_B.size,1, order='F')
                        z_B = z_B[idx_data]
                        if self.model_type == 'RBF':
                            blue_model.fit_with_l2_regularization(PP.T, z_B, l2_coef)
                        elif self.model_type == 'MLP':
                            blue_model.fit(PP.T, z_B.ravel())
                        else:
                            blue_model.fit(PP.T, z_B)
                        
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
            
        # 停止计时并获取用时
        self.time = self.timer.stop()
        return np.round(im_rec,4)

if __name__ == "__main__":
    # 设置图像路径
    image_name = "/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/processed/unsplash_BBU_fYagADI.tif"
    oimage = Image.open("/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/original/unsplash_BBU_fYagADI.tif")
    image = Image.open(image_name)

    # 设置修复参数
    fill_rgb = np.array([255, 0, 0])  # 需要填充的红色
    spacing = 3  # 采样点间距
    width = 2    # 采样范围宽度
    l2_coef = 0.5  # L2正则化系数（仅用于RBF）
    tolerance = 0.5  # 颜色相似度阈值 0.1 ~ 0.5

    # 使用不同方法进行图像修复
    inpainter_gmm = ImageInpainting(model_type='GMM')
    inpainter_knn = ImageInpainting(model_type='KNN')
    inpainter_rbf = ImageInpainting(model_type='RBF')
    inpainter_kms = ImageInpainting(model_type='KMS')
    inpainter_mlp = ImageInpainting(model_type='MLP')

    '''
    # TOL=0.5 所有算法效果极佳
    # TOL=0.1 才可以观察出区别
    # im_rec = inpainter_rbf.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    im_rec1 = inpainter_knn.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    im_rec1_alter = inpainter_knn.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.1)
    # im_rec2 = inpainter_rbf.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    im_rec3 = inpainter_kms.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    im_rec3_alter = inpainter_kms.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.1)
    im_rec4 = inpainter_mlp.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    im_rec4_alter = inpainter_mlp.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.1)

    # 显示修复结果对比
    plt.figure(figsize=(12, 8))
    
    # 原始图像
    plt.subplot(2, 3, 1)  
    plt.imshow(oimage)
    plt.title('original image')  
    plt.axis('off')

    # GMM修复结果
    plt.subplot(2, 3, 2) 
    plt.imshow(im_rec)
    plt.title('GMM')  
    plt.axis('off')  

    # KNN修复结果
    plt.subplot(2, 3, 3)
    plt.imshow(im_rec1)
    plt.title('KNN') 
    plt.axis('off')  




    # 原始图像
    plt.subplot(2, 3, 4)  
    plt.imshow(image)
    plt.title('destroyed image')  
    plt.axis('off')

    # KMS修复结果
    plt.subplot(2, 3, 5) 
    plt.imshow(im_rec3)
    plt.title('KMS') 
    plt.axis('off') 

    # MLP修复结果
    plt.subplot(2, 3, 6)
    plt.imshow(im_rec4)
    plt.title('MLP')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    '''

    
    
    # 显示修复结果对比
    plt.figure(figsize=(12, 8))
    
    # 原始图像
    im_rec1 = inpainter_knn.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    plt.subplot(2, 3, 1)  
    plt.imshow(im_rec1)
    plt.title('KNN\nTOL: 0.5\nTime: %.2fs' % inpainter_knn.time)  
    plt.axis('off')

    # GMM修复结果
    im_rec3 = inpainter_kms.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    plt.subplot(2, 3, 2) 
    plt.imshow(im_rec3)
    plt.title('KMS\nTOL: 0.5\nTime: %.2fs' % inpainter_kms.time)  
    plt.axis('off')  

    # KNN修复结果
    im_rec4 = inpainter_mlp.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance)
    plt.subplot(2, 3, 3)
    plt.imshow(im_rec4)
    plt.title('MLP\nTOL: 0.5\nTime: %.2fs' % inpainter_mlp.time) 
    plt.axis('off')  




    # 原始图像
    im_rec1_alter = inpainter_knn.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.1)
    plt.subplot(2, 3, 4)  
    plt.imshow(im_rec1_alter)
    plt.title('KNN\nTOL: 0.1\nTime: %.2fs' % inpainter_knn.time)  
    plt.axis('off')

    # KMS修复结果
    im_rec3_alter = inpainter_kms.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.1)
    plt.subplot(2, 3, 5) 
    plt.imshow(im_rec3_alter)
    plt.title('KMS\nTOL: 0.1\nTime: %.2fs' % inpainter_kms.time) 
    plt.axis('off') 

    # MLP修复结果
    im_rec4_alter = inpainter_mlp.inpaint(image_name, fill_rgb, spacing, width, l2_coef, tolerance=0.1)
    plt.subplot(2, 3, 6)
    plt.imshow(im_rec4_alter)
    plt.title('MLP\nTOL: 0.1\nTime: %.2fs' % inpainter_mlp.time)
    plt.axis('off')

    plt.tight_layout()
    plt.show()