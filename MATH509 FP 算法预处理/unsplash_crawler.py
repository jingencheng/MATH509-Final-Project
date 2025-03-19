import requests
import os
import time
import random
from urllib.parse import urljoin
import json
from PIL import Image, ImageDraw, ImageFont

class UnsplashCrawler:
    def __init__(self, watermark_text="MATH509", font_size_ratio=0.01, spacing_ratio=2.5, output_format="tif"):
        # Unsplash API配置
        self.api_base_url = "https://api.unsplash.com"
        # 请替换为你的Access Key
        self.access_key = "py82ZShK0sJTmIbS99oipvaaBMq6jrBx3QyCezWA0p4"
        self.headers = {
            "Authorization": f"Client-ID {self.access_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        # 创建必要的目录
        self.base_dir = "unsplash_images"
        self.original_dir = os.path.join(self.base_dir, "original")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        
        for dir_path in [self.original_dir, self.processed_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        # 水印设置
        self.watermark_text = watermark_text
        self.font_size_ratio = font_size_ratio
        self.spacing_ratio = spacing_ratio
        self.font_path = "/System/Library/Fonts/Supplemental/Arial Narrow.ttf"
        self.text_color = (255, 0, 0)  # 红色
        
        # 输出格式设置
        self.output_format = output_format.lower()
        if not self.output_format.startswith('.'):
            self.output_format = '.' + self.output_format

    def add_text_watermark(self, img, text, font, spacing_ratio):
        """在整个图片上添加随机分布的文字水印"""
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # 获取单个文字的大小
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算基础间距
        h_spacing = int(text_width * spacing_ratio)
        v_spacing = int(text_height * spacing_ratio)
        
        # 计算网格
        cols = width // h_spacing
        rows = height // v_spacing
        
        # 随机选择一些位置绘制文字
        for row in range(rows):
            # 随机决定是否绘制这一行
            if random.random() < 0.6:  # 60%的概率绘制这一行
                # 随机决定这一行绘制多少个文字
                num_words = random.randint(cols // 3, cols // 2)
                # 随机选择位置
                col_positions = random.sample(range(cols), num_words)
                
                for col in col_positions:
                    # 添加一些随机偏移
                    x_offset = random.randint(-h_spacing//4, h_spacing//4)
                    y_offset = random.randint(-v_spacing//4, v_spacing//4)
                    
                    x = col * h_spacing + x_offset
                    y = row * v_spacing + y_offset
                    
                    # 确保文字不会超出图片边界
                    if 0 <= x <= width-text_width and 0 <= y <= height-text_height:
                        draw.text((x, y), text, fill=self.text_color, font=font)
        
        return img

    def process_image(self, input_path):
        """处理图片：RGBA转RGB并添加水印"""
        try:
            with Image.open(input_path) as img:
                # 检查并转换RGBA为RGB
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 获取图片尺寸
                width, height = img.size
                
                # 设置字体大小
                font_size = int(min(width, height) * self.font_size_ratio)
                font = ImageFont.truetype(self.font_path, font_size)
                
                # 添加水印
                img = self.add_text_watermark(img, self.watermark_text, font, self.spacing_ratio)
                
                # 保存处理后的图片，修改扩展名为.tif
                output_filename = os.path.splitext(os.path.basename(input_path))[0] + self.output_format
                output_path = os.path.join(self.processed_dir, output_filename)
                
                # 保存为TIFF格式
                img.save(output_path, format='TIFF', compression='tiff_lzw')
                print(f"图片处理完成: {output_filename}")
                return True
        except Exception as e:
            print(f"图片处理失败 {input_path}: {str(e)}")
            return False

    def download_image(self, img_url, filename):
        """下载图片到original目录"""
        try:
            response = requests.get(img_url, headers=self.headers)
            if response.status_code == 200:
                # 先将图片数据转换为PIL Image对象
                from io import BytesIO
                img = Image.open(BytesIO(response.content))
                
                # 如果是RGBA格式，转换为RGB
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 修改文件名为.tif格式
                filename = os.path.splitext(filename)[0] + self.output_format
                file_path = os.path.join(self.original_dir, filename)
                
                # 保存为TIFF格式
                img.save(file_path, format='TIFF', compression='tiff_lzw')
                print(f"成功下载: {filename}")
                return file_path
        except Exception as e:
            print(f"下载失败 {img_url}: {str(e)}")
        return None

    def crawl(self, num_images=10):
        try:
            # 使用Unsplash API获取图片列表
            endpoint = f"{self.api_base_url}/photos/random"
            params = {
                "count": num_images
            }
            
            response = requests.get(endpoint, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code}")
                print("请确保已设置正确的Access Key")
                return
                
            photos = response.json()
            downloaded_count = 0
            processed_count = 0
            
            for photo in photos:
                # 获取原始图片URL
                img_url = photo['urls']['regular']
                # 获取图片ID作为文件名的一部分
                photo_id = photo['id']
                filename = f"unsplash_{photo_id}.jpg"
                
                # 下载图片
                downloaded_path = self.download_image(img_url, filename)
                if downloaded_path:
                    downloaded_count += 1
                    # 处理图片
                    if self.process_image(downloaded_path):
                        processed_count += 1
                    # 添加延迟以遵守API限制
                    time.sleep(1)
            
            print(f"\n总共成功下载 {downloaded_count} 张图片")
            print(f"成功处理 {processed_count} 张图片")
            
        except Exception as e:
            print(f"爬取过程中出错: {str(e)}")

    def process_specific_file(self, filename):
        """处理original目录下指定的tif文件
        
        Args:
            filename (str): 要处理的文件名（包含.tif扩展名）
            
        Returns:
            bool: 处理是否成功
        """
        input_path = os.path.join(self.original_dir, filename)
        if not os.path.exists(input_path):
            print(f"文件不存在: {input_path}")
            return False
            
        if not filename.lower().endswith('.tif'):
            print(f"文件格式错误，只支持.tif文件: {filename}")
            return False
            
        return self.process_image(input_path)

if __name__ == "__main__":
    # 可以自定义水印文字和大小比例
    crawler = UnsplashCrawler(
        watermark_text="MATH509",  # 水印文字
        font_size_ratio=0.1,  # 字体大小比例
        spacing_ratio=1.2,  # 文字间距比例
        output_format=".tif"  # 输出格式
    )
    
    # 密集小字体 效果不好 0.01， 1
    # 稀疏小字体 效果一般 0.01， 10
    # 稍微密集大字体 效果显著 0.1， 1 可以用来举例 困难例子
    # 稀疏大字体 效果显著 0.1， 1.2 可以用来举例 简单例子

    # 处理指定文件示例
    # crawler.process_specific_file("/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images/original/unsplash_BBU_fYagADI.tif")
    crawler.crawl(num_images=1)  # 下载1张图片 