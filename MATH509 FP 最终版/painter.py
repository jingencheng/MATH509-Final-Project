import requests
import os
import time
import random
from urllib.parse import urljoin
import json
from PIL import Image, ImageDraw, ImageFont
import math

class UnsplashCrawlerWithPositions:
    def __init__(self, watermark_text=None, num_watermarks=10, output_format="tif", test_mode=False):
        # 创建必要的目录
        self.base_dir = "/Users/ec/Desktop/MATH509 FP 算法预处理/unsplash_images"
        self.original_dir = os.path.join(self.base_dir, "original")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.test_dir = os.path.join(self.base_dir, "test")
        
        for dir_path in [self.original_dir, self.processed_dir, self.test_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
        # 水印设置
        self.watermark_text = watermark_text  # 允许为None，表示使用图片编号
        self.num_watermarks = num_watermarks
        self.font_path = "/System/Library/Fonts/Supplemental/Arial Narrow.ttf"
        # self.text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.text_color = (255, 0, 0) # 红色
        # 输出格式设置
        self.output_format = output_format.lower()
        if not self.output_format.startswith('.'):
            self.output_format = '.' + self.output_format
            
        # 存储水印位置信息
        self.watermark_positions = []
        
        # 水印信息文件路径
        self.watermark_info_file = "watermark_info.json"
        
        # 测试模式设置
        self.test_mode = test_mode
        
        # 初始化水印信息文件
        if not os.path.exists(self.watermark_info_file):
            with open(self.watermark_info_file, 'w', encoding='utf-8') as f:
                json.dump({"watermark_info": []}, f, ensure_ascii=False, indent=4)

    def save_watermark_info(self, watermark_info):
        """保存水印信息到JSON文件"""
        try:
            # 读取现有的水印信息
            with open(self.watermark_info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 添加新的水印信息
            data["watermark_info"].extend(watermark_info)
            
            # 保存更新后的信息
            with open(self.watermark_info_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            print(f"水印信息已保存到: {self.watermark_info_file}")
        except Exception as e:
            print(f"保存水印信息失败: {str(e)}")

    def add_text_watermark(self, img, text, font):
        """在图片上均匀分布指定数量的水印"""
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        # 获取单个文字的大小
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 计算网格大小
        grid_size = int(math.sqrt(self.num_watermarks))
        h_spacing = width / (grid_size + 1)
        v_spacing = height / (grid_size + 1)
        
        # 清空之前的位置记录
        self.watermark_positions = []
        
        # 均匀分布水印
        for i in range(self.num_watermarks):
            row = i // grid_size
            col = i % grid_size
            
            # 计算基础位置
            x = (col + 1) * h_spacing
            y = (row + 1) * v_spacing
            
            # 添加随机偏移，使水印不会完全对齐
            x_offset = random.randint(-int(h_spacing/4), int(h_spacing/4))
            y_offset = random.randint(-int(v_spacing/4), int(v_spacing/4))
            
            x += x_offset
            y += y_offset
            
            # 确保水印在图片范围内
            x = max(text_width/2, min(width - text_width/2, x))
            y = max(text_height/2, min(height - text_height/2, y))
            
            # 记录水印位置
            position = {
                'top_left': (x - text_width/2, y - text_height/2),
                'top_right': (x + text_width/2, y - text_height/2),
                'bottom_left': (x - text_width/2, y + text_height/2),
                'bottom_right': (x + text_width/2, y + text_height/2)
            }
            self.watermark_positions.append(position)
            
            # 绘制水印
            draw.text((x - text_width/2, y - text_height/2), text, fill=self.text_color, font=font)
        
        return img

    def draw_watermark_box(self, img, position, color=(255, 0, 0), width=2):
        """在图片上绘制水印框"""
        draw = ImageDraw.Draw(img)
        
        # 底部向下偏移5个像素
        bottom_offset = 15
        
        # 获取原始坐标
        top_left = position['top_left']
        top_right = position['top_right']
        bottom_left = position['bottom_left']
        bottom_right = position['bottom_right']
        
        # 调整底部坐标
        bottom_left = (bottom_left[0], bottom_left[1] + bottom_offset)
        bottom_right = (bottom_right[0], bottom_right[1] + bottom_offset)
        
        # 绘制四条边
        # 上边
        draw.line([top_left, top_right], fill=color, width=width)
        # 右边
        draw.line([top_right, bottom_right], fill=color, width=width)
        # 下边
        draw.line([bottom_right, bottom_left], fill=color, width=width)
        # 左边
        draw.line([bottom_left, top_left], fill=color, width=width)
        
        return img

    def process_image(self, input_path):
        """处理图片：RGBA转RGB并添加水印"""
        try:
            with Image.open(input_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                width, height = img.size
                font_size = int(min(width, height) * 0.05)  # 固定字体大小为图片最小边长的5%
                font = ImageFont.truetype(self.font_path, font_size)
                
                # 获取图片文件名（不含扩展名）作为水印文本
                filename = os.path.basename(input_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # 如果未指定水印文本，使用图片编号
                if self.watermark_text is None:
                    # 从文件名中提取编号（例如从 "unsplash_7oLemmP3XVk" 提取 "7oLemmP3XVk"）
                    watermark_text = name_without_ext.split('_')[1] if '_' in name_without_ext else name_without_ext
                else:
                    watermark_text = self.watermark_text
                
                # 添加水印
                img = self.add_text_watermark(img, watermark_text, font)
                
                # 如果是测试模式，绘制水印框
                if self.test_mode:
                    for position in self.watermark_positions:
                        img = self.draw_watermark_box(img, position)
                
                # 根据模式选择输出目录
                output_dir = self.test_dir if self.test_mode else self.processed_dir
                output_filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, output_filename)
                
                img.save(output_path, format='TIFF', compression='tiff_lzw')
                print(f"图片处理完成: {output_filename}")
                return True, self.watermark_positions
        except Exception as e:
            print(f"图片处理失败 {input_path}: {str(e)}")
            return False, []

    def process_all_images(self):
        """处理original目录下的所有.tif图片"""
        try:
            # 获取所有.tif文件
            tif_files = [f for f in os.listdir(self.original_dir) if f.endswith('.tif')]
            total_files = len(tif_files)
            processed_count = 0
            watermark_info = []
            
            print(f"找到 {total_files} 个.tif文件")
            
            for filename in tif_files:
                input_path = os.path.join(self.original_dir, filename)
                print(f"\n处理图片: {filename}")
                
                success, positions = self.process_image(input_path)
                if success:
                    processed_count += 1
                    watermark_info.append({
                        'filename': filename,
                        'positions': positions
                    })
                time.sleep(0.1)  # 短暂延迟，避免系统负载过高
            
            print(f"\n处理完成:")
            print(f"总文件数: {total_files}")
            print(f"成功处理: {processed_count}")
            
            # 保存水印信息
            if watermark_info:
                self.save_watermark_info(watermark_info)
            
            return watermark_info
            
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            return []

if __name__ == "__main__":
    # 创建实例
    processor = UnsplashCrawlerWithPositions(
        watermark_text="math509",  # 设置为None表示使用图片编号作为水印
        num_watermarks=2,
        output_format=".tif",
        test_mode=False  # 启用/不启用 测试模式
    )
    
    # 处理所有图片
    processor.process_all_images()
    