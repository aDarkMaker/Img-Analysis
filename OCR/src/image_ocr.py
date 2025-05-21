import os
import json
from pathlib import Path
from cnocr import CnOcr
import cv2

class ImageOCR:
    def __init__(self):
        # 初始化中文OCR识别器
        self.ocr = CnOcr(det_model_name='ch_PP-OCRv4_det_server')
        self.img_dir = Path('./img')
        self.ocr_result_dir = Path('./ocr_result')
    
    def extract_text_with_position(self, ocr_result):
        """从OCR结果中提取文字内容及其位置信息"""
        texts = []
        for item in ocr_result:
            text_info = {
                'text': item['text'],
                'position': item['position'],  # 文字在图片中的位置
                'confidence': float(item['score'])  # OCR的置信度
            }
            texts.append(text_info)
        return texts
        
    def process_single_image(self, img_path):
        """处理单张图片并生成对应的OCR结果"""
        try:
            print(f"\nProcessing image: {img_path}")
            # 读取图片信息
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]
            
            # 使用CnOcr进行识别
            ocr_out = self.ocr.ocr(str(img_path))
            
            # 提取文字内容和位置信息
            texts_with_info = self.extract_text_with_position(ocr_out)
            # 构建结果字典
            result = {
                'path': str(img_path),
                'ocr_result': {
                    'text_blocks': [{'text': item['text'], 'confidence': item['confidence']} for item in texts_with_info],
                    'raw_text': [item['text'] for item in texts_with_info],
                    'avg_confidence': sum(item['confidence'] for item in texts_with_info) / len(texts_with_info) if texts_with_info else 0
                }
            }
            
            print(f"Processed {img_path.name} with {len(texts_with_info)} text blocks")
            return result
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return None
    def process_all_images(self):
        """处理img文件夹中的所有图片"""
        self.ocr_result_dir.mkdir(exist_ok=True)
        
        for img_file in self.img_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                try:
                    # 生成对应的JSON文件名
                    json_filename = img_file.stem + '.json'
                    json_path = self.ocr_result_dir / json_filename
                    
                    # 处理图片
                    ocr_result = self.process_single_image(img_file)
                    
                    if ocr_result:                        # 直接使用OCR结果
                        result = ocr_result
                        
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(self._convert_to_json_serializable(result), f, ensure_ascii=False, indent=2)
                            
                        print(f'Successfully processed {img_file.name}')
                    
                except Exception as e:
                    print(f'Error processing {img_file.name}: {str(e)}')

    def _convert_to_json_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # 处理numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__str__'):  # 处理其他不可序列化的对象
            return str(obj)
        return obj

if __name__ == '__main__':
    ocr_processor = ImageOCR()
    ocr_processor.process_all_images()
