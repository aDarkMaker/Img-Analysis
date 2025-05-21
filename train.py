import os
import sys
from pathlib import Path

def setup_environment():
    """确保所有必要的目录存在"""
    directories = ['img', 'ocr_result', 'Trained_result']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    print("=== 开始表情包处理流程 ===")
    
    # 确保环境准备就绪
    setup_environment()
    
    # 1. 运行OCR处理
    print("\n[1/2] 开始OCR文字识别...")
    try:
        from OCR.src.image_ocr import ImageOCR
        ocr_processor = ImageOCR()
        ocr_processor.process_all_images()
        print("OCR处理完成！")
    except Exception as e:
        print(f"OCR处理时发生错误: {str(e)}")
        sys.exit(1)
    
    # 2. 运行语义分析
    print("\n[2/2] 开始HanLP语义分析...")
    try:
        from HanLP.src.semantic_analysis import SemanticAnalyzer
        analyzer = SemanticAnalyzer()
        analyzer.process_all_ocr_results()
        print("语义分析完成！")
    except Exception as e:
        print(f"语义分析时发生错误: {str(e)}")
        sys.exit(1)
    
    print("\n=== 所有处理完成！===")
    print(f"处理结果保存在 {os.path.abspath('Trained_result')} 目录中")

if __name__ == "__main__":
    main()
