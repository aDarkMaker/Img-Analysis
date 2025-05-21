import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
from transformers import BertTokenizer
import hanlp

# 尝试从 model_training.py 导入 MemeAnalyzer
# 确保 model_training.py 在同一目录下或 PYTHONPATH 中
try:
    from model_training import MemeAnalyzer
except ImportError as e:
    print(f"无法导入 MemeAnalyzer : {e}")
    exit(1)

# --- 全局配置和模型初始化 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './model/meme_analyzer.pth'
TOKENIZER_PATH = 'bert-base-chinese' # 与训练时一致
HANLP_MODEL_PATH = hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
CACHE_DIR = './database/model'
MEME_DATA_DIR = 'Trained_result' # 包含JSON文件的目录
IMG_DIR = 'img' # 包含图片文件的目录

# 初始化外部模型 (确保路径和训练时一致)
# OCR_MODEL = CnOcr() # 如果需要动态OCR，则取消注释
# HANLP_MODEL = hanlp.load(HANLP_MODEL_PATH, download_dir=CACHE_DIR) # 如果需要动态处理文本，则取消注释
TOKENIZER = BertTokenizer.from_pretrained(TOKENIZER_PATH, cache_dir=CACHE_DIR)

# 图像预处理转换 (与 MemeDataset 中定义一致)
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 类别映射 (与 MemeDataset 中定义一致，用于解码模型输出)
SCENES_MAP = {'工作': 0, '生活': 1, '社交': 2, '娱乐': 3}
EMOTIONS_MAP = {'积极': 0, '消极': 1, '中性': 2}
INTENTS_MAP = {'吐槽': 0, '安慰': 1, '庆祝': 2, '调侃': 3, '自嘲': 4}
CONTEXTS_MAP = {'正式': 0, '非正式': 1, '亲密': 2}

IDX_TO_SCENE = {v: k for k, v in SCENES_MAP.items()}
IDX_TO_EMOTION = {v: k for k, v in EMOTIONS_MAP.items()}
IDX_TO_INTENT = {v: k for k, v in INTENTS_MAP.items()}
IDX_TO_CONTEXT = {v: k for k, v in CONTEXTS_MAP.items()}

IDX_TO_MAPS = {
    'scene': IDX_TO_SCENE,
    'emotion': IDX_TO_EMOTION,
    'intent': IDX_TO_INTENT,
    'social_context': IDX_TO_CONTEXT
}

# --- 辅助函数 ---

def build_semantic_vector(analysis_json):
    """
    根据JSON中的'analysis'部分构建语义向量，与MemeDataset中的逻辑类似。
    """
    semantic_vec = torch.zeros(256) # 与MemeDataset中定义的大小一致
    try:
        if 'sentiment_analysis' in analysis_json and analysis_json['sentiment_analysis']:
            overall_tone = analysis_json['sentiment_analysis'].get('overall_tone')
            if overall_tone == 'positive': sentiment_score = 1.0
            elif overall_tone == 'negative': sentiment_score = 0.0
            else: sentiment_score = 0.5
            semantic_vec[0] = sentiment_score
            semantic_vec[1] = analysis_json['sentiment_analysis'].get('confidence', 0.0)

        if 'text_structure' in analysis_json and analysis_json['text_structure']:
            semantic_vec[2] = analysis_json['text_structure'].get('word_count', 0) / 10.0
            semantic_vec[3] = analysis_json['text_structure'].get('sentence_count', 0) / 5.0

        if 'context_hints' in analysis_json and analysis_json['context_hints']:
            formality = analysis_json['context_hints'].get('formality_level')
            if formality == 'formal': context_formality = 1.0
            elif formality == 'informal': context_formality = 0.0
            else: context_formality = 0.5
            semantic_vec[4] = context_formality
    except Exception as e:
        # print(f"构建语义向量时出错: {e} 对于数据: {analysis_json}")
        pass # 保持向量为零或部分填充
    return semantic_vec

def preprocess_meme_for_model(image_path, ocr_text, analysis_json, tokenizer, transform):
    """
    为单个表情包准备模型输入。
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
    except FileNotFoundError:
        # print(f"图片文件未找到: {image_path}")
        return None
    except Exception as e:
        # print(f"加载或转换图片时出错 {image_path}: {e}")
        return None

    encoded_text = tokenizer(ocr_text, padding='max_length', max_length=32, # 与训练时一致
                             truncation=True, return_tensors='pt')
    input_ids = encoded_text['input_ids'].squeeze(0)
    attention_mask = encoded_text['attention_mask'].squeeze(0)
    
    semantic_vec_tensor = build_semantic_vector(analysis_json)
    
    return image_tensor, input_ids, attention_mask, semantic_vec_tensor

def predict_meme_categories(model, device, image_tensor, input_ids, attention_mask, semantic_vec_tensor, idx_to_maps):
    """
    使用模型预测表情包的类别。
    """
    model.eval()
    with torch.no_grad():
        # 为批处理添加维度
        img_batch = image_tensor.unsqueeze(0).to(device)
        ids_batch = input_ids.unsqueeze(0).to(device)
        mask_batch = attention_mask.unsqueeze(0).to(device)
        sem_vec_batch = semantic_vec_tensor.unsqueeze(0).to(device)
        
        outputs = model(img_batch, ids_batch, mask_batch, sem_vec_batch)
        
        predictions = {}
        for task_name, task_output in outputs.items():
            pred_idx = torch.argmax(task_output, dim=1).cpu().item()
            predictions[task_name] = idx_to_maps[task_name].get(pred_idx, "未知")
    return predictions

def load_meme_database_with_predictions(data_dir, img_dir, model, device, tokenizer, transform, idx_to_maps):
    """
    加载所有表情包数据，并为每个表情包添加模型预测的类别。
    会递归搜索 data_dir 下的子目录以查找JSON文件。
    """
    full_meme_database = []
    print("正在加载和预处理表情包数据库...")
    if not os.path.isdir(data_dir):
        print(f"错误: 数据目录 {data_dir} 未找到。")
        return full_meme_database
        
    json_files_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files_paths.append(os.path.join(root, file))

    print(f"在 {data_dir} 及其子目录中找到 {len(json_files_paths)} 个JSON文件。")
    if not json_files_paths:
        print(f"警告: 在 {data_dir} 及其子目录中没有找到 .json 文件。")
        return full_meme_database

    for i, file_path in enumerate(json_files_paths):
        filename = os.path.basename(file_path)
        if (i + 1) % 50 == 0:
            print(f"  已处理 {i+1}/{len(json_files_paths)} 个JSON文件 ({filename})...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"调试: 加载JSON文件 {file_path} 失败: {e}")
            continue

        # 验证必要的字段是否存在 (与MemeDataset类似)
        required_fields = ['path', 'ocr_result', 'analysis', 'human_annotation']
        if not all(field in data for field in required_fields):
            print(f"调试: JSON文件 {file_path} 缺少必要字段。跳过。")
            missing = [field for field in required_fields if field not in data]
            print(f"  缺少: {missing}")
            continue
        
        # 从JSON中提取信息
        original_image_path = data['path'] # 这是原始路径，可能需要调整
        base_img_filename = os.path.basename(original_image_path)
        # 假设图片与JSON文件在img目录下的对应结构中，或者img目录是平坦的
        # 如果img目录也有与Trained_result类似的子目录结构，这里的路径拼接逻辑可能需要调整
        current_image_path = os.path.join(img_dir, base_img_filename) 

        if not os.path.exists(current_image_path):
            # 尝试另一种常见的模式：图片与JSON文件在同一原始子目录结构下，但根目录是img_dir
            # 例如 Trained_result/1/abc.json -> img/1/abc.jpg (如果原始路径是 ./1/abc.jpg)
            # 或者 Trained_result/1/abc.json -> img/abc.jpg (如果原始路径是 abc.jpg)
            relative_path_from_data_dir = os.path.relpath(os.path.dirname(file_path), data_dir)
            potential_image_path_in_subdir = os.path.join(img_dir, relative_path_from_data_dir, base_img_filename)
            
            if os.path.exists(potential_image_path_in_subdir):
                current_image_path = potential_image_path_in_subdir
            else:
                print(f"调试: 图片 {current_image_path} (尝试1) 和 {potential_image_path_in_subdir} (尝试2) (源自 {original_image_path} 在 {file_path} 中) 均不存在。跳过。")
                continue
            
        ocr_blocks = data.get('ocr_result', {}).get('text_blocks', [])
        ocr_text = ' '.join([block['text'] for block in ocr_blocks])
        analysis_json = data.get('analysis', {})
        human_annotations = data.get('human_annotation', {})

        processed_input = preprocess_meme_for_model(current_image_path, ocr_text, analysis_json, tokenizer, transform)
        if processed_input is None:
            print(f"调试: 文件 {filename} 的 preprocess_meme_for_model 返回 None。跳过。")
            continue
        
        image_tensor, input_ids, attention_mask, semantic_vec_tensor = processed_input
        
        model_predictions = predict_meme_categories(model, device, image_tensor, input_ids, attention_mask, semantic_vec_tensor, idx_to_maps)
        
        full_meme_database.append({
            'image_path': current_image_path,
            'ocr_text': ocr_text,
            'human_annotations': human_annotations,
            'model_predictions': model_predictions
        })
    print(f"表情包数据库加载完成，共 {len(full_meme_database)} 个条目。")
    return full_meme_database

# --- 用户查询处理 ---
KEYWORD_TO_CATEGORY_MAPPINGS = {
    'scene': {
        '工作': ['工作', '上班', '公司', '项目', '代码', '开会'],
        '生活': ['生活', '日常', '家里', '吃饭', '睡觉'],
        '社交': ['社交', '朋友', '聚会', '聊天'],
        '娱乐': ['娱乐', '游戏', '搞笑', '沙雕', '有趣', '好玩']
    },
    'emotion': {
        '积极': ['开心', '高兴', '快乐', '积极', '兴奋', '搞笑', '有趣', '喜欢'],
        '消极': ['难过', '伤心', '悲伤', '消极', '生气', '郁闷', '裂开'],
        '中性': ['中性', '一般', '平静', '无语']
    },
    'intent': {
        '吐槽': ['吐槽', '抱怨'],
        '安慰': ['安慰', '鼓励'],
        '庆祝': ['庆祝', '祝贺'],
        '调侃': ['调侃', '开玩笑', '皮一下'],
        '自嘲': ['自嘲', '我太难了']
    },
    'social_context': {
        '正式': ['正式', '严肃'],
        '非正式': ['非正式', '随便', '休闲'],
        '亲密': ['亲密', '好朋友']
    }
}

def parse_user_query_for_categories(user_query_text, keyword_map):
    user_query_text_lower = user_query_text.lower()
    target_categories = {}
    for category_type, mappings in keyword_map.items():
        for category_value, keywords in mappings.items():
            if any(keyword in user_query_text_lower for keyword in keywords):
                target_categories[category_type] = category_value
                break # 每个类别类型只取第一个匹配项
    return target_categories

def find_memes_by_category(target_categories, meme_database_with_predictions, max_results=5):
    """
    根据目标类别查找表情包。
    """
    if not target_categories:
        return []

    matching_memes = []
    for meme_entry in meme_database_with_predictions:
        match_score = 0
        # 主要根据模型预测进行匹配
        current_categories = meme_entry['model_predictions'] 
        # 也可以选择 human_annotations: current_categories = meme_entry['human_annotations']
        
        for cat_type, cat_value in target_categories.items():
            if cat_type in current_categories and current_categories[cat_type] == cat_value:
                match_score += 1
        
        # 要求所有指定的类别都匹配
        if match_score == len(target_categories):
            matching_memes.append(meme_entry['image_path'])
            if len(matching_memes) >= max_results:
                break
    return matching_memes

# --- 主函数 ---
def main():
    print(f"使用设备: {DEVICE}")

    # 加载训练好的模型
    try:
        torch.hub.set_dir(CACHE_DIR) # 确保hub模型下载到指定目录
        model = MemeAnalyzer() 
        # 加载模型权重时，确保MemeAnalyzer的定义与训练时完全一致
        # 设置 weights_only=False 以允许加载包含 tokenizer 的检查点
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        print("MemeAnalyzer 模型加载成功。")
    except FileNotFoundError:
        print(f"模型文件 {MODEL_PATH} 未找到。请确保路径正确并且模型已训练保存。")
        return
    except RuntimeError as e:
        print(f"加载模型状态字典时出错: {e}")
        print("这通常意味着模型的结构定义与保存的权重不匹配。请检查MemeAnalyzer的定义。")
        return
    except Exception as e:
        print(f"加载模型时发生未知错误: {e}")
        return

    # 加载并预处理整个表情包数据库（这可能需要一些时间）
    # 在实际应用中，这部分可以缓存起来
    meme_db = load_meme_database_with_predictions(
        MEME_DATA_DIR, IMG_DIR, model, DEVICE, 
        TOKENIZER, IMG_TRANSFORM, IDX_TO_MAPS
    )

    if not meme_db:
        print("未能加载表情包数据库，程序退出。")
        return

    print("\n欢迎使用表情包推荐系统！")
    print("输入描述来查找表情包 (例如 '搞笑的上班吐槽')。输入 'exit' 退出。")

    while True:
        try:
            user_input = input("> ") 
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue

            target_cats = parse_user_query_for_categories(user_input, KEYWORD_TO_CATEGORY_MAPPINGS)
            
            if not target_cats:
                print("未能从您的描述中识别出明确的类别。请尝试更具体的描述。")
                print(f"例如，您可以包含场景（工作、生活）、情感（开心、难过）、意图（吐槽、庆祝）等关键词。")
                continue
            
            print(f"正在查找类别: {target_cats}")
            recommended_paths = find_memes_by_category(target_cats, meme_db)
            
            if recommended_paths:
                print("为您找到以下表情包:")
                for path in recommended_paths:
                    print(f"  - {path}")
            else:
                print("抱歉，没有找到完全匹配的表情包。尝试更换描述或减少条件。")
                
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    main()
