import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import json
import os
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel
from cnocr import CnOcr
import hanlp

# 初始化OCR和HanLP模型
os.makedirs('./database/model', exist_ok=True)
ocr = CnOcr()
hanlp_model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH, download_dir='./database/model')  # 加载多任务模型

def process_image_text(image_path):
    """处理图片中的文字，返回OCR结果和语义分析结果"""
    # OCR识别
    result = ocr.ocr(image_path)
    text = ' '.join([line['text'] for line in result])
    
    # HanLP语义分析
    semantic_analysis = hanlp_model([text])
    
    return {
        'ocr_result': result,
        'semantic_analysis': {
            'tokens': semantic_analysis['tok/fine'][0],
            'pos_tags': semantic_analysis['pos/ctb'][0],
            'ner': semantic_analysis['ner/msra'][0],
            'dep': semantic_analysis['dep'][0]
        }
    }

class MemeDataset(Dataset):
    def __init__(self, data_dir, img_dir):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.samples = []
        
        # 遍历所有结果文件
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    result_path = os.path.join(root, file)
                    try:
                        with open(result_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 验证必要的字段是否存在
                            required_fields = ['path', 'ocr_result', 'analysis', 'human_annotation']
                            if all(field in data for field in required_fields):
                                if self.validate_data_structure(data):
                                    self.samples.append(data)
                                else:
                                    print(f"数据结构验证失败: {file}")
                            else:
                                print(f"缺少必要字段: {file}")
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误 {file}: {str(e)}")
                        with open(result_path, 'r', encoding='utf-8') as f:
                            problematic_line = f.readlines()[e.lineno - 1]
                            print(f"问题行: {problematic_line}")
                    except Exception as e:
                        print(f"加载文件 {file} 时出错: {str(e)}")
                        continue
        
        if not self.samples:
            raise RuntimeError("没有找到有效的训练数据！请确保数据集已正确标注。")
        
        print(f"成功加载 {len(self.samples)} 个样本")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
          # 文本处理
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./database/model')
    
    @staticmethod
    def validate_data_structure(data):
        """验证数据结构的完整性"""
        try:
            # 验证图像路径
            if not isinstance(data['path'], str):
                return False
                
            # 验证OCR结果
            if not isinstance(data['ocr_result'], dict) or 'text_blocks' not in data['ocr_result']:
                return False
                
            # 验证分析结果
            if not isinstance(data['analysis'], dict):
                return False
            if 'semantic_features' not in data['analysis']:
                return False
                
            # 验证人工标注
            if not isinstance(data['human_annotation'], dict):
                return False
            required_annotations = ['scene', 'emotion', 'intent', 'social_context']
            if not all(field in data['human_annotation'] for field in required_annotations):
                return False
                
            return True
        except Exception:
            return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        img_path = os.path.join(self.img_dir, os.path.basename(sample['path']))
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 处理OCR文本
        ocr_text = ' '.join([block['text'] for block in sample['ocr_result']['text_blocks']])
        
        # 处理语义特征
        semantic_features = sample['analysis']['semantic_features']
        tokens = semantic_features['tokens']
        pos_tags = semantic_features['pos_tags']
        
        # 编码文本
        encoded = self.tokenizer(ocr_text, padding='max_length', max_length=32,
                               truncation=True, return_tensors='pt')
        
        # 构建语义特征向量
        semantic_vec = torch.zeros(256)  # 预设大小的特征向量
        try:
            # 情感分析特征
            sentiment_score = 1.0 if sample['analysis']['sentiment_analysis']['overall_tone'] == 'positive' else \
                            0.0 if sample['analysis']['sentiment_analysis']['overall_tone'] == 'negative' else 0.5
            semantic_vec[0] = sentiment_score
            semantic_vec[1] = sample['analysis']['sentiment_analysis']['confidence']
            
            # 文本结构特征
            semantic_vec[2] = sample['analysis']['text_structure']['word_count'] / 10.0  # 归一化
            semantic_vec[3] = sample['analysis']['text_structure']['sentence_count'] / 5.0  # 归一化
            
            # 上下文提示特征
            context_formality = 1.0 if sample['analysis']['context_hints']['formality_level'] == 'formal' else \
                              0.0 if sample['analysis']['context_hints']['formality_level'] == 'informal' else 0.5
            semantic_vec[4] = context_formality
        except Exception as e:
            print(f"处理语义特征时出错: {e}")
        
        # 处理标签
        labels = {
            'scene': self.scene_to_idx(sample['human_annotation']['scene']),
            'emotion': self.emotion_to_idx(sample['human_annotation']['emotion']),
            'intent': self.intent_to_idx(sample['human_annotation']['intent']),
            'social_context': self.context_to_idx(sample['human_annotation']['social_context'])
        }
        
        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'semantic_features': semantic_vec,
            'labels': labels
        }
    
    @staticmethod
    def scene_to_idx(scene):
        scenes = {'工作': 0, '生活': 1, '社交': 2, '娱乐': 3}
        return scenes.get(scene, 0)
    
    @staticmethod
    def emotion_to_idx(emotion):
        emotions = {'积极': 0, '消极': 1, '中性': 2}
        return emotions.get(emotion, 2)
    
    @staticmethod
    def intent_to_idx(intent):
        intents = {'吐槽': 0, '安慰': 1, '庆祝': 2, '调侃': 3, '自嘲': 4}
        return intents.get(intent, 0)
    
    @staticmethod
    def context_to_idx(context):
        contexts = {'正式': 0, '非正式': 1, '亲密': 2}
        return contexts.get(context, 1)

class MemeAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()        # 图像编码器 
        torch.hub.set_dir('./database/model')
        self.image_encoder = torch.hub.load('pytorch/vision:v0.10.0','resnet50', weights=ResNet50_Weights.DEFAULT)
        self.image_encoder.fc = nn.Linear(2048, 512)
          # 文本编码器 (BERT)
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese', cache_dir='./database/model')
        self.text_projector = nn.Linear(768, 512)
        
        # 语义特征处理
        self.semantic_encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(1280, 1024),  # 512 + 512 + 256 = 1280
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多任务分类器
        self.classifier = nn.ModuleDict({
            'scene': nn.Linear(1024, 4),      # 工作、生活、社交、娱乐
            'emotion': nn.Linear(1024, 3),    # 积极、消极、中性
            'intent': nn.Linear(1024, 5),     # 吐槽、安慰、庆祝、调侃、自嘲
            'social_context': nn.Linear(1024, 3)  # 正式、非正式、亲密
        })
        
    def forward(self, image, input_ids, attention_mask, semantic_features):
        # 图像特征
        img_features = self.image_encoder(image)
        
        # 文本特征
        text_output = self.text_encoder(input_ids=input_ids,
                                      attention_mask=attention_mask)
        text_features = self.text_projector(text_output.pooler_output)
        
        # 语义特征
        semantic_vec = self.semantic_encoder(semantic_features)
        
        # 特征融合
        combined = torch.cat([img_features, text_features, semantic_vec], dim=1)
        combined = self.fusion_layer(combined)
        
        # 多任务预测
        return {
            'scene': self.classifier['scene'](combined),
            'emotion': self.classifier['emotion'](combined),
            'intent': self.classifier['intent'](combined),
            'social_context': self.classifier['social_context'](combined)
        }

def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 准备数据
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            semantic_features = batch['semantic_features'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            # 前向传播
            outputs = model(images, input_ids, attention_mask, semantic_features)
            
            # 计算损失
            loss = sum(criterion(outputs[k], labels[k]) for k in outputs)
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                semantic_features = batch['semantic_features'].to(device)
                labels = {k: v.to(device) for k, v in batch['labels'].items()}
                
                outputs = model(images, input_ids, attention_mask, semantic_features)
                val_loss += sum(criterion(outputs[k], labels[k]) for k in outputs).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}：Training Loss: {total_loss/len(train_loader):.4f} Validation Loss: {val_loss/len(val_loader):.4f}')

def main():
    # 数据加载
    dataset = MemeDataset('Trained_result', 'img')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
      # 创建模型
    model = MemeAnalyzer()
    
    # 训练模型
    train_model(model, train_loader, val_loader)
    
    # 确保模型保存目录存在
    os.makedirs('./model', exist_ok=True)
    
    # 保存模型状态和tokenizer
    tokenizer = dataset.tokenizer  # 获取数据集中的tokenizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer
    }, './model/meme_analyzer.pth')
    print('Model have saved to model')

if __name__ == "__main__":
    main()
