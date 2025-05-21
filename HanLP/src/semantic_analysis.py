import os
import json
from pathlib import Path
import hanlp

class SemanticAnalyzer:
    def __init__(self):
        # 加载HanLP模型
        self.analyzer = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
        self.ocr_result_dir = Path('./ocr_result')
        self.analysis_result_dir = Path('./HanLP/result')
        self.trained_result_dir = Path('./Trained_result')

    def analyze_text(self, ocr_data):
        """对文本进行语义分析"""        
        analysis_result = {
            'path': ocr_data['path'],
            'ocr_result': ocr_data['ocr_result'],
            'analysis': {
                'semantic_features': {},
                'sentiment_analysis': {},
                'text_structure': {},
                'context_hints': {}
            }
        }
        
        # 获取原始文本
        texts = ocr_data['ocr_result']['raw_text']
        combined_text = ' '.join(texts)
        
        if combined_text.strip():
            try:
                # HanLP分析
                doc = self.analyzer(combined_text)
                
                # 1. 语义特征分析
                analysis_result['analysis']['semantic_features'] = {
                    'tokens': doc['tok/fine'],
                    'pos_tags': doc['pos/ctb'],
                    'named_entities': doc['ner/msra'],
                    'dependencies': doc['dep']
                }
                
                # 2. 情感分析
                # 基于词性和关键词进行简单情感判断
                sentiment_words = self._extract_sentiment_words(doc)
                analysis_result['analysis']['sentiment_analysis'] = {
                    'sentiment_words': sentiment_words,
                    'overall_tone': self._determine_tone(sentiment_words),
                    'confidence': 0.8  # 示例值，实际应基于分析结果
                }
                
                # 3. 文本结构分析
                analysis_result['analysis']['text_structure'] = {
                    'word_count': len(doc['tok/fine']),
                    'sentence_count': len([t for t in texts if t.strip()]),
                    'key_phrases': self._extract_key_phrases(doc),
                    'main_topics': self._extract_topics(doc)
                }
                
                # 4. 上下文提示
                analysis_result['analysis']['context_hints'] = {
                    'possible_scenarios': self._suggest_scenarios(doc),
                    'style_markers': self._identify_style(doc),
                    'formality_level': self._assess_formality(doc)
                }
                
                # 5. 添加自动标注建议
                analysis_result['suggested_annotation'] = {
                    'scene': self._suggest_scene(doc),
                    'emotion': self._suggest_emotion(analysis_result['analysis']['sentiment_analysis']),
                    'intent': self._suggest_intent(doc),
                    'social_context': self._suggest_social_context(analysis_result['analysis'])
                }
                
            except Exception as e:
                print(f"分析过程出错: {str(e)}")
                
        return analysis_result
    
    def _extract_sentiment_words(self, doc):
        """提取情感词"""
        sentiment_words = []
        pos_tags = doc['pos/ctb']
        words = doc['tok/fine']
        
        # 这里可以添加更复杂的情感词提取逻辑
        for word, pos in zip(words, pos_tags):
            if pos in ['VA', 'VV', 'JJ', 'AD']:  # 形容词、动词、情感副词等
                sentiment_words.append(word)
        
        return sentiment_words
    
    def _determine_tone(self, sentiment_words):
        """确定整体语气"""
        # 这里可以实现更复杂的语气判断逻辑
        return "neutral" if not sentiment_words else "informal"
    
    def _extract_key_phrases(self, doc):
        """提取关键短语"""
        key_phrases = []
        words = doc['tok/fine']
        pos_tags = doc['pos/ctb']
        
        current_phrase = []
        for word, pos in zip(words, pos_tags):
            if pos.startswith('N'):  # 名词短语
                current_phrase.append(word)
            else:
                if current_phrase:
                    key_phrases.append(''.join(current_phrase))
                    current_phrase = []
        
        if current_phrase:
            key_phrases.append(''.join(current_phrase))
        
        return key_phrases
    
    def _extract_topics(self, doc):
        """提取主题"""
        topics = []
        # 使用命名实体识别结果作为主题
        if 'ner/msra' in doc:
            for entity in doc['ner/msra']:
                if entity[-1] != 'O':  # 不是其他类型的实体
                    topics.append(entity[0])
        return topics
    
    def _suggest_scenarios(self, doc):
        """建议可能的使用场景"""
        # 基于关键词和语言风格推测可能的场景
        scenarios = []
        words = set(doc['tok/fine'])
        
        # 示例规则
        if {'工作', '会议', '项目'} & words:
            scenarios.append('工作场合')
        if {'朋友', '玩', '哈哈'} & words:
            scenarios.append('社交场合')
        if not scenarios:
            scenarios.append('日常生活')
        
        return scenarios
    
    def _identify_style(self, doc):
        """识别语言风格"""
        style_markers = []
        words = set(doc['tok/fine'])
        
        # 网络用语标记
        if {'666', '笑死', '哈哈'} & words:
            style_markers.append('网络用语')
        # 正式用语标记
        if {'您好', '请', '谢谢'} & words:
            style_markers.append('正式用语')
            
        return style_markers
    
    def _assess_formality(self, doc):
        """评估语言正式程度"""
        words = set(doc['tok/fine'])
        
        if {'您', '先生', '女士', '请'} & words:
            return 'formal'
        elif {'哈', '啦', '呢', '吧'} & words:
            return 'informal'
        else:
            return 'neutral'
    
    def _suggest_scene(self, doc):
        """建议场景分类"""
        scenarios = self._suggest_scenarios(doc)
        if '工作场合' in scenarios:
            return '工作'
        elif '社交场合' in scenarios:
            return '社交'
        else:
            return '生活'
    
    def _suggest_emotion(self, sentiment_analysis):
        """建议情感分类"""
        tone = sentiment_analysis['overall_tone']
        if tone == 'positive':
            return '积极'
        elif tone == 'negative':
            return '消极'
        else:
            return '中性'
    
    def _suggest_intent(self, doc):
        """建议意图分类"""
        words = set(doc['tok/fine'])
        
        if {'哈哈', '笑死'} & words:
            return '调侃'
        elif {'加油', '支持'} & words:
            return '鼓励'
        elif {'什么', '为什么'} & words:
            return '疑问'
        else:
            return '陈述'
        
    def _suggest_social_context(self, analysis):
        """建议社交语境"""
        if analysis['text_structure']['formality_level'] == 'formal':
            return '正式'
        elif '网络用语' in analysis['context_hints']['style_markers']:
            return '非正式'
        else:
            return '中性'

    def process_all_ocr_results(self):
        """处理所有OCR结果文件"""
        self.analysis_result_dir.mkdir(parents=True, exist_ok=True)
        self.trained_result_dir.mkdir(exist_ok=True)
        
        # 遍历OCR结果文件
        index = 1
        for ocr_file in self.ocr_result_dir.glob('*.json'):
            try:
                print(f"\n处理文件: {ocr_file}")
                
                # 读取OCR结果
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # 进行语义分析
                result = self.analyze_text(ocr_data)
                
                # 创建子目录
                trained_subdir = self.trained_result_dir / str(index)
                trained_subdir.mkdir(exist_ok=True)
                
                # 保存分析结果
                trained_path = trained_subdir / "analysis.json"
                with open(trained_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                index += 1
                print(f'成功分析 {ocr_file.name}')
                
            except Exception as e:
                print(f'处理 {ocr_file.name} 时出错: {str(e)}')

if __name__ == '__main__':
    analyzer = SemanticAnalyzer()
    analyzer.process_all_ocr_results()
