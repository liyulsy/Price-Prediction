import json
from datetime import datetime
import os
from collections import Counter
import re
import time
import numpy as np
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class BertFeatureExtractor:
    def __init__(self, model_dir: str = "models"):
        # 创建模型保存目录
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 初始化BERT模型和分词器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'bert-base-uncased'
        
        # 设置模型保存路径
        self.model_path = os.path.join(model_dir, self.model_name)
        
        # 如果模型不存在，则下载并保存
        if not os.path.exists(self.model_path):
            print(f"正在下载BERT模型到 {self.model_path}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            
            # 保存模型和分词器
            self.tokenizer.save_pretrained(self.model_path)
            self.model.save_pretrained(self.model_path)
        else:
            print(f"正在加载本地BERT模型从 {self.model_path}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertModel.from_pretrained(self.model_path)
        
        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_bert_embeddings(self, text: str) -> np.ndarray:
        """
        使用BERT获取文本的嵌入向量
        """
        # 对文本进行分词和编码
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为整个序列的表示
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

    def get_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度
        """
        emb1 = self.get_bert_embeddings(text1)
        emb2 = self.get_bert_embeddings(text2)
        return cosine_similarity(emb1, emb2)[0][0]

def extract_features(article: Dict[str, Any], bert_extractor: BertFeatureExtractor) -> Dict[str, Any]:
    """
    提取新闻特征
    """
    features = {}
    
    # 1. 基础信息
    title = article.get("TITLE", "")
    subtitle = article.get("SUBTITLE", "")
    body = article.get("BODY", "")
    url = article.get("URL", "")
    published_ts = article.get("PUBLISHED_ON", 0)
    created_ts = article.get("CREATED_ON", 0)
    updated_ts = article.get("UPDATED_ON", 0)
    if not updated_ts:
        updated_ts = published_ts
    if not created_ts:
        created_ts = published_ts
    sentiment = article.get("SENTIMENT", "UNKNOWN")
    status = article.get("STATUS", "UNKNOWN")
    keywords = article.get("KEYWORDS", "")
    category_list = [cat['NAME'] for cat in article.get("CATEGORY_DATA", [])]
    source_data = article.get("SOURCE_DATA", {})
    # source_launch_ts = int(source_data.get("LAUNCH_DATE", 0))  # 来源启动时间
    # source_created_ts = int(source_data.get("CREATED_ON", 0))  # 来源创建时间
    # source_updated_ts = int(source_data.get("UPDATED_ON", 0))  # 来源更新时间
    # source_last_updated_ts = int(source_data.get("LAST_UPDATED_TS", 0))  # 来源最后更新时间

    # 2. 情感和状态特征
    # 将情感转换为数值
    sentiment_map = {
        "POSITIVE": 1,
        "NEUTRAL": 0,
        "NEGATIVE": -1,
        "UNKNOWN": 0
    }
    
    # 将状态转换为数值
    status_map = {
        "ACTIVE": 1,
        "INACTIVE": 0,
        "DELETED": -1,
        "UNKNOWN": 0
    }
    
    features.update({
        'sentiment': sentiment,
        'sentiment_value': sentiment_map.get(sentiment, 0),
        'status': status,
        'status_value': status_map.get(status, 0)
    })
    
    # 2. BERT特征
    # 2.1 获取文本嵌入
    title_embedding = bert_extractor.get_bert_embeddings(title)
    subtitle_embedding = bert_extractor.get_bert_embeddings(subtitle)
    body_embedding = bert_extractor.get_bert_embeddings(body)
    
    # 2.2 计算相似度
    title_subtitle_similarity = bert_extractor.get_similarity(title, subtitle)
    title_body_similarity = bert_extractor.get_similarity(title, body)
    subtitle_body_similarity = bert_extractor.get_similarity(subtitle, body)
    
    features.update({
        'title_embedding': title_embedding.tolist(),
        'subtitle_embedding': subtitle_embedding.tolist(),
        'body_embedding': body_embedding.tolist(),
        'title_subtitle_similarity': title_subtitle_similarity,
        'title_body_similarity': title_body_similarity,
        'subtitle_body_similarity': subtitle_body_similarity
    })
    
    # 3. 基础文本特征
    features.update({
        'title_length': len(title),
        'title_words': len(title.split()),
        'subtitle_length': len(subtitle),
        'subtitle_words': len(subtitle.split()),
        'body_length': len(body),
        'body_words': len(body.split()),
        'author': article.get("AUTHORS", ""),
        'language': article.get("LANG", "")
    })
    
    # 4. 时间特征
    features.update({
        'published_ts': datetime.utcfromtimestamp(published_ts).strftime('%Y-%m-%d %H:%M:%S'),
        'created_ts': datetime.utcfromtimestamp(created_ts).strftime('%Y-%m-%d %H:%M:%S'),
        'updated_ts': datetime.utcfromtimestamp(updated_ts).strftime('%Y-%m-%d %H:%M:%S'),
        'is_recent': (datetime.now() - datetime.utcfromtimestamp(updated_ts)).days <= 1,
    })
    
    # 5. 来源特征
    features.update({
        'source_name': source_data.get("NAME", ""),
        'source_type': source_data.get("SOURCE_TYPE", ""),
        'source_score': source_data.get("BENCHMARK_SCORE", 0),
        'source_language': source_data.get("LANG", ""),
        'source_launch_date': source_data.get("LAUNCH_DATE", 0)
    })
    
    # 6. 分类特征
    features.update({
        'categories': category_list,
        'category_count': len(category_list)
    })
    
    # 7. 互动特征
    features.update({
        'upvotes': article.get("UPVOTES", 0),
        'downvotes': article.get("DOWNVOTES", 0),
        'score': article.get("SCORE", 0)
    })
    
    # 8. 内容特征
    features.update({
        'has_image': bool(article.get("IMAGE_URL")),
        'has_url': bool(url),
        'has_keywords': bool(keywords),
        'keywords_list': keywords.split('|') if keywords else []
    })
    
    return features

def clean_article(article):
    # 1. 必要字段检查
    if not article.get("TITLE") or not article.get("PUBLISHED_ON"):
        return None  # 丢弃无效数据

    # 2. 标题、正文清洗
    def clean_text(text):
        if not text:
            return ""
        text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
        text = re.sub(r'[\r\n\t]', ' ', text)  # 去除换行、制表
        text = re.sub(r'\s+', ' ', text)  # 多空格合并
        return text.strip()

    article["TITLE"] = clean_text(article.get("TITLE", ""))
    article["SUBTITLE"] = clean_text(article.get("SUBTITLE", ""))
    article["BODY"] = clean_text(article.get("BODY", ""))

    # 3. 发布时间异常处理
    try:
        ts = int(article["PUBLISHED_ON"])
        if ts < 946684800 or ts > int(time.time()):  # 2000年之后且不晚于当前
            return None
    except:
        return None

    # 4. 其他字段标准化
    article["SENTIMENT"] = article.get("SENTIMENT", "UNKNOWN").upper()

    return article

def convert_to_serializable(obj):
    """
    将对象转换为可JSON序列化的格式
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def process_news_file(input_file: str, output_dir: str = "features", bert_extractor: BertFeatureExtractor = None) -> None:
    """
    处理新闻文件并提取特征
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # 数据清洗和去重
    cleaned_articles = []
    seen_ids = set()
    for article in articles:
        cleaned = clean_article(article)
        if cleaned and cleaned.get("ID") not in seen_ids:
            cleaned_articles.append(cleaned)
            seen_ids.add(cleaned.get("ID"))

    # 提取特征
    all_features = []
    for article in cleaned_articles:
        features = extract_features(article, bert_extractor)
        # 转换特征为可序列化格式
        serializable_features = convert_to_serializable(features)
        all_features.append(serializable_features)
    
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成输出文件名
    base_name = os.path.basename(input_file).replace('.json', '_features.json')
    output_file = os.path.join(output_dir, base_name)
    
    # 保存特征数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n=== 特征统计信息 ===")
    print(f"原始文章数: {len(articles)}")
    print(f"清洗后文章数: {len(cleaned_articles)}")
    print(f"总特征数: {len(all_features)}")
    
    # 打印详细统计
    if all_features:
        print("\n=== 详细统计 ===")
        print(f"平均标题长度: {np.mean([f['title_length'] for f in all_features]):.2f}")
        print(f"平均正文长度: {np.mean([f['body_length'] for f in all_features]):.2f}")
        print(f"平均标题-副标题相似度: {np.mean([f['title_subtitle_similarity'] for f in all_features]):.2f}")
        print(f"平均标题-正文相似度: {np.mean([f['title_body_similarity'] for f in all_features]):.2f}")
    
    print(f"\n特征数据已保存到: {output_file}")

def main():
    # 设置模型目录
    model_dir = "Project1/crypto_new_analyzer/models"
    
    # 初始化BERT特征提取器
    print("正在初始化BERT模型...")
    bert_extractor = BertFeatureExtractor(model_dir=model_dir)
    
    news_dir = "Project1/crypto_new_analyzer/crypto_news"
    output_dir = "Project1/crypto_new_analyzer/features"
    if not os.path.exists(news_dir):
        print("未找到新闻数据目录")
        return

    files = [f for f in os.listdir(news_dir) if f.endswith('.json') and not f.endswith('_features.json')]
    if not files:
        print("未找到新闻数据文件")
        return

    for file in files:
        input_file = os.path.join(news_dir, file)
        print(f"\n正在处理文件: {input_file}")
        process_news_file(input_file, output_dir=output_dir, bert_extractor=bert_extractor)

if __name__ == "__main__":
    main() 