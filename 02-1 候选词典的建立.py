import pandas as pd
import numpy as np
from googletrans import Translator
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 下载NLTK数据
nltk.download('punkt')
nltk.download('stopwords')

# 初始化翻译器
translator = Translator()


def translate_text(text):
    """翻译非英文文本到英文"""
    if pd.isna(text):
        return ""

    text = str(text)
    # 简单的英文检测
    if re.match(r'^[a-zA-Z\s\.,!?;:]+$', text):
        return text

    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except:
        return text


def preprocess_text(text):
    """文本预处理"""
    text = str(text).lower()
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 移除停用词
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens


def build_seed_dictionary():
    """构建种子属性词典"""
    # 基于专业餐饮评测的高频属性词
    professional_attributes = [
        # 食物相关
        'flavor', 'taste', 'delicious', 'tasty', 'yummy', 'savory', 'flavorful',
        'fresh', 'freshness', 'quality', 'cooked', 'cooking', 'preparation',
        'seasoning', 'spices', 'ingredients', 'presentation', 'plating',
        'texture', 'tenderness', 'juiciness', 'crispy', 'crunchy', 'creamy',

        # 服务相关
        'service', 'serving', 'friendly', 'attentive', 'prompt', 'efficient',
        'professional', 'courteous', 'helpful', 'knowledgeable', 'staff',
        'waiter', 'waitress', 'server', 'manager', 'host', 'hostess',

        # 环境氛围
        'ambiance', 'atmosphere', 'decor', 'decoration', 'interior', 'design',
        'clean', 'cleanliness', 'hygiene', 'comfortable', 'comfort',
        'lighting', 'music', 'noise', 'volume', 'spacious', 'space',
        'cozy', 'romantic', 'intimate', 'vibe', 'mood',

        # 价格价值
        'price', 'pricing', 'expensive', 'affordable', 'reasonable', 'value',
        'worth', 'portion', 'size', 'quantity', 'amount', 'budget',
        'cost', 'money', 'expensive', 'cheap', 'overpriced',

        # 具体菜品
        'menu', 'selection', 'variety', 'choice', 'dish', 'appetizer',
        'starter', 'entree', 'main', 'dessert', 'sweet', 'drink',
        'beverage', 'cocktail', 'wine', 'beer', 'coffee', 'tea',

        # 体验相关
        'experience', 'visit', 'dining', 'meal', 'dinner', 'lunch',
        'breakfast', 'brunch', 'recommend', 'recommendation', 'return',
        'again', 'favorite', 'popular', 'famous', 'known',

        # 设施位置
        'location', 'place', 'restaurant', 'parking', 'valet', 'reservation',
        'booking', 'wait', 'waiting', 'time', 'seat', 'table',
        'view', 'outdoor', 'patio', 'terrace'
    ]

    return set(professional_attributes)


def train_word2vec(corpus_tokens, save_path='word2vec_model'):
    """训练Word2Vec模型并保存"""
    print("训练Word2Vec模型...")
    model = Word2Vec(
        sentences=corpus_tokens,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=10
    )

    # 保存模型
    model.save(f"{save_path}.model")
    # 保存为KeyedVectors格式（更轻量，便于加载）
    model.wv.save(f"{save_path}.kv")
    # 保存为文本格式（可读）
    model.wv.save_word2vec_format(f"{save_path}.txt", binary=False)

    print(f"Word2Vec模型已保存为:")
    print(f"- {save_path}.model (完整模型)")
    print(f"- {save_path}.kv (词向量)")
    print(f"- {save_path}.txt (文本格式)")

    return model


def load_word2vec_model(model_path='word2vec_model.model'):
    """加载Word2Vec模型"""
    if os.path.exists(model_path):
        print(f"加载已存在的Word2Vec模型: {model_path}")
        return Word2Vec.load(model_path)
    else:
        return None


def load_word_vectors(kv_path='word2vec_model.kv'):
    """加载词向量"""
    if os.path.exists(kv_path):
        print(f"加载词向量: {kv_path}")
        return gensim.models.KeyedVectors.load(kv_path)
    else:
        return None


def find_similar_words(model, seed_words, threshold=0.6):
    """找到与种子词相似的候选词"""
    similar_words = {}

    for word in seed_words:
        if word in model.wv:
            try:
                similar = model.wv.most_similar(word, topn=20)
                for similar_word, similarity in similar:
                    if similarity > threshold:
                        if similar_word not in similar_words:
                            similar_words[similar_word] = []
                        similar_words[similar_word].append((word, similarity))
            except:
                continue

    return similar_words


def get_word_vector(model, word):
    """获取单词的词向量"""
    if word in model.wv:
        return model.wv[word]
    else:
        return None


def export_word_vectors(model, output_file='word_vectors.csv'):
    """导出所有词向量到CSV文件"""
    words = list(model.wv.index_to_key)
    vectors = [model.wv[word] for word in words]

    # 创建DataFrame
    vector_df = pd.DataFrame(vectors, index=words)
    vector_df.to_csv(output_file)
    print(f"词向量已导出到 {output_file}")

    return vector_df


def main():
    # 检查是否已有训练好的模型
    existing_model = load_word2vec_model()
    existing_kv = load_word_vectors()

    if existing_model is not None:
        print("使用已存在的模型进行相似词查找...")
        model = existing_model
        # 读取处理过的数据（如果存在）
        try:
            df = pd.read_csv('processed_corpus.csv')
            corpus_tokens = eval(df['tokens'].iloc[0])  # 假设tokens以字符串形式存储
        except:
            df = None
            corpus_tokens = []
    else:
        # 1. 读取语料库
        print("读取语料库...")
        try:
            df = pd.read_excel('Restaurant corpus.xlsx')
        except:
            # 如果文件不存在，使用示例数据
            print("未找到Restaurant corpus.xlsx，使用示例数据")
            df = pd.DataFrame({
                'review_text': [
                    "The food was absolutely delicious and the service was excellent!",
                    "非常美味的食物，服务也很好！",
                    "Ambiance was perfect for a romantic dinner.",
                    "Le service était impeccable et la nourriture délicieuse.",
                    "Prices are reasonable for the quality of food."
                ]
            })

        # 2. 翻译非英文文本
        print("翻译非英文文本...")
        df['translated_text'] = df['review_text'].apply(translate_text)

        # 3. 文本预处理和分词
        print("文本预处理...")
        df['tokens'] = df['translated_text'].apply(preprocess_text)

        # 保存处理后的数据
        df.to_csv('processed_corpus.csv', index=False)
        print("处理后的语料库已保存到 processed_corpus.csv")

        # 获取所有token
        corpus_tokens = df['tokens'].tolist()

        # 4. 训练Word2Vec模型
        model = train_word2vec(corpus_tokens)

    # 获取所有token的频率统计
    all_tokens = [token for tokens_list in corpus_tokens for token in tokens_list]
    word_freq = Counter(all_tokens)
    print(f"\n语料库词汇量: {len(word_freq)}")
    print("Top 30 高频词:")
    for word, freq in word_freq.most_common(30):
        print(f"{word}: {freq}")

    # 5. 构建种子词典
    seed_dict = build_seed_dictionary()
    print(f"\n种子词典 ({len(seed_dict)} 个词):")
    print(sorted(seed_dict))

    # 6. 寻找相似词
    print("\n寻找相似词...")
    threshold = 0.65
    similar_words = find_similar_words(model, seed_dict, threshold)

    # 7. 显示候选词
    print(f"\n找到 {len(similar_words)} 个候选词 (阈值: {threshold}):")
    candidate_words = []
    for word, similarities in sorted(similar_words.items(), key=lambda x: max([s[1] for s in x[1]]), reverse=True):
        max_similarity = max([s[1] for s in similarities])
        seed_matches = [s[0] for s in similarities]
        print(f"{word}: {max_similarity:.3f} (匹配种子词: {seed_matches})")
        candidate_words.append((word, max_similarity, seed_matches))

    # 8. 保存结果
    results = {
        'seed_dictionary': list(seed_dict),
        'candidate_words': candidate_words,
        'threshold': threshold,
        'vocabulary_size': len(model.wv.index_to_key),
        'word_frequency': dict(word_freq.most_common(100))
    }

    # 保存到文件
    import json
    with open('attribute_mining_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 导出候选词表格
    candidate_df = pd.DataFrame(candidate_words, columns=['word', 'similarity', 'seed_matches'])
    candidate_df.to_excel('candidate_words.xlsx', index=False)

    # 导出词向量
    vector_df = export_word_vectors(model)

    print("\n所有文件已保存:")
    print("- word2vec_model.model (Word2Vec模型)")
    print("- word2vec_model.kv (词向量)")
    print("- word2vec_model.txt (文本格式词向量)")
    print("- attribute_mining_results.json (挖掘结果)")
    print("- candidate_words.xlsx (候选词表格)")
    print("- word_vectors.csv (所有词向量)")
    print("- processed_corpus.csv (处理后的语料库)")

    return results, model, vector_df


# 在其他程序中使用的辅助函数
def load_trained_model():
    """在其他程序中加载训练好的模型"""
    model = load_word2vec_model()
    if model is None:
        raise FileNotFoundError("未找到训练好的Word2Vec模型，请先运行主程序")
    return model


def get_similar_words(word, topn=10, threshold=0.6):
    """获取与指定词相似的词"""
    model = load_trained_model()
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=topn)
        return [(w, score) for w, score in similar_words if score > threshold]
    else:
        return []


def get_word_similarity(word1, word2):
    """计算两个词的相似度"""
    model = load_trained_model()
    if word1 in model.wv and word2 in model.wv:
        return model.wv.similarity(word1, word2)
    else:
        return None


if __name__ == "__main__":
    results, model, vector_df = main()

    # 演示如何使用
    print("\n演示如何使用:")
    example_word = "service"
    if example_word in model.wv:
        similar = get_similar_words(example_word)
        print(f"与 '{example_word}' 相似的词: {similar[:5]}")

        vector = get_word_vector(model, example_word)
        print(f"'{example_word}' 的词向量维度: {len(vector)}")