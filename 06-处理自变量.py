import pandas as pd
import string
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet


def count_sentiwordnet_sentiment_words(text):
    """
    使用 SentiWordNet 来统计情感词数量。
    我们简单处理：将句子拆分为词，并尝试获取每个词的 WordNet 同义词集，
    再查询其 SentiWordNet 的得分（pos_score, neg_score），若 scores 高于阈值，则计数。
    """
    count = 0
    for token in text.lower().split():
        word = token.strip(string.punctuation)
        if not word:
            continue
        # 获取该词的第一个 WordNet 同义词集
        synsets = wordnet.synsets(word)
        if not synsets:
            continue
        synset = synsets[0]
        try:
            swn_syn = swn.senti_synset(synset.name())
        except:
            continue
        # 可以设定阈值，例如正负情感分数之和超过 0.5 就认为是情感词
        if (swn_syn.pos_score() + swn_syn.neg_score()) > 0.5:
            count += 1
    return count


def count_vader_sentiment_words(text, sia):
    """
    使用 VADER 的字典。
    VADER 自带词典里有情感分值，我们统计出现在词典中的词数量。
    """
    count = 0
    for token in text.lower().split():
        word = token.strip(string.punctuation)
        if word in sia.lexicon:
            count += 1
    return count


def process_excel(input_path, output_path, method="vader"):
    # 读取 Excel
    df = pd.read_excel(input_path)

    # 基础列统计
    df["len_review_text"] = df["review_text"].astype(str).apply(lambda x: len(x.split()))
    punctuations = set(string.punctuation)
    df["num_punctuation"] = df["review_text"].astype(str).apply(lambda x: sum(1 for c in x if c in punctuations))

    # 初始化情感分析工具
    sia = SentimentIntensityAnalyzer()

    # 选择方法： "vader" 或 "sentiwordnet"
    if method == "vader":
        df["num_senti_words"] = df["review_text"].astype(str).apply(lambda x: count_vader_sentiment_words(x, sia))
    elif method == "sentiwordnet":
        df["num_senti_words"] = df["review_text"].astype(str).apply(count_sentiwordnet_sentiment_words)
    elif method == "both":
        df["num_senti_vader"] = df["review_text"].astype(str).apply(lambda x: count_vader_sentiment_words(x, sia))
        df["num_senti_swn"] = df["review_text"].astype(str).apply(count_sentiwordnet_sentiment_words)
    else:
        raise ValueError("method must be one of 'vader', 'sentiwordnet', 'both'")

    # 对指定评分列进行裁剪
    cols_to_clip = ["Price", "Flavor", "Drinks", "Service Attitude",
                    "Atmosphere", "Waitstaff", "Environment", "Seating"]
    for col in cols_to_clip:
        if col in df.columns:
            df[col] = df[col].clip(lower=-2, upper=2)

    # 保存输出
    df.to_excel(output_path, index=False)
    print(f"已保存处理后的文件：{output_path}")


if __name__ == "__main__":
    import nltk

    nltk.download("vader_lexicon")
    nltk.download("sentiwordnet")
    nltk.download("wordnet")

    input_path = "test.xlsx"
    output_path = "test_processed.xlsx"

    # 可选方法： "vader", "sentiwordnet", 或 "both"
    process_excel(input_path, output_path, method="both")
