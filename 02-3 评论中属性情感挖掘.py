import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
import os

# 加载spacy英语模型
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("请先安装spacy英语模型: python -m spacy download en_core_web_sm")
    nlp = None


def load_cluster_results():
    """加载聚类结果"""
    try:
        with open('final_cluster_results.json', 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
        return cluster_data
    except FileNotFoundError:
        print("未找到聚类结果文件，请先运行聚类分析")
        return None


def create_attribute_dictionary(cluster_data):
    """根据聚类结果创建属性词典"""
    attribute_dict = {}
    cluster_mapping = {}

    for cluster_id, cluster_info in cluster_data['clusters'].items():
        cluster_label = cluster_info['label']
        words = cluster_info['words']

        # 为每个词创建映射
        for word in words:
            attribute_dict[word] = {
                'cluster_id': int(cluster_id),
                'cluster_label': cluster_label,
                'attribute_type': cluster_label.replace('_group', '').replace('_attributes', '')
            }

        cluster_mapping[int(cluster_id)] = {
            'label': cluster_label,
            'attribute_type': cluster_label.replace('_group', '').replace('_attributes', ''),
            'count': len(words)
        }

    return attribute_dict, cluster_mapping


def load_professional_sentiment_lexicons():
    """加载多个专业情感词典"""
    sentiment_lexicon = {
        'positive': set(),
        'negative': set(),
        'intensifiers': set(),
        'negations': set(),
        'sentiment_scores': {}  # 词到情感得分的映射
    }

    # 1. AFINN词典 (专业情感词典)
    try:
        afinn_url = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt"
        response = requests.get(afinn_url)
        if response.status_code == 200:
            for line in response.text.split('\n'):
                if line.strip():
                    word, score = line.strip().split('\t')
                    score = float(score)
                    sentiment_lexicon['sentiment_scores'][word] = score
                    if score > 0:
                        sentiment_lexicon['positive'].add(word)
                    elif score < 0:
                        sentiment_lexicon['negative'].add(word)
            print(f"AFINN词典加载成功: {len(sentiment_lexicon['sentiment_scores'])} 个词")
    except Exception as e:
        print(f"AFINN词典加载失败: {e}")

    # 2. Bing Liu's Opinion Lexicon
    try:
        # 正面词
        positive_url = "https://raw.githubusercontent.com/shekhargulati/opinion-lexicon-English/master/positive-words.txt"
        response = requests.get(positive_url)
        if response.status_code == 200:
            words = set(response.text.split('\n'))
            sentiment_lexicon['positive'].update(words)
            for word in words:
                sentiment_lexicon['sentiment_scores'][word] = 1.0
            print(f"Bing Liu正面词典加载成功: {len(words)} 个词")
    except Exception as e:
        print(f"Bing Liu正面词典加载失败: {e}")

    try:
        # 负面词
        negative_url = "https://raw.githubusercontent.com/shekhargulati/opinion-lexicon-English/master/negative-words.txt"
        response = requests.get(negative_url)
        if response.status_code == 200:
            words = set(response.text.split('\n'))
            sentiment_lexicon['negative'].update(words)
            for word in words:
                sentiment_lexicon['sentiment_scores'][word] = -1.0
            print(f"Bing Liu负面词典加载成功: {len(words)} 个词")
    except Exception as e:
        print(f"Bing Liu负面词典加载失败: {e}")

    # 3. 添加基础情感词（作为备用）
    base_positive = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'awesome', 'perfect', 'outstanding', 'superb', 'brilliant', 'fabulous',
        'terrific', 'magnificent', 'delicious', 'tasty', 'yummy', 'fresh',
        'quality', 'nice', 'pleasant', 'satisfying', 'enjoyable', 'impressive',
        'remarkable', 'exceptional', 'stellar', 'phenomenal', 'love', 'lovely',
        'beautiful', 'charming', 'delightful', 'splendid', 'marvelous'
    }

    base_negative = {
        'bad', 'poor', 'terrible', 'awful', 'horrible', 'disappointing',
        'mediocre', 'average', 'ordinary', 'bland', 'tasteless', 'overcooked',
        'undercooked', 'cold', 'soggy', 'greasy', 'salty', 'spicy', 'sour',
        'bitter', 'expensive', 'overpriced', 'costly', 'pricey', 'slow',
        'rude', 'unfriendly', 'inattentive', 'noisy', 'crowded', 'dirty',
        'hate', 'dislike', 'disgusting', 'unpleasant', 'awful', 'horrible'
    }

    sentiment_lexicon['positive'].update(base_positive)
    sentiment_lexicon['negative'].update(base_negative)

    # 为基础词添加得分
    for word in base_positive:
        if word not in sentiment_lexicon['sentiment_scores']:
            sentiment_lexicon['sentiment_scores'][word] = 1.0

    for word in base_negative:
        if word not in sentiment_lexicon['sentiment_scores']:
            sentiment_lexicon['sentiment_scores'][word] = -1.0

    # 4. 强度修饰词
    sentiment_lexicon['intensifiers'] = {
        'very', 'really', 'extremely', 'quite', 'absolutely', 'totally',
        'completely', 'utterly', 'highly', 'particularly', 'especially',
        'incredibly', 'exceptionally', 'remarkably', 'surprisingly'
    }

    # 5. 否定词
    sentiment_lexicon['negations'] = {
        'not', 'no', 'never', 'none', 'neither', 'nor', 'without',
        'lack', 'lacking', 'missing', 'hardly', 'scarcely', 'barely'
    }

    print(f"情感词典汇总: {len(sentiment_lexicon['positive'])} 正面词, "
          f"{len(sentiment_lexicon['negative'])} 负面词, "
          f"{len(sentiment_lexicon['sentiment_scores'])} 有得分词")

    return sentiment_lexicon


def extract_attribute_sentiment_pairs(doc, attribute_dict, sentiment_lexicon):
    """使用依存句法分析提取(属性词, 情感词)对"""
    pairs = []

    # 找到所有属性词
    attribute_tokens = []
    for token in doc:
        if token.text.lower() in attribute_dict:
            attribute_tokens.append(token)

    # 为每个属性词寻找相关的情感词
    for attr_token in attribute_tokens:
        sentiment_info = find_related_sentiment(attr_token, sentiment_lexicon)
        if sentiment_info:
            pairs.append({
                'attribute': attr_token.text.lower(),
                'attribute_info': attribute_dict[attr_token.text.lower()],
                'sentiment_word': sentiment_info['word'],
                'sentiment_score': sentiment_info['score'],
                'sentiment_strength': sentiment_info['strength'],
                'relation_type': sentiment_info['relation'],
                'sentence': doc.text
            })

    return pairs


def find_related_sentiment(attr_token, sentiment_lexicon):
    """在依存关系中寻找与属性词相关的情感词"""
    # 检查属性词本身是否是情感词
    attr_word = attr_token.text.lower()
    if attr_word in sentiment_lexicon['sentiment_scores']:
        score = sentiment_lexicon['sentiment_scores'][attr_word]
        return {
            'word': attr_word,
            'score': score,
            'strength': abs(score),
            'relation': 'self_sentiment'
        }

    # 检查依存关系中的情感词
    sentiment_candidates = []

    # 1. 检查子节点
    for child in attr_token.children:
        child_word = child.text.lower()
        if child_word in sentiment_lexicon['sentiment_scores']:
            sentiment_candidates.append((child, child.dep_, 'child'))

    # 2. 检查父节点
    head_word = attr_token.head.text.lower()
    if head_word in sentiment_lexicon['sentiment_scores']:
        sentiment_candidates.append((attr_token.head, attr_token.dep_, 'head'))

    # 3. 检查相邻词（窗口大小为3）
    window_size = 3
    start = max(0, attr_token.i - window_size)
    end = min(len(attr_token.doc), attr_token.i + window_size + 1)

    for i in range(start, end):
        if i != attr_token.i:  # 排除自己
            token = attr_token.doc[i]
            token_word = token.text.lower()
            if token_word in sentiment_lexicon['sentiment_scores']:
                distance = i - attr_token.i
                sentiment_candidates.append((token, f'window_{distance}', 'window'))

    # 选择最相关的情感词
    if sentiment_candidates:
        # 优先选择依存关系更紧密的词
        best_candidate = None
        for token, dep, rel_type in sentiment_candidates:
            # 根据关系类型赋予权重
            if dep in ['amod', 'acomp', 'nsubj', 'dobj']:  # 紧密的语法关系
                best_candidate = (token, rel_type, dep)
                break
            elif 'window' in dep and best_candidate is None:
                best_candidate = (token, rel_type, dep)

        if best_candidate:
            token, rel_type, dep = best_candidate
            score = calculate_sentiment_score(token, sentiment_lexicon)
            return {
                'word': token.text.lower(),
                'score': score,
                'strength': abs(score),
                'relation': f'{rel_type}_{dep}'
            }

    return None


def calculate_sentiment_score(token, sentiment_lexicon):
    """计算情感词的得分，考虑强度修饰词和否定词"""
    base_word = token.text.lower()
    if base_word not in sentiment_lexicon['sentiment_scores']:
        return 0.0

    base_score = sentiment_lexicon['sentiment_scores'][base_word]

    # 检查强度修饰词
    intensity = 1.0
    for child in token.children:
        if child.text.lower() in sentiment_lexicon['intensifiers']:
            intensity = 1.5  # 增强情感强度
            break

    # 检查否定词
    negation = 1.0
    # 检查子节点中的否定词
    for child in token.children:
        if child.text.lower() in sentiment_lexicon['negations']:
            negation = -1.0
            break

    # 检查父节点中的否定词
    if negation == 1.0:
        for child in token.head.children:
            if child.text.lower() in sentiment_lexicon['negations']:
                negation = -1.0
                break

    return base_score * intensity * negation


def analyze_review_sentiment(review_text, attribute_dict, sentiment_lexicon):
    """分析单条评论的情感"""
    doc = nlp(review_text)
    pairs = extract_attribute_sentiment_pairs(doc, attribute_dict, sentiment_lexicon)

    # 按属性聚类汇总情感
    cluster_sentiments = defaultdict(list)
    attribute_details = []

    for pair in pairs:
        cluster_id = pair['attribute_info']['cluster_id']
        cluster_sentiments[cluster_id].append(pair['sentiment_score'])
        attribute_details.append(pair)

    # 计算每个聚类的平均情感
    result = {}
    for cluster_id, scores in cluster_sentiments.items():
        if scores:
            result[cluster_id] = {
                'average_sentiment': np.mean(scores),
                'count': len(scores),
                'scores': scores
            }

    return result, attribute_details


def process_corpus(corpus_file, attribute_dict, sentiment_lexicon):
    """处理整个语料库"""
    try:
        df = pd.read_excel(corpus_file)
        print(f"成功读取语料库，共 {len(df)} 条评论")
        print(f"列名: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"未找到文件: {corpus_file}")
        return None, None

    results = []
    detailed_pairs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理评论"):
        review_text = str(row.get('review_text', ''))
        reviewer_href = row.get('reviewer_href', '')
        star_rate = row.get('star_rate', None)

        if not review_text or review_text.lower() in ['nan', 'null', '']:
            continue

        try:
            sentiment_result, attribute_details = analyze_review_sentiment(
                review_text, attribute_dict, sentiment_lexicon
            )

            # 添加基本结果
            result_row = {
                'review_id': idx,
                'reviewer_href': reviewer_href,
                'star_rate': star_rate,
                'review_text': review_text[:200] + '...' if len(review_text) > 200 else review_text,
                'total_attributes_found': len(attribute_details)
            }

            # 添加每个聚类的情绪信息
            for cluster_id in range(8):  # 假设有8个聚类
                if cluster_id in sentiment_result:
                    result_row[f'cluster_{cluster_id}_score'] = sentiment_result[cluster_id]['average_sentiment']
                    result_row[f'cluster_{cluster_id}_count'] = sentiment_result[cluster_id]['count']
                else:
                    result_row[f'cluster_{cluster_id}_score'] = 0.0
                    result_row[f'cluster_{cluster_id}_count'] = 0

            results.append(result_row)

            # 保存详细的对信息
            for detail in attribute_details:
                detail['review_id'] = idx
                detail['reviewer_href'] = reviewer_href
                detail['star_rate'] = star_rate
                detailed_pairs.append(detail)

        except Exception as e:
            print(f"处理第 {idx} 条评论时出错: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(detailed_pairs)


def visualize_results(results_df, cluster_mapping):
    """可视化分析结果"""
    # 计算每个聚类的总体情感
    cluster_sentiments = []
    for cluster_id in range(8):
        scores = results_df[f'cluster_{cluster_id}_score'].replace(0, np.nan).dropna()
        counts = results_df[f'cluster_{cluster_id}_count']
        total_mentions = counts.sum()

        if len(scores) > 0:
            cluster_info = cluster_mapping.get(cluster_id, {'attribute_type': f'cluster_{cluster_id}'})
            cluster_sentiments.append({
                'cluster_id': cluster_id,
                'attribute_type': cluster_info['attribute_type'],
                'average_sentiment': scores.mean(),
                'total_mentions': total_mentions,
                'review_coverage': len(scores) / len(results_df)
            })

    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 情感得分条形图
    sentiment_df = pd.DataFrame(cluster_sentiments)
    colors = ['green' if x > 0 else 'red' for x in sentiment_df['average_sentiment']]
    ax1.bar(range(len(sentiment_df)), sentiment_df['average_sentiment'], color=colors)
    ax1.set_xticks(range(len(sentiment_df)))
    ax1.set_xticklabels([f"{row['attribute_type']}\n({row['cluster_id']})" for _, row in sentiment_df.iterrows()],
                        rotation=45)
    ax1.set_title('各属性情感平均得分')
    ax1.set_ylabel('情感得分')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 提及次数条形图
    ax2.bar(range(len(sentiment_df)), sentiment_df['total_mentions'])
    ax2.set_xticks(range(len(sentiment_df)))
    ax2.set_xticklabels([f"{row['attribute_type']}\n({row['cluster_id']})" for _, row in sentiment_df.iterrows()],
                        rotation=45)
    ax2.set_title('各属性提及次数')
    ax2.set_ylabel('提及次数')

    # 情感分布箱线图
    sentiment_data = []
    labels = []
    for cluster_id in range(8):
        scores = results_df[f'cluster_{cluster_id}_score'].replace(0, np.nan).dropna()
        if len(scores) > 0:
            sentiment_data.append(scores)
            labels.append(f"Cluster {cluster_id}")

    ax3.boxplot(sentiment_data)
    ax3.set_xticklabels(labels, rotation=45)
    ax3.set_title('各属性情感分布')
    ax3.set_ylabel('情感得分')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 覆盖率饼图
    coverage = [row['review_coverage'] for row in cluster_sentiments]
    labels = [f"{row['attribute_type']}\n({row['cluster_id']})" for row in cluster_sentiments]
    ax4.pie(coverage, labels=labels, autopct='%1.1f%%')
    ax4.set_title('各属性评论覆盖率')

    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("开始属性情感分析...")

    # 1. 加载聚类结果
    cluster_data = load_cluster_results()
    if not cluster_data:
        return

    # 2. 创建属性词典
    attribute_dict, cluster_mapping = create_attribute_dictionary(cluster_data)
    print(f"创建属性词典完成，共 {len(attribute_dict)} 个属性词")

    # 3. 加载专业情感词典
    sentiment_lexicon = load_professional_sentiment_lexicons()

    # 4. 处理语料库
    corpus_file = "attr_senti_corpus.xlsx"
    results_df, detailed_pairs_df = process_corpus(corpus_file, attribute_dict, sentiment_lexicon)

    if results_df is not None:
        # 5. 保存结果
        results_df.to_excel('attribute_sentiment_results.xlsx', index=False)
        detailed_pairs_df.to_excel('detailed_attribute_sentiment_pairs.xlsx', index=False)

        print(f"\n分析完成！")
        print(f"处理了 {len(results_df)} 条评论")
        print(f"找到了 {len(detailed_pairs_df)} 个(属性词, 情感词)对")

        # 6. 可视化结果
        visualize_results(results_df, cluster_mapping)

        # 7. 显示摘要统计
        total_attributes = detailed_pairs_df['attribute'].nunique()
        print(f"\n摘要统计:")
        print(f"总评论数: {len(results_df)}")
        print(f"总属性提及次数: {len(detailed_pairs_df)}")
        print(f"唯一属性词数: {total_attributes}")

        # 各聚类统计
        cluster_stats = detailed_pairs_df['attribute_info'].apply(lambda x: x['cluster_id']).value_counts()
        print("\n各聚类提及次数:")
        for cluster_id, count in cluster_stats.items():
            cluster_info = cluster_mapping.get(cluster_id, {'attribute_type': f'cluster_{cluster_id}'})
            print(f"  聚类 {cluster_id} ({cluster_info['attribute_type']}): {count} 次")

    print("\n结果文件已保存:")
    print("- attribute_sentiment_results.xlsx (情感分析结果)")
    print("- detailed_attribute_sentiment_pairs.xlsx (详细配对信息)")
    print("- sentiment_analysis_results.png (可视化图表)")


if __name__ == "__main__":
    main()