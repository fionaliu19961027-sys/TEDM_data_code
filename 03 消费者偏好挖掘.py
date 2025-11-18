import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# 下载必要的数据
nltk.download('vader_lexicon')
nltk.download('punkt')


class ConsumerPreferenceAnalyzer:
    def __init__(self):
        self.attribute_categories = {}
        self.sia = SentimentIntensityAnalyzer()
        self.consumer_data = defaultdict(lambda: defaultdict(list))

    def load_attribute_categories(self, cluster_file='final_cluster_results.json'):
        """加载属性分类"""
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)

        for cluster_id, cluster_info in cluster_data['clusters'].items():
            category_name = cluster_info['label']
            attributes = cluster_info['words']
            self.attribute_categories[category_name] = set(attributes)

        print(f"加载了 {len(self.attribute_categories)} 个属性类别")
        return self.attribute_categories

    def load_consumer_reviews(self, review_file='2025年8月全部数据.xlsx'):
        """加载消费者评论数据"""
        print("加载消费者评论数据...")
        df = pd.read_excel(review_file, sheet_name='全部历史评论')

        # 按用户分组评论
        user_reviews = df.groupby('user_id')['review_text'].apply(list).to_dict()
        user_ratings = df.groupby('user_id')['star_rate'].apply(list).to_dict()

        print(f"加载了 {len(user_reviews)} 个用户的评论数据")
        return user_reviews, user_ratings

    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text

    def extract_attribute_sentences(self, text, attribute_words):
        """提取包含特定属性词的句子"""
        sentences = sent_tokenize(str(text))
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            words = word_tokenize(sentence_lower)
            if any(word in attribute_words for word in words):
                relevant_sentences.append(sentence)

        return relevant_sentences

    def calculate_sentiment_score(self, text):
        """计算情感得分"""
        return self.sia.polarity_scores(text)['compound']  # -1到1之间的得分

    def analyze_consumer_preferences(self, user_reviews):
        """分析每个消费者的属性偏好"""
        print("分析消费者偏好...")

        # 初始化结果字典
        preference_matrix = {}
        detailed_stats = {}

        for user_id, reviews in tqdm(user_reviews.items(), desc="处理用户"):
            user_stats = {category: {'mentions': 0, 'sentiment_scores': []}
                          for category in self.attribute_categories.keys()}

            # 处理用户的每条评论
            for review in reviews:
                preprocessed_review = self.preprocess_text(review)

                for category, attribute_words in self.attribute_categories.items():
                    # 提取包含该属性类别词汇的句子
                    attribute_sentences = self.extract_attribute_sentences(
                        preprocessed_review, attribute_words
                    )

                    if attribute_sentences:
                        user_stats[category]['mentions'] += len(attribute_sentences)

                        # 计算每个句子的情感得分
                        for sentence in attribute_sentences:
                            sentiment = self.calculate_sentiment_score(sentence)
                            user_stats[category]['sentiment_scores'].append(sentiment)

            # 计算每个属性的权重
            weights = {}
            total_weight = 0

            for category, stats in user_stats.items():
                f_kj = stats['mentions']  # 词频

                if stats['sentiment_scores']:
                    s_kj = np.mean(np.abs(stats['sentiment_scores']))  # 平均情感强度绝对值
                else:
                    s_kj = 0

                # 计算未归一化的权重
                weight = f_kj * s_kj
                weights[category] = weight
                total_weight += weight

            # 归一化权重
            if total_weight > 0:
                normalized_weights = {cat: weight / total_weight for cat, weight in weights.items()}
            else:
                normalized_weights = {cat: 0 for cat in self.attribute_categories.keys()}

            preference_matrix[user_id] = normalized_weights
            detailed_stats[user_id] = user_stats

        return preference_matrix, detailed_stats

    def create_preference_matrix(self, preference_data):
        """创建消费者-属性偏好矩阵"""
        print("创建偏好矩阵...")

        # 获取所有用户和属性类别
        users = list(preference_data.keys())
        categories = list(self.attribute_categories.keys())

        # 创建DataFrame
        matrix_data = []
        for user_id, weights in preference_data.items():
            row = {'user_id': user_id}
            for category in categories:
                row[category] = weights.get(category, 0)
            matrix_data.append(row)

        preference_df = pd.DataFrame(matrix_data)
        return preference_df

    def save_results(self, preference_matrix, detailed_stats, output_file='consumer_preference_matrix.xlsx'):
        """保存结果"""
        print("保存结果...")

        # 保存偏好矩阵
        pref_df = self.create_preference_matrix(preference_matrix)
        pref_df.to_excel(output_file, index=False)

        # 保存详细统计
        detailed_results = []
        for user_id, stats in detailed_stats.items():
            for category, category_stats in stats.items():
                detailed_results.append({
                    'user_id': user_id,
                    'category': category,
                    'mentions': category_stats['mentions'],
                    'avg_sentiment': np.mean(category_stats['sentiment_scores']) if category_stats[
                        'sentiment_scores'] else 0,
                    'sentiment_intensity': np.mean(np.abs(category_stats['sentiment_scores'])) if category_stats[
                        'sentiment_scores'] else 0,
                    'num_sentences': len(category_stats['sentiment_scores'])
                })

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_excel('consumer_preference_detailed.xlsx', index=False)

        # 保存汇总统计
        summary = {
            'total_users': len(preference_matrix),
            'total_categories': len(self.attribute_categories),
            'preference_matrix_shape': pref_df.shape
        }

        with open('preference_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("结果保存完成!")

    def run_analysis(self):
        """运行完整分析流程"""
        print("开始消费者偏好分析...")

        # 1. 加载属性分类
        self.load_attribute_categories()

        # 2. 加载消费者数据
        user_reviews, user_ratings = self.load_consumer_reviews()

        # 3. 分析偏好
        preference_matrix, detailed_stats = self.analyze_consumer_preferences(user_reviews)

        # 4. 保存结果
        self.save_results(preference_matrix, detailed_stats)

        print("分析完成!")
        return preference_matrix, detailed_stats


# 主函数
def main():
    analyzer = ConsumerPreferenceAnalyzer()
    preference_matrix, detailed_stats = analyzer.run_analysis()

    # 显示摘要信息
    print(f"\n=== 分析结果摘要 ===")
    print(f"分析用户数量: {len(preference_matrix)}")
    print(f"属性类别数量: {len(analyzer.attribute_categories)}")

    # 显示前几个用户的偏好
    print("\n前3个用户的偏好权重:")
    for i, (user_id, weights) in enumerate(preference_matrix.items()):
        if i >= 3:
            break
        print(f"\n用户 {user_id}:")
        for category, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0:
                print(f"  {category}: {weight:.3f}")


if __name__ == "__main__":
    main()