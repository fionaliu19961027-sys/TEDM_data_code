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


class StrictnessAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.basic_attributes = {}  # 存储基本属性词汇
        self.consumer_strictness = {}  # 存储每个消费者的严格指数
        self.category_analysis = {}  # 存储类别分析结果

    def load_basic_attributes(self, basic_attributes_list):
        """加载基本属性词汇"""
        # 假设basic_attributes_list是基本属性类别名称列表
        self.basic_attribute_names = basic_attributes_list

        # 从之前的结果中加载属性词汇
        try:
            with open('final_cluster_results.json', 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)

            for category_name in basic_attributes_list:
                # 在聚类结果中查找对应的属性词汇
                for cluster_id, cluster_info in cluster_data['clusters'].items():
                    if cluster_info['label'] == category_name:
                        self.basic_attributes[category_name] = set(cluster_info['words'])
                        break

            print(f"加载了 {len(self.basic_attributes)} 个基本属性的词汇")

        except FileNotFoundError:
            print("警告: 未找到聚类结果文件，使用默认基本属性词汇")
            # 这里可以设置一些默认的基本属性词汇
            self.basic_attributes = {
                'atmospherethe_ambiancethe_group': {'atmosphere', 'ambiance', 'environment', 'decor', 'clean', 'noise'},
                'price_attributes': {'price', 'cost', 'expensive', 'affordable', 'value', 'worth', 'cheap',
                                     'overpriced'},
                'waitress_server_group': {'waitress', 'server', 'waiter', 'staff', 'service', 'attentive', 'prompt'},
                'courteous_personable_group': {'friendly', 'courteous', 'polite', 'kind', 'helpful', 'nice',
                                               'professional'}
            }

        # 合并所有基本属性词汇用于快速查找
        self.all_basic_words = set()
        for words in self.basic_attributes.values():
            self.all_basic_words.update(words)

        return self.basic_attributes

    def load_consumer_reviews(self, review_file='2025年8月全部数据.xlsx'):
        """加载消费者评论数据"""
        print("加载消费者评论数据...")
        try:
            df = pd.read_excel(review_file, sheet_name='全部历史评论')

            # 按用户分组评论和评分
            user_data = {}
            for user_id, group in df.groupby('user_id'):
                user_data[user_id] = {
                    'reviews': group['review_text'].tolist(),
                    'ratings': group['star_rate'].tolist()
                }

            print(f"加载了 {len(user_data)} 个用户的评论数据")
            return user_data

        except Exception as e:
            print(f"加载数据失败: {e}")
            # 返回示例数据用于测试
            return self.create_sample_data()

    def create_sample_data(self):
        """创建示例数据用于测试"""
        print("使用示例数据进行测试...")
        user_data = {
            'user1': {
                'reviews': [
                    "The service was terrible but the food was good.",
                    "Very expensive prices, not worth it.",
                    "Friendly staff and great atmosphere."
                ],
                'ratings': [2, 1, 5]
            },
            'user2': {
                'reviews': [
                    "The waiter was rude but the food was amazing.",
                    "Good value for money.",
                    "Clean environment and nice decor."
                ],
                'ratings': [4, 5, 5]
            }
        }
        return user_data

    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text

    def contains_basic_attribute(self, text):
        """检查文本是否包含基本属性词"""
        words = word_tokenize(text.lower())
        return any(word in self.all_basic_words for word in words)

    def is_negative_sentence(self, sentence):
        """判断句子是否为负面情感"""
        sentiment_score = self.sia.polarity_scores(sentence)['compound']
        return sentiment_score < -0.1  # 阈值可调整

    def analyze_review_strictness(self, review_text, rating):
        """分析单条评论的严格性"""
        sentences = sent_tokenize(str(review_text))
        has_negative_basic = False

        for sentence in sentences:
            # 检查是否包含基本属性词且为负面情感
            if (self.contains_basic_attribute(sentence) and
                    self.is_negative_sentence(sentence)):
                has_negative_basic = True
                break

        if has_negative_basic:
            if rating <= 3:  # 严格反应
                return 'strict'
            else:  # 非严格反应
                return 'non_strict'

        return 'no_negative_basic'  # 没有基础属性负面句子

    def calculate_strictness_index(self, user_data):
        """计算每个消费者的严格指数"""
        print("计算消费者严格指数...")

        strictness_results = {}

        for user_id, data in tqdm(user_data.items(), desc="处理用户"):
            strict_count = 0
            non_strict_count = 0
            review_details = []

            for review, rating in zip(data['reviews'], data['ratings']):
                result = self.analyze_review_strictness(review, rating)

                if result == 'strict':
                    strict_count += 1
                elif result == 'non_strict':
                    non_strict_count += 1

                review_details.append({
                    'review': review[:100] + '...' if len(review) > 100 else review,
                    'rating': rating,
                    'result': result
                })

            # 计算严格指数
            total_relevant = strict_count + non_strict_count
            if total_relevant > 0:
                strictness_index = strict_count / total_relevant
            else:
                strictness_index = 0  # 如果没有相关评论，设为0

            strictness_results[user_id] = {
                'strictness_index': strictness_index,
                'strict_count': strict_count,
                'non_strict_count': non_strict_count,
                'total_relevant_reviews': total_relevant,
                'total_reviews': len(data['reviews']),
                'review_details': review_details
            }

        return strictness_results

    def analyze_by_attribute_category(self, user_data):
        """按属性类别分析严格性"""
        print("按属性类别分析严格性...")

        category_analysis = {category: {'strict': 0, 'non_strict': 0} for category in self.basic_attributes.keys()}

        for user_id, data in user_data.items():
            for review, rating in zip(data['reviews'], data['ratings']):
                sentences = sent_tokenize(str(review))

                for sentence in sentences:
                    if self.is_negative_sentence(sentence):
                        # 检查属于哪个基本属性类别
                        sentence_lower = sentence.lower()
                        for category, words in self.basic_attributes.items():
                            if any(word in sentence_lower for word in words):
                                if rating <= 3:
                                    category_analysis[category]['strict'] += 1
                                else:
                                    category_analysis[category]['non_strict'] += 1
                                break

        # 计算每个属性类别的严格指数
        category_strictness = {}
        for category, counts in category_analysis.items():
            total = counts['strict'] + counts['non_strict']
            if total > 0:
                category_strictness[category] = counts['strict'] / total
            else:
                category_strictness[category] = 0

        # 保存到实例变量中
        self.category_analysis = category_analysis
        return category_strictness

    def save_results(self, strictness_results, category_strictness, output_file='strictness_analysis.xlsx'):
        """保存结果"""
        print("保存结果...")

        # 保存消费者严格指数
        strictness_data = []
        for user_id, stats in strictness_results.items():
            strictness_data.append({
                'user_id': user_id,
                'strictness_index': stats['strictness_index'],
                'strict_count': stats['strict_count'],
                'non_strict_count': stats['non_strict_count'],
                'total_relevant_reviews': stats['total_relevant_reviews'],
                'total_reviews': stats['total_reviews'],
                'has_strict_behavior': 1 if stats['strict_count'] > 0 else 0
            })

        strictness_df = pd.DataFrame(strictness_data)
        strictness_df.to_excel(output_file, index=False)

        # 保存属性类别分析
        category_data = []
        for category, strictness in category_strictness.items():
            counts = self.category_analysis[category]
            category_data.append({
                'attribute_category': category,
                'strictness_index': strictness,
                'strict_count': counts['strict'],
                'non_strict_count': counts['non_strict'],
                'total_negative_mentions': counts['strict'] + counts['non_strict']
            })

        category_df = pd.DataFrame(category_data)
        category_df.to_excel('category_strictness_analysis.xlsx', index=False)

        # 保存详细统计
        summary = {
            'total_users': len(strictness_results),
            'users_with_strict_behavior': sum(1 for s in strictness_results.values() if s['strict_count'] > 0),
            'avg_strictness_index': np.mean(
                [s['strictness_index'] for s in strictness_results.values() if s['total_relevant_reviews'] > 0]),
            'median_strictness_index': np.median(
                [s['strictness_index'] for s in strictness_results.values() if s['total_relevant_reviews'] > 0]),
            'category_strictness': category_strictness,
            'category_analysis': self.category_analysis
        }

        with open('strictness_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("结果保存完成!")

        return strictness_df, category_df

    def run_analysis(self):
        """运行完整分析流程"""
        print("开始严格指数分析...")

        # 1. 定义基本属性
        basic_attributes = [
            'atmospherethe_ambiancethe_group',
            'price_attributes',
            'waitress_server_group',
            'courteous_personable_group'
        ]

        self.load_basic_attributes(basic_attributes)

        # 2. 加载消费者数据
        user_data = self.load_consumer_reviews()

        # 3. 计算严格指数
        strictness_results = self.calculate_strictness_index(user_data)

        # 4. 按属性类别分析
        category_strictness = self.analyze_by_attribute_category(user_data)

        # 5. 保存结果
        strictness_df, category_df = self.save_results(strictness_results, category_strictness)

        print("分析完成!")
        return strictness_results, category_strictness


# 主函数
def main():
    analyzer = StrictnessAnalyzer()
    strictness_results, category_strictness = analyzer.run_analysis()

    # 显示摘要信息
    print(f"\n=== 严格指数分析结果摘要 ===")
    print(f"分析用户数量: {len(strictness_results)}")

    # 计算有相关评论的用户
    users_with_relevant = [s for s in strictness_results.values() if s['total_relevant_reviews'] > 0]

    if users_with_relevant:
        strictness_values = [s['strictness_index'] for s in users_with_relevant]
        print(f"有相关评论的用户数: {len(users_with_relevant)}")
        print(f"平均严格指数: {np.mean(strictness_values):.3f}")
        print(f"严格指数标准差: {np.std(strictness_values):.3f}")
        print(f"具有严格行为的用户数: {sum(1 for s in strictness_results.values() if s['strict_count'] > 0)}")
    else:
        print("没有找到包含基本属性负面句子的评论")

    # 显示各属性类别的严格指数
    print(f"\n各属性类别的严格指数:")
    for category, strictness in category_strictness.items():
        counts = analyzer.category_analysis[category]
        print(f"  {category}: {strictness:.3f} (严格: {counts['strict']}, 非严格: {counts['non_strict']})")

    # 显示前几个用户的严格指数
    print(f"\n前5个用户的严格指数:")
    for i, (user_id, stats) in enumerate(strictness_results.items()):
        if i >= 5:
            break
        if stats['total_relevant_reviews'] > 0:
            print(f"用户 {user_id}: 严格指数={stats['strictness_index']:.3f}, "
                  f"严格次数={stats['strict_count']}, 非严格次数={stats['non_strict_count']}")
        else:
            print(f"用户 {user_id}: 无相关评论")


if __name__ == "__main__":
    main()