import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict
from scipy import stats
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 下载必要数据
nltk.download('vader_lexicon')
nltk.download('punkt')


class EnhancedUtilityGainAnalyzer:
    def __init__(self, delta=0.2, min_data_points=3):
        self.delta = delta
        self.min_data_points = min_data_points  # 降低数据点要求
        self.sia = SentimentIntensityAnalyzer()
        self.all_attributes = {}  # 使用所有属性

    def load_data(self):
        """加载所有数据"""
        print("加载数据...")

        # 加载评论数据
        try:
            df = pd.read_excel('2025年8月全部数据.xlsx', sheet_name='全部历史评论')
            user_data = {}
            for user_id, group in df.groupby('user_id'):
                user_data[user_id] = {
                    'reviews': group['review_text'].tolist(),
                    'ratings': group['star_rate'].tolist()
                }
            print(f"加载了 {len(user_data)} 个用户的评论数据")
        except Exception as e:
            print(f"加载评论数据失败: {e}")
            return None

        # 加载所有属性词汇
        try:
            with open('final_cluster_results.json', 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)

            self.all_attributes = {}
            for cluster_id, cluster_info in cluster_data['clusters'].items():
                self.all_attributes[cluster_info['label']] = set(cluster_info['words'])

            print(f"加载了 {len(self.all_attributes)} 个属性类别")

        except Exception as e:
            print(f"加载属性词典失败: {e}")
            # 使用默认属性词汇
            self.all_attributes = self._get_default_attributes()

        return user_data

    def _get_default_attributes(self):
        """获取默认属性词汇"""
        return {
            'service': {'service', 'staff', 'waiter', 'waitress', 'friendly', 'helpful'},
            'food': {'food', 'taste', 'delicious', 'fresh', 'quality', 'flavor'},
            'environment': {'environment', 'atmosphere', 'clean', 'comfortable', 'decor'},
            'price': {'price', 'expensive', 'affordable', 'value', 'cost', 'worth'},
            'location': {'location', 'convenient', 'parking', 'access', 'nearby'},
            'experience': {'experience', 'enjoy', 'recommend', 'return', 'visit'}
        }

    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text

    def analyze_sentence_sentiment(self, sentence):
        """分析句子情感得分"""
        return self.sia.polarity_scores(sentence)['compound']

    def extract_all_attribute_sentences(self, review_text):
        """提取包含任何属性词汇的句子"""
        sentences = sent_tokenize(str(review_text))
        attribute_sentences = []

        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            # 检查是否包含任何属性词汇
            for attribute_words in self.all_attributes.values():
                if any(word in attribute_words for word in words):
                    attribute_sentences.append(sentence)
                    break  # 一个句子可能包含多个属性，但只计数一次

        return attribute_sentences

    def robust_segment_regression(self, sentiments, ratings):
        """健壮的分段回归分析"""
        sentiments = np.array(sentiments)
        ratings = np.array(ratings)

        results = {'neg': {'slope': np.nan, 'n_points': 0},
                   'pos': {'slope': np.nan, 'n_points': 0}}

        # 负面区域
        neg_mask = sentiments < 0
        if np.sum(neg_mask) >= 2:  # 至少2个点
            try:
                slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = stats.linregress(
                    sentiments[neg_mask], ratings[neg_mask]
                )
                results['neg'].update({
                    'slope': slope_neg,
                    'intercept': intercept_neg,
                    'r_squared': r_value_neg ** 2
                })
            except:
                pass

        # 正面区域
        pos_mask = sentiments >= 0
        if np.sum(pos_mask) >= 2:
            try:
                slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos = stats.linregress(
                    sentiments[pos_mask], ratings[pos_mask]
                )
                results['pos'].update({
                    'slope': slope_pos,
                    'intercept': intercept_pos,
                    'r_squared': r_value_pos ** 2
                })
            except:
                pass

        results['neg']['n_points'] = np.sum(neg_mask)
        results['pos']['n_points'] = np.sum(pos_mask)

        return results

    def classify_utility_type(self, slope_neg, slope_pos):
        """分类效用增益类型"""
        if np.isnan(slope_neg) or np.isnan(slope_pos) or abs(slope_neg) < 1e-10:
            return 'unknown', np.nan

        p_value = abs(slope_pos) / abs(slope_neg)

        if p_value > 1 + self.delta:
            return 'left_flat_right_steep', p_value
        elif p_value < 1 - self.delta:
            return 'left_steep_right_flat', p_value
        else:
            return 'linear', p_value

    def analyze_user_utility(self, user_reviews, user_ratings):
        """分析单个用户的效用增益特征"""
        all_sentiments = []
        all_ratings = []

        # 收集所有包含属性词汇的句子
        for review, rating in zip(user_reviews, user_ratings):
            preprocessed_review = self.preprocess_text(review)
            attribute_sentences = self.extract_all_attribute_sentences(preprocessed_review)

            for sentence in attribute_sentences:
                sentiment = self.analyze_sentence_sentiment(sentence)
                all_sentiments.append(sentiment)
                all_ratings.append(rating)

        # 确保每个用户都有分析结果
        utility_profile = {
            'dominant_type': 'unknown',
            'p_value': np.nan,
            'total_data_points': len(all_sentiments),
            'neg_slope': np.nan,
            'pos_slope': np.nan,
            'regression_quality': 'poor'
        }

        if len(all_sentiments) >= self.min_data_points:
            try:
                # 进行分段回归
                regression_results = self.robust_segment_regression(all_sentiments, all_ratings)
                slope_neg = regression_results['neg'].get('slope', np.nan)
                slope_pos = regression_results['pos'].get('slope', np.nan)

                utility_profile.update({
                    'neg_slope': slope_neg,
                    'pos_slope': slope_pos,
                    'neg_points': regression_results['neg']['n_points'],
                    'pos_points': regression_results['pos']['n_points']
                })

                # 分类效用类型
                if not np.isnan(slope_neg) and not np.isnan(slope_pos):
                    utility_type, p_value = self.classify_utility_type(slope_neg, slope_pos)
                    utility_profile.update({
                        'dominant_type': utility_type,
                        'p_value': p_value,
                        'regression_quality': 'good'
                    })

                # 如果只有单侧数据，进行启发式判断
                elif not np.isnan(slope_neg) and np.isnan(slope_pos):
                    utility_profile['dominant_type'] = 'left_steep_only'
                elif np.isnan(slope_neg) and not np.isnan(slope_pos):
                    utility_profile['dominant_type'] = 'right_steep_only'

            except Exception as e:
                utility_profile['error'] = str(e)

        return utility_profile

    def run_analysis(self):
        """运行分析"""
        user_data = self.load_data()
        if user_data is None:
            return None

        print("开始效用增益特征分析...")

        results = {}
        for user_id, data in tqdm(user_data.items(), desc="分析消费者"):
            try:
                utility_profile = self.analyze_user_utility(data['reviews'], data['ratings'])
                results[user_id] = utility_profile
            except Exception as e:
                # 确保即使出错也有基本结果
                results[user_id] = {
                    'dominant_type': 'error',
                    'p_value': np.nan,
                    'total_data_points': 0,
                    'error': str(e)
                }

        return results

    def enhance_results_with_fallback(self, results):
        """使用回退策略增强结果"""
        enhanced_results = results.copy()

        # 统计类型分布
        type_counts = defaultdict(int)
        valid_p_values = []

        for user_id, profile in results.items():
            if profile['dominant_type'] not in ['unknown', 'error']:
                type_counts[profile['dominant_type']] += 1
            if not np.isnan(profile['p_value']):
                valid_p_values.append(profile['p_value'])

        # 计算平均p值
        avg_p_value = np.mean(valid_p_values) if valid_p_values else 1.0

        # 为unknown类型的用户分配最常见类型
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'linear'

        for user_id, profile in enhanced_results.items():
            if profile['dominant_type'] in ['unknown', 'error']:
                enhanced_results[user_id] = {
                    'dominant_type': most_common_type,
                    'p_value': avg_p_value,
                    'total_data_points': profile.get('total_data_points', 0),
                    'neg_slope': np.nan,
                    'pos_slope': np.nan,
                    'regression_quality': 'imputed',
                    'original_type': profile['dominant_type']
                }

        return enhanced_results

    def save_results(self, results, output_file='enhanced_utility_analysis.xlsx'):
        """保存结果"""
        # 准备数据
        result_data = []

        for user_id, profile in results.items():
            result_data.append({
                'user_id': user_id,
                'utility_type': profile['dominant_type'],
                'p_value': profile['p_value'],
                'total_data_points': profile['total_data_points'],
                'neg_slope': profile.get('neg_slope', np.nan),
                'pos_slope': profile.get('pos_slope', np.nan),
                'neg_points': profile.get('neg_points', 0),
                'pos_points': profile.get('pos_points', 0),
                'regression_quality': profile.get('regression_quality', 'unknown'),
                'has_error': 'error' in profile
            })

        # 保存到Excel
        df = pd.DataFrame(result_data)
        df.to_excel(output_file, index=False)

        # 保存统计信息
        stats = self.calculate_statistics(df)
        with open('utility_analysis_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到 {output_file}")
        return df

    def calculate_statistics(self, result_df):
        """计算统计信息"""
        stats = {
            'total_users': len(result_df),
            'type_distribution': result_df['utility_type'].value_counts().to_dict(),
            'avg_p_value': result_df['p_value'].mean(),
            'median_p_value': result_df['p_value'].median(),
            'avg_data_points': result_df['total_data_points'].mean(),
            'users_with_less_than_min_data': len(result_df[result_df['total_data_points'] < self.min_data_points]),
            'regression_quality_distribution': result_df['regression_quality'].value_counts().to_dict()
        }
        return stats


# 主函数
def main():
    # 初始化分析器（降低数据点要求）
    analyzer = EnhancedUtilityGainAnalyzer(delta=0.2, min_data_points=3)

    # 运行分析
    raw_results = analyzer.run_analysis()

    if raw_results:
        # 使用回退策略确保所有用户都有有效结果
        enhanced_results = analyzer.enhance_results_with_fallback(raw_results)

        # 保存结果
        result_df = analyzer.save_results(enhanced_results)

        # 显示摘要信息
        print("\n=== 分析完成 ===")
        print(f"分析了 {len(enhanced_results)} 个消费者")
        print(f"效用类型分布:")
        for type_name, count in result_df['utility_type'].value_counts().items():
            print(f"  {type_name}: {count} 用户")

        print(f"\n平均每个用户数据点: {result_df['total_data_points'].mean():.1f}")
        print(f"平均p值: {result_df['p_value'].mean():.3f}")


if __name__ == "__main__":
    main()