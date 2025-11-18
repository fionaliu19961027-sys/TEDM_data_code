import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 下载NLTK数据
nltk.download('punkt')
nltk.download('stopwords')


class AttributeAnalyzer:
    def __init__(self, fast_mode=False):
        self.review_df = None
        self.attribute_categories = {}
        self.feature_matrix = None
        self.shap_values = None
        self.category_stats = {}
        self.fast_mode = fast_mode
        self.comparison_results = {}  # 存储不同权重设置的结果

    def load_data(self, review_file='text_review_corpus.xlsx',
                  cluster_file='final_cluster_results.json'):
        """加载评论数据和属性分类数据"""
        print("加载数据...")
        start_time = time.time()

        # 加载评论数据
        try:
            self.review_df = pd.read_excel(review_file)
            print(f"加载评论数据: {len(self.review_df)} 条记录")
            if len(self.review_df) > 10000 and self.fast_mode:
                print("数据量较大，启用快速模式：随机采样5000条数据")
                self.review_df = self.review_df.sample(n=5000, random_state=42)
        except Exception as e:
            print(f"加载评论数据失败: {e}")
            return False

        # 加载属性分类数据
        try:
            with open(cluster_file, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)

            # 解析属性分类
            for cluster_id, cluster_info in cluster_data['clusters'].items():
                category_name = cluster_info['label']
                attributes = cluster_info['words']
                self.attribute_categories[category_name] = attributes

            print(f"加载属性分类: {len(self.attribute_categories)} 个类别")

            # 显示每个类别的属性词数量
            for category, attributes in self.attribute_categories.items():
                print(f"  {category}: {len(attributes)} 个属性词")

            load_time = time.time() - start_time
            print(f"数据加载完成，耗时: {load_time:.1f} 秒")
            return True

        except Exception as e:
            print(f"加载属性分类数据失败: {e}")
            return False

    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return []

        text = str(text).lower()
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # 分词
        tokens = word_tokenize(text)
        # 移除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return tokens

    def extract_category_occurrence(self, text):
        """从文本中提取每个类别的出现情况"""
        tokens = self.preprocess_text(text)
        category_occurrence = {category: 0 for category in self.attribute_categories.keys()}

        for token in tokens:
            for category, attributes in self.attribute_categories.items():
                if token in attributes:
                    category_occurrence[category] = 1  # 只要出现该类别的任何一个词就算出现
                    break  # 一个词只属于一个类别

        return category_occurrence

    def build_feature_matrix(self):
        """构建特征矩阵（每个类别是否出现）"""
        print("构建特征矩阵...")
        start_time = time.time()

        # 为每个类别创建特征列
        feature_data = {category: [] for category in self.attribute_categories.keys()}

        # 处理每条评论
        total_reviews = len(self.review_df)
        category_found_count = {category: 0 for category in self.attribute_categories.keys()}

        # 使用tqdm显示进度条
        for i in tqdm(range(total_reviews), desc="处理评论"):
            row = self.review_df.iloc[i]
            text = row['review_text']
            category_occurrence = self.extract_category_occurrence(text)

            for category, occurred in category_occurrence.items():
                feature_data[category].append(occurred)
                if occurred:
                    category_found_count[category] += 1

        # 显示每个类别的出现次数
        print("\n每个类别的出现次数:")
        for category, count in category_found_count.items():
            percentage = count / total_reviews * 100
            print(f"  {category}: {count}次 ({percentage:.1f}%)")

        self.feature_matrix = pd.DataFrame(feature_data)

        build_time = time.time() - start_time
        print(f"特征矩阵构建完成: {self.feature_matrix.shape}, 耗时: {build_time:.1f} 秒")

        return True

    def calculate_frequency_scores(self):
        """计算每个类别的频次分数"""
        print("计算频次分数...")
        start_time = time.time()

        if self.feature_matrix is None or self.feature_matrix.empty:
            print("特征矩阵为空，无法计算频次")
            return None

        # 统计每个类别出现的总次数
        category_counts = self.feature_matrix.sum()

        # 归一化频次分数 (0-1)
        min_count = category_counts.min()
        max_count = category_counts.max()

        if max_count > min_count:
            frequency_scores = (category_counts - min_count) / (max_count - min_count)
        else:
            frequency_scores = pd.Series([0] * len(category_counts), index=category_counts.index)

        # 存储统计信息
        for category in category_counts.index:
            self.category_stats[category] = {
                'frequency': int(category_counts[category]),
                'frequency_score': float(frequency_scores[category]),
                'frequency_percentage': float(category_counts[category] / len(self.review_df) * 100)
            }

        freq_time = time.time() - start_time
        print(f"频次计算完成，耗时: {freq_time:.1f} 秒")
        return frequency_scores

    def train_model_and_calculate_shap(self):
        """训练模型并计算SHAP值"""
        print("训练预测模型...")
        start_time = time.time()

        if self.feature_matrix is None or self.feature_matrix.empty:
            print("特征矩阵为空，无法训练模型")
            return None

        # 准备特征和目标变量
        X = self.feature_matrix
        y = self.review_df['star_rate'].values

        # 简化模式：使用更少的树和更小的测试集
        if self.fast_mode:
            test_size = 0.1
            n_estimators = 50
        else:
            test_size = 0.2
            n_estimators = 100

        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 训练随机森林模型
        try:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            # 评估模型
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            print(f"模型性能 - R²: {r2:.3f}, RMSE: {rmse:.3f}")

            if r2 < 0.1:
                print("警告: 模型解释力较低，结果可能需要谨慎解读")

            # 计算SHAP值
            print("计算SHAP值...")

            try:
                # 使用interventional方法避免additivity check错误
                explainer = shap.TreeExplainer(model, feature_perturbation="interventional")

                if self.fast_mode:
                    # 快速模式：使用较小的样本计算SHAP
                    shap_sample = X.sample(n=min(1000, len(X)), random_state=42)
                    shap_values = explainer.shap_values(shap_sample)
                else:
                    # 完整模式
                    shap_values = explainer.shap_values(X)

                # 计算每个类别的平均绝对SHAP值
                category_shap = {}
                for category in X.columns:
                    col_idx = X.columns.get_loc(category)
                    abs_shap = np.mean(np.abs(shap_values[:, col_idx]))
                    category_shap[category] = abs_shap

                # 归一化SHAP分数 (0-1)
                shap_values_list = list(category_shap.values())
                min_shap = min(shap_values_list) if shap_values_list else 0
                max_shap = max(shap_values_list) if shap_values_list else 1

                for category in X.columns:
                    if category in category_shap:
                        if max_shap > min_shap:
                            shap_score = (category_shap[category] - min_shap) / (max_shap - min_shap)
                        else:
                            shap_score = 0

                        if category in self.category_stats:
                            self.category_stats[category]['shap_value'] = float(category_shap[category])
                            self.category_stats[category]['shap_score'] = float(shap_score)

                self.shap_values = category_shap

                shap_time = time.time() - start_time
                print(f"SHAP计算完成，耗时: {shap_time:.1f} 秒")
                return category_shap

            except Exception as e:
                print(f"SHAP计算失败: {e}")
                print("使用特征重要性作为替代")
                return self.calculate_feature_importance(model, X)

        except Exception as e:
            print(f"模型训练失败: {e}")
            return None

    def calculate_feature_importance(self, model, X):
        """使用特征重要性作为SHAP的替代"""
        print("使用特征重要性作为替代指标...")

        feature_importance = model.feature_importances_
        category_importance = {}

        for i, category in enumerate(X.columns):
            if i < len(feature_importance):
                category_importance[category] = feature_importance[i]

        # 归一化重要性分数 (0-1)
        importance_values = list(category_importance.values())
        min_imp = min(importance_values) if importance_values else 0
        max_imp = max(importance_values) if importance_values else 1

        for category in X.columns:
            if category in category_importance:
                if max_imp > min_imp:
                    imp_score = (category_importance[category] - min_imp) / (max_imp - min_imp)
                else:
                    imp_score = 0

                if category in self.category_stats:
                    self.category_stats[category]['shap_value'] = float(category_importance[category])
                    self.category_stats[category]['shap_score'] = float(imp_score)

        self.shap_values = category_importance
        return category_importance

    def calculate_composite_scores(self, alpha=0.4, beta=0.6):
        """计算综合得分"""
        composite_scores = {}
        for category, stats in self.category_stats.items():
            freq_score = stats.get('frequency_score', 0)
            shap_score = stats.get('shap_score', 0)
            composite_score = alpha * freq_score + beta * shap_score
            composite_scores[category] = composite_score

        return composite_scores

    def compare_different_weights(self):
        """比较不同权重设置下的结果"""
        print("\n=== 不同权重设置对比分析 ===")

        # 定义不同的权重组合
        weight_combinations = [
            ('仅频次', 1.0, 0.0),
            ('仅解释力', 0.0, 1.0),
            ('频次主导', 0.7, 0.3),
            ('均衡', 0.5, 0.5),
            ('解释力主导', 0.3, 0.7),
            ('当前设置', 0.4, 0.6)
        ]

        comparison_results = {}

        for name, alpha, beta in weight_combinations:
            composite_scores = self.calculate_composite_scores(alpha, beta)

            # 排序并选择基础类别（前50%）
            sorted_categories = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            threshold_rank = max(1, len(sorted_categories) // 2)  # 选择前50%作为基础属性
            basic_categories = [cat for cat, score in sorted_categories[:threshold_rank]]

            comparison_results[name] = {
                'alpha': alpha,
                'beta': beta,
                'basic_categories': basic_categories,
                'composite_scores': composite_scores,
                'ranking': [cat for cat, score in sorted_categories]
            }

            print(f"\n{name} (α={alpha}, β={beta}):")
            print(f"  基础属性类别: {', '.join(basic_categories)}")

        self.comparison_results = comparison_results
        return comparison_results

    def visualize_weight_comparison(self):
        """可视化不同权重设置的比较结果"""
        if not self.comparison_results:
            return

        # 创建比较数据
        comparison_data = []
        categories = list(self.category_stats.keys())

        for weight_name, result in self.comparison_results.items():
            for i, category in enumerate(result['ranking']):
                rank = i + 1
                comparison_data.append({
                    'weight_setting': weight_name,
                    'category': category,
                    'rank': rank,
                    'alpha': result['alpha'],
                    'beta': result['beta']
                })

        comparison_df = pd.DataFrame(comparison_data)

        # 创建热力图显示排名
        pivot_df = comparison_df.pivot_table(
            index='category',
            columns='weight_setting',
            values='rank',
            aggfunc='first'
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='d', cmap='RdYlGn_r',
                    cbar_kws={'label': '排名 (1=最好)'})
        plt.title('不同权重设置下的属性类别排名比较')
        plt.tight_layout()
        plt.savefig('weight_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 创建雷达图显示权重敏感性
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

        categories = list(self.category_stats.keys())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图

        # 为每个权重设置绘制一条线
        for weight_name, result in self.comparison_results.items():
            if weight_name not in ['仅频次', '仅解释力']:  # 只显示组合权重
                scores = [result['composite_scores'][cat] for cat in categories]
                scores += scores[:1]  # 闭合雷达图
                ax.plot(angles, scores, label=weight_name)
                ax.fill(angles, scores, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_title('不同权重设置下的综合得分比较')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('weight_comparison_radar.png', dpi=300, bbox_inches='tight')
        plt.show()

    def identify_basic_categories(self, alpha=0.4, beta=0.6, threshold_percentile=80):
        """识别基础属性类别"""
        print("识别基础属性类别...")

        if not self.category_stats:
            print("没有类别统计信息")
            return [], [], 0

        composite_scores = self.calculate_composite_scores(alpha, beta)
        scores_list = list(composite_scores.values())
        threshold = np.percentile(scores_list, threshold_percentile) if scores_list else 0

        basic_categories = []
        other_categories = []

        for category, score in composite_scores.items():
            if score >= threshold:
                basic_categories.append(category)
            else:
                other_categories.append(category)

        print(f"基础属性类别: {len(basic_categories)} 个")
        print(f"其他属性类别: {len(other_categories)} 个")

        return basic_categories, other_categories, threshold

    def visualize_results(self, basic_categories, other_categories):
        """可视化结果"""
        print("生成可视化...")

        # 准备可视化数据
        viz_data = []
        for category, stats in self.category_stats.items():
            viz_data.append({
                'category': category,
                'frequency': stats['frequency'],
                'frequency_percentage': stats['frequency_percentage'],
                'shap_value': stats.get('shap_value', 0),
                'shap_score': stats.get('shap_score', 0),
                'composite_score': stats.get('composite_score', 0),
                'is_basic': category in basic_categories
            })

        viz_df = pd.DataFrame(viz_data)

        if len(viz_df) > 1:
            try:
                # 1. 散点图：频次 vs SHAP值
                plt.figure(figsize=(12, 8))
                colors = ['red' if x else 'blue' for x in viz_df['is_basic']]
                plt.scatter(viz_df['frequency'], viz_df['shap_value'], c=colors, alpha=0.6, s=100)

                # 添加类别标签
                for i, row in viz_df.iterrows():
                    plt.annotate(row['category'], (row['frequency'], row['shap_value']),
                                 xytext=(5, 5), textcoords='offset points', fontsize=10,
                                 fontweight='bold' if row['is_basic'] else 'normal')

                plt.xlabel('Frequency')
                plt.ylabel('SHAP Value (Absolute Impact)')
                plt.title('Category Analysis: Frequency vs SHAP Value')
                plt.grid(True, alpha=0.3)
                plt.savefig('category_analysis_scatter.png', dpi=300, bbox_inches='tight')
                plt.show()

            except Exception as e:
                print(f"散点图生成失败: {e}")

        return viz_df

    def save_results(self, basic_categories, other_categories, viz_df):
        """保存结果"""
        print("保存结果...")

        # 保存详细类别统计
        detailed_results = []
        for category, stats in self.category_stats.items():
            detailed_results.append({
                'category': category,
                'frequency': stats['frequency'],
                'frequency_percentage': stats['frequency_percentage'],
                'shap_value': stats.get('shap_value', 0),
                'shap_score': stats.get('shap_score', 0),
                'composite_score': stats.get('composite_score', 0),
                'is_basic': category in basic_categories
            })

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_excel('category_analysis_detailed.xlsx', index=False)

        # 保存权重比较结果
        if self.comparison_results:
            comparison_data = []
            for weight_name, result in self.comparison_results.items():
                for category in self.category_stats.keys():
                    comparison_data.append({
                        'weight_setting': weight_name,
                        'category': category,
                        'composite_score': result['composite_scores'].get(category, 0),
                        'rank': result['ranking'].index(category) + 1 if category in result['ranking'] else len(
                            result['ranking']),
                        'is_basic': category in result['basic_categories']
                    })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel('weight_comparison_results.xlsx', index=False)

        # 保存汇总报告
        summary = {
            'total_categories': len(self.category_stats),
            'basic_categories_count': len(basic_categories),
            'other_categories_count': len(other_categories),
            'basic_categories_ratio': len(basic_categories) / len(self.category_stats),
            'analysis_parameters': {
                'alpha': 0.4,
                'beta': 0.6,
                'threshold_percentile': 80
            },
            'basic_categories': basic_categories,
            'other_categories': other_categories,
            'weight_comparison': {k: v['basic_categories'] for k, v in self.comparison_results.items()}
        }

        with open('category_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("所有结果已保存!")

    def run_analysis(self, alpha=0.4, beta=0.6, threshold_percentile=80):
        """运行完整分析流程"""
        print("开始属性类别分析...")
        total_start_time = time.time()

        if self.fast_mode:
            print("⚠️  快速模式已启用，结果可能略有不同")

        # 1. 加载数据
        if not self.load_data():
            return None, None

        # 2. 构建特征矩阵
        if not self.build_feature_matrix():
            return None, None

        # 3. 计算频次分数
        freq_scores = self.calculate_frequency_scores()
        if freq_scores is None:
            return None, None

        # 4. 训练模型和计算SHAP值
        shap_scores = self.train_model_and_calculate_shap()
        if shap_scores is None:
            # 如果SHAP计算失败，只使用频次分数
            print("使用频次分数作为综合得分")
            for category in self.category_stats:
                self.category_stats[category]['composite_score'] = self.category_stats[category]['frequency_score']

        # 5. 比较不同权重设置
        self.compare_different_weights()

        # 6. 使用指定权重识别基础属性类别
        basic_categories, other_categories, threshold = self.identify_basic_categories(alpha, beta,
                                                                                       threshold_percentile)

        # 7. 可视化
        viz_df = self.visualize_results(basic_categories, other_categories)
        self.visualize_weight_comparison()

        # 8. 保存结果
        self.save_results(basic_categories, other_categories, viz_df)

        total_time = time.time() - total_start_time
        print(f"分析完成! 总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
        return basic_categories, other_categories


# 主函数
def main():
    analyzer = AttributeAnalyzer(fast_mode=True)
    basic_cats, other_cats = analyzer.run_analysis()

    if basic_cats is not None:
        print(f"\n=== 最终分析结果摘要 ===")
        print(f"总属性类别数量: {len(analyzer.category_stats)}")
        print(f"基础属性类别: {len(basic_cats)} 个")
        print(f"其他属性类别: {len(other_cats)} 个")

        if basic_cats:
            print(f"\n基础属性类别列表 (按综合得分排序):")
            composite_scores = analyzer.calculate_composite_scores(0.4, 0.6)
            sorted_categories = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)

            for category, score in sorted_categories:
                if category in basic_cats:
                    stats = analyzer.category_stats[category]
                    print(f"  {category}: 综合得分={score:.3f}, "
                          f"频次={stats['frequency']} ({stats['frequency_percentage']:.1f}%), "
                          f"SHAP值={stats.get('shap_value', 0):.4f}")
    else:
        print("分析失败，请检查数据")


if __name__ == "__main__":
    main()