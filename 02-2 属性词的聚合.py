import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from collections import defaultdict
import re


def convert_numpy_types(obj):
    """将numpy数据类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def load_results_and_model():
    """加载挖掘结果和Word2Vec模型"""
    # 加载挖掘结果
    with open('attribute_mining_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 加载Word2Vec模型
    try:
        model = Word2Vec.load('word2vec_model.model')
        print("Word2Vec模型加载成功")
    except:
        print("无法加载Word2Vec模型，请确保已运行之前的代码")
        return None, None

    return results, model


def prepare_clustering_data(results, model):
    """准备聚类数据"""
    # 获取所有候选词（包括种子词和发现的候选词）
    all_words = set(results['seed_dictionary'])

    # 添加候选词
    for candidate_info in results['candidate_words']:
        if isinstance(candidate_info, list) and len(candidate_info) >= 1:
            all_words.add(candidate_info[0])
        elif isinstance(candidate_info, str):
            all_words.add(candidate_info)

    # 过滤掉不在词汇表中的词
    valid_words = [word for word in all_words if word in model.wv]
    print(f"有效词汇数量: {len(valid_words)}/{len(all_words)}")

    # 获取词向量
    word_vectors = np.array([model.wv[word] for word in valid_words])

    return valid_words, word_vectors


def determine_optimal_clusters(word_vectors, max_clusters=15):
    """使用肘部法则和轮廓系数确定最佳聚类数量"""
    distortions = []
    silhouette_scores = []
    cluster_range = range(2, min(max_clusters + 1, len(word_vectors) - 1))

    for n_clusters in cluster_range:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(word_vectors)
            distortions.append(kmeans.inertia_)

            if n_clusters > 1:  # 轮廓系数需要至少2个聚类
                labels = kmeans.labels_
                if len(set(labels)) > 1:  # 确保有多个聚类
                    silhouette_avg = silhouette_score(word_vectors, labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
        except Exception as e:
            print(f"聚类数量 {n_clusters} 时出错: {e}")
            distortions.append(0)
            silhouette_scores.append(0)

    return distortions, silhouette_scores, list(cluster_range)


def plot_cluster_analysis(distortions, silhouette_scores, cluster_range):
    """绘制聚类分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 肘部法则图
    ax1.plot(cluster_range, distortions, 'bx-')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Distortion')
    ax1.set_title('Elbow Method For Optimal k')
    ax1.grid(True)

    # 轮廓系数图
    ax2.plot(cluster_range, silhouette_scores, 'rx-')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score For Optimal k')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_clustering(word_vectors, valid_words, n_clusters):
    """执行K-means聚类"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(word_vectors)

    # 创建聚类结果字典
    cluster_results = defaultdict(list)
    for word, cluster_id in zip(valid_words, cluster_labels):
        cluster_results[int(cluster_id)].append(word)  # 转换为int类型

    return cluster_results, kmeans


def analyze_clusters(cluster_results, model):
    """分析聚类结果，为每个聚类生成标签"""
    cluster_analysis = {}

    for cluster_id, words in cluster_results.items():
        # 计算聚类中心词（与聚类中心最相似的词）
        cluster_vectors = np.array([model.wv[word] for word in words])
        cluster_center = cluster_vectors.mean(axis=0)

        # 找到与聚类中心最相似的词
        similarities = []
        for word in words:
            word_vector = model.wv[word]
            similarity = np.dot(cluster_center, word_vector) / (
                    np.linalg.norm(cluster_center) * np.linalg.norm(word_vector)
            )
            similarities.append((word, float(similarity)))  # 转换为float类型

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 选择top词作为标签候选
        top_words = [word for word, sim in similarities[:5]]

        # 基于常见模式生成标签
        label = generate_cluster_label(top_words, words)

        cluster_analysis[int(cluster_id)] = {  # 确保键是int类型
            'label': label,
            'words': words,
            'size': len(words),
            'top_words': top_words,
            'all_similarities': similarities
        }

    return cluster_analysis


def generate_cluster_label(top_words, all_words):
    """为聚类生成有意义的标签"""
    # 常见属性类别映射
    category_patterns = {
        'food': ['flavor', 'taste', 'delicious', 'fresh', 'quality'],
        'service': ['service', 'friendly', 'attentive', 'staff', 'waiter'],
        'ambiance': ['ambiance', 'atmosphere', 'decor', 'clean', 'comfortable'],
        'price': ['price', 'expensive', 'affordable', 'value', 'worth'],
        'menu': ['menu', 'dish', 'appetizer', 'entree', 'dessert'],
        'location': ['location', 'parking', 'reservation', 'wait', 'recommend']
    }

    # 检查与已知类别的匹配
    for category, keywords in category_patterns.items():
        matches = sum(1 for word in top_words if word in keywords)
        if matches >= 2:  # 如果有至少2个匹配，使用该类别
            return f"{category}_attributes"

    # 如果没有明显匹配，使用top词生成标签
    if len(top_words) >= 3:
        return f"{top_words[0]}_{top_words[1]}_group"
    else:
        return f"cluster_{hash(tuple(top_words)) % 1000}"


def visualize_clusters(word_vectors, cluster_labels, valid_words, kmeans):
    """可视化聚类结果"""
    # 使用PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    word_vectors_2d = pca.fit_transform(word_vectors)

    # 创建可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1],
                          c=cluster_labels, cmap='tab20', alpha=0.7, s=100)

    # 添加标签（只显示部分词以避免拥挤）
    for i, word in enumerate(valid_words):
        if i % 3 == 0:  # 每3个词显示一个标签
            plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                         xytext=(5, 2), textcoords='offset points',
                         fontsize=8, alpha=0.8)

    plt.colorbar(scatter)
    plt.title('Word Clusters Visualization (PCA-reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('word_clusters_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return word_vectors_2d


def save_cluster_results(cluster_analysis, word_vectors_2d, valid_words, cluster_labels):
    """保存聚类结果"""
    # 创建详细的结果DataFrame
    results_data = []
    for i, word in enumerate(valid_words):
        cluster_id = int(cluster_labels[i])  # 转换为int类型
        cluster_info = cluster_analysis[cluster_id]
        results_data.append({
            'word': word,
            'cluster_id': cluster_id,
            'cluster_label': cluster_info['label'],
            'pca_x': float(word_vectors_2d[i, 0]),  # 转换为float类型
            'pca_y': float(word_vectors_2d[i, 1]),  # 转换为float类型
            'in_top_words': word in cluster_info['top_words']
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_excel('word_clusters_detailed.xlsx', index=False)

    # 保存聚类摘要
    cluster_summary = []
    for cluster_id, info in cluster_analysis.items():
        cluster_summary.append({
            'cluster_id': int(cluster_id),  # 转换为int类型
            'cluster_label': info['label'],
            'word_count': int(info['size']),  # 转换为int类型
            'top_5_words': ', '.join(info['top_words']),
            'all_words': ', '.join(info['words'])
        })

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.to_excel('cluster_summary.xlsx', index=False)

    # 保存JSON格式的完整结果（确保所有数据类型都是JSON可序列化的）
    final_results = {
        'clusters': convert_numpy_types(cluster_analysis),
        'clustering_metrics': {
            'total_words': int(len(valid_words)),
            'total_clusters': int(len(cluster_analysis)),
            'average_cluster_size': float(np.mean([info['size'] for info in cluster_analysis.values()]))
        }
    }

    with open('final_cluster_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    return results_df, summary_df


def main():
    """主函数：执行同义属性词聚类"""
    print("开始同义属性词聚类...")

    # 1. 加载数据和模型
    results, model = load_results_and_model()
    if results is None or model is None:
        return

    # 2. 准备聚类数据
    valid_words, word_vectors = prepare_clustering_data(results, model)
    print(f"准备进行聚类的词汇: {valid_words[:10]}...")  # 显示前10个词

    if len(valid_words) < 3:
        print("词汇数量太少，无法进行聚类分析")
        return

    # 3. 确定最佳聚类数量
    print("确定最佳聚类数量...")
    distortions, silhouette_scores, cluster_range = determine_optimal_clusters(word_vectors)

    if not silhouette_scores or max(silhouette_scores) == 0:
        print("无法确定最佳聚类数量，使用默认值5")
        optimal_clusters = 5
    else:
        # 绘制分析图
        plot_cluster_analysis(distortions, silhouette_scores, cluster_range)

        # 选择最佳聚类数量（基于轮廓系数）
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"根据轮廓系数确定的最佳聚类数量: {optimal_clusters}")

    # 4. 执行聚类
    print(f"执行K-means聚类 (k={optimal_clusters})...")
    cluster_results, kmeans = perform_clustering(word_vectors, valid_words, optimal_clusters)

    # 5. 分析聚类结果
    print("分析聚类结果并生成标签...")
    cluster_analysis = analyze_clusters(cluster_results, model)

    # 6. 可视化聚类结果
    print("生成可视化...")
    try:
        word_vectors_2d = visualize_clusters(word_vectors, kmeans.labels_, valid_words, kmeans)
    except Exception as e:
        print(f"可视化生成失败: {e}")
        word_vectors_2d = np.zeros((len(valid_words), 2))

    # 7. 保存结果
    print("保存聚类结果...")
    detailed_df, summary_df = save_cluster_results(cluster_analysis, word_vectors_2d, valid_words, kmeans.labels_)

    # 8. 显示摘要
    print("\n=== 聚类结果摘要 ===")
    print(f"总词汇数: {len(valid_words)}")
    print(f"聚类数量: {len(cluster_analysis)}")
    print("\n各聚类详情:")
    for cluster_id, info in cluster_analysis.items():
        print(f"聚类 {cluster_id} ({info['label']}): {info['size']}个词")
        print(f"  代表性词汇: {', '.join(info['top_words'][:5])}")
        print(f"  所有词汇: {', '.join(info['words'][:8])}{'...' if len(info['words']) > 8 else ''}")
        print()

    print("结果文件已保存:")
    print("- word_clusters_detailed.xlsx (详细聚类结果)")
    print("- cluster_summary.xlsx (聚类摘要)")
    print("- final_cluster_results.json (完整JSON结果)")
    print("- cluster_analysis.png (聚类分析图)")
    print("- word_clusters_visualization.png (聚类可视化)")

    return cluster_analysis, detailed_df, summary_df


if __name__ == "__main__":
    cluster_analysis, detailed_df, summary_df = main()