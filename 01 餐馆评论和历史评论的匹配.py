import pandas as pd


def process_review_data():
    # 读取Excel文件
    file_path = "2025年8月全部数据.xlsx"

    try:
        # 读取"去重餐馆评论"sheet，获取唯一uid集合和对应的评论信息
        unique_reviewers_sheet = pd.read_excel(file_path, sheet_name="去重餐馆评论")
        unique_uids = set(unique_reviewers_sheet['reviewer_href'].dropna())
        print(f"从'去重餐馆评论'sheet中找到 {len(unique_uids)} 个唯一用户ID")

        # 创建去重评论的字典，key为uid，value为(评论文本, 评分)
        unique_reviews_dict = {}
        for _, row in unique_reviewers_sheet.iterrows():
            uid = row['reviewer_href']
            if pd.notna(uid):
                unique_reviews_dict[uid] = (row['review_text'], row['star_rate'])

        # 读取"全部历史评论"sheet
        all_reviews_sheet = pd.read_excel(file_path, sheet_name="全部历史评论")
        print(f"'全部历史评论'sheet中共有 {len(all_reviews_sheet)} 条评论")

        # 筛选出目标uid的评论
        target_reviews = all_reviews_sheet[all_reviews_sheet['user_id'].isin(unique_uids)]
        print(f"筛选后找到 {len(target_reviews)} 条目标用户的评论")

        # 按user_id分组，统计每个用户的评论信息
        user_reviews_stats = target_reviews.groupby('user_id').agg({
            'review_text': list,  # 收集所有评论内容
            'star_rate': list,  # 收集所有评分
            'user_id': 'count'  # 统计评论数量
        }).rename(columns={'user_id': 'review_count'})

        # 重置索引，使user_id成为一列
        user_reviews_stats.reset_index(inplace=True)

        # 重命名列
        user_reviews_stats.columns = ['uid', 'reviews', 'ratings', 'review_count']

        # 添加去重餐馆评论中的评论文本和打分
        user_reviews_stats['去重评论_文本'] = user_reviews_stats['uid'].map(
            lambda x: unique_reviews_dict.get(x, ('', ''))[0]
        )
        user_reviews_stats['去重评论_评分'] = user_reviews_stats['uid'].map(
            lambda x: unique_reviews_dict.get(x, '')
        )

        return user_reviews_stats

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# 执行处理
result = process_review_data()

if result is not None:
    # 显示前几个结果
    print("\n前5个用户的评论统计：")
    print(result[['uid', 'review_count', '去重评论_评分']].head())

    # 显示更详细的信息
    print("\n详细信息：")
    for i, row in result.head().iterrows():
        print(f"\n用户 {row['uid']}:")
        print(f"评论数量: {row['review_count']}")
        print(f"去重评论评分: {row['去重评论_评分']}")
        print(f"去重评论预览: {str(row['去重评论_文本'])[:100]}...")
        print(f"历史评分列表: {row['ratings'][:5]}{'...' if len(row['ratings']) > 5 else ''}")

    # 保存完整的统计结果到Excel文件
    result.to_excel("用户评论统计完整结果.xlsx", index=False)
    print("\n完整结果已保存到 '用户评论统计完整结果.xlsx'")

    # 保存用户评论数量统计（包含去重评论信息）
    count_stats = result[['uid', 'review_count', '去重评论_文本', '去重评论_评分']].copy()
    count_stats.to_excel("用户评论数量统计.xlsx", index=False)
    print("用户评论数量统计已保存到 '用户评论数量统计.xlsx'")

    # 显示统计摘要
    print(f"\n统计摘要：")
    print(f"总用户数: {len(result)}")
    print(f"总评论数: {result['review_count'].sum()}")
    print(f"平均每个用户评论数: {result['review_count'].mean():.2f}")
    print(f"最多评论数: {result['review_count'].max()}")
    print(f"最少评论数: {result['review_count'].min()}")

    # 检查是否有用户没有对应的去重评论
    missing_reviews = result[result['去重评论_文本'] == '']
    if len(missing_reviews) > 0:
        print(f"\n警告：有 {len(missing_reviews)} 个用户在去重餐馆评论中没有找到对应记录")

else:
    print("处理失败，请检查数据文件")