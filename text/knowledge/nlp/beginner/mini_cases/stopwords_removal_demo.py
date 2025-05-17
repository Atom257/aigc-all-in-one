"""
stopwords_removal_demo.py

功能：
    对分词后的中文词列表进行去停用词处理，输出去除停用词前后的对比结果。
    演示如何通过停用词过滤降低无效信息干扰。
用法：
    直接运行脚本，打印原始分词结果和去停用词结果。
依赖库：
    jieba

输入：
    text = "中文文本字符串"
    stopwords = set([...])  # 可换为完整的停用词表

输出说明：
    - “原始分词结果”：jieba分词后的所有词语
    - “去停用词后结果”：过滤停用词后的词语列表

作用说明：
    - 停用词过滤可提升后续特征提取和文本建模的有效性。
    - 有助于消除“的”“了”“是”等无实际含义的高频词对模型的干扰。
"""


import jieba

# 示例文本
text = "人工智能正在改变世界。NLP是自然语言处理的重要分支，近年来发展迅速。"

# 分词
words = jieba.lcut(text)

# 简易停用词表（实际项目推荐用完整中文停用词表文件）
stopwords = {'的', '，', '。', '是', '在', '和', '与', '了', '为', '也', '而', '于', '及', '重要'}

# 哈工大停用词表：https://github.com/goto456/stopwords
# 去停用词
filtered_words = [w for w in words if w not in stopwords]

print("原始分词结果：")
print(words)

print("\n去停用词后结果：")
print(filtered_words)
