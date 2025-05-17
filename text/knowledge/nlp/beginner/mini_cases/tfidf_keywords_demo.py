"""
tfidf_keywords_demo.py

功能：
    用 scikit-learn + jieba 实现中文文本的关键词自动提取（TF-IDF算法）。
    适合 NLP 初学者体验关键词提取和特征工程。
用法：
    直接运行脚本，将输出每句话的关键词及其权重。
依赖库：
    scikit-learn, jieba

输入：
    corpus = [句子1, 句子2, ...]（在代码内已设置好示例文本）

输出说明：
    - 关键词列表：本组文本中最具区分度的10个关键词。
    - 每句话的关键词权重：对于每条文本，输出这些关键词在其中的“TF-IDF分数”，分数越高说明该词对这句话越有代表性。

输出示例：
    关键词列表： ['ai' 'nlp' '关键词' ...]
    句子2：NLP是自然语言处理的重要分支。
      nlp: 0.589
      自然语言: 0.417
      处理: 0.417
      分支: 0.417

作用说明：
    - 可以用来做新闻/评论/文档的自动标签、摘要、检索推荐等实际NLP任务。
    - 本脚本帮助理解如何用TF-IDF自动提取文本中最有“代表性”的词汇。
"""



from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 示例文本（多句话作为一个小“文档集”）
corpus = [
    "人工智能正在改变世界。",
    "NLP是自然语言处理的重要分支。",
    "近年来，AI和大模型的发展让NLP越来越火。",
    "文本数据需要分词、去停用词和关键词提取。"
]

# 自定义分词器（用jieba处理中文）
def jieba_tokenizer(text):
    return list(jieba.cut(text))

# 初始化TF-IDF向量器
vectorizer = TfidfVectorizer(
    tokenizer=jieba_tokenizer,       # 用jieba分词
    stop_words=['的', '和', '是', '了', '在', '、', '。', '，', '让', '需要', '越来越'],  # 简单停用词
    max_features=10                  # 只取权重最高的10个关键词
)

tfidf = vectorizer.fit_transform(corpus)

# 获取关键词
keywords = vectorizer.get_feature_names_out()
print("关键词列表：", keywords)

# 输出每句话的关键词权重
for i, doc in enumerate(corpus):
    print(f"\n句子{i+1}：{doc}")
    for j, keyword in enumerate(keywords):
        print(f"  {keyword}: {tfidf[i, j]:.3f}")
