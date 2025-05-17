"""
simple_sentiment_naive_bayes.py

功能：
    用 scikit-learn 的朴素贝叶斯模型实现中文文本的简单情感分析（正面/负面分类）。
    演示入门级文本分类的完整流程（分词→特征→建模→预测）。
用法：
    直接运行脚本，输出分类准确率及每条样本的预测结果。
依赖库：
    jieba, scikit-learn

输入：
    data = [（文本, 标签）] 列表，标签1为正面，0为负面

输出说明：
    - 测试集准确率
    - 每条样本的文本、真实标签、预测标签

作用说明：
    - 体验最基础的NLP建模和自动文本情感判别流程。
    - 入门实战：可自由更换数据、方法，快速见效。
"""

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 1. 示例数据集（(文本, 标签)，1=正面，0=负面）
data = [
    ("这部电影太棒了，感动人心", 1),
    ("垃圾电影，浪费时间", 0),
    ("剧情精彩，演员演技一流", 1),
    ("内容无聊，毫无意义", 0),
    ("喜欢这样的结局，非常满意", 1),
    ("太失望了，不推荐", 0),
    ("视觉效果很震撼，特效一流", 1),
    ("剧情老套，毫无新意", 0),
    ("演员表演自然，代入感强", 1),
    ("故事太拖沓，看得很困", 0),
    ("感情真挚，音乐也很好听", 1),
    ("烂片，简直受不了", 0),
    ("精彩绝伦，值得再刷", 1),
    ("演技尴尬，台词生硬", 0),
    ("节奏紧凑，全程无尿点", 1),
    ("根本不值得花钱", 0)
]


texts, labels = zip(*data)


# 2. 分词
def jieba_tokenizer(text):
    return list(jieba.cut(text))


# 3. 特征提取
vectorizer = CountVectorizer(tokenizer=jieba_tokenizer)
X = vectorizer.fit_transform(texts)
y = list(labels)

# 4. 划分训练集/测试集
# 把数据随机分为“训练集”（机器学的部分）和“测试集”（检验预测准确度的部分）
# 比如这里有6条评论，自动分成4条训练、2条测试
# test_size=0.3 代表“测试集”占 30%，“训练集”占 70%
# random_state=42 随机数种子,保证每次随机分法一样（方便复现实验）
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, texts, test_size=0.3, random_state=42)

# 5. 朴素贝叶斯分类(建模与预测)
clf = MultinomialNB()  # 创建朴素贝叶斯分类器
clf.fit(X_train, y_train)  # 用训练集让模型“学会分辨”正/负面

# 6. 预测
y_pred = clf.predict(X_test)

# 7. 输出结果
print("测试集准确率：%.2f" % (clf.score(X_test, y_test)))
print("\n预测明细：")
for text, label_true, label_pred in zip(text_test, y_test, y_pred):
    print(f"文本: {text} | 真实标签: {label_true} | 预测: {label_pred}")


"""
输出示例：
测试集准确率：0.20

预测明细：
文本: 这部电影太棒了，感动人心 | 真实标签: 1 | 预测: 0
文本: 垃圾电影，浪费时间 | 真实标签: 0 | 预测: 1
文本: 太失望了，不推荐 | 真实标签: 0 | 预测: 0
文本: 节奏紧凑，全程无尿点 | 真实标签: 1 | 预测: 0
文本: 演技尴尬，台词生硬 | 真实标签: 0 | 预测: 1



🎯 1. 为什么模型预测效果差？
① 数据量太小
样本只有16条，分完训练和测试集后每类实际用于训练的很少。

机器学习模型通常需要几十、几百、几千甚至几万条样本才能学到比较可靠的规律。

② 文本特征稀疏/简单
用的是最基础的词袋模型（CountVectorizer），很多句子分词后“词重叠不多”或有用词太少。

比如“这部电影太棒了”和“节奏紧凑，全程无尿点”在分词后没什么重合词，模型不容易找到共性。

③ 中文分词不够精准
jieba分词虽然很方便，但遇到短句和口语时有时切得不理想，可能会丢掉一些关键信息。

④ 小样本容易“过拟合”或者“瞎猜”
训练集数据少，模型可能只“记住了”训练集，对测试集新样本完全没见过，于是随机瞎猜。

⑤ 情感词不典型
有些表达方式是“正面/负面”但不直接包含典型情感词（如“全程无尿点”，其实是夸奖，但模型可能无法分辨）。

🚩 2. 怎么改进？
① 扩充样本数量
多找一些正负面评论（上百条更好，哪怕复制稍微改写也有提升）。

也可以用开源的豆瓣影评/酒店评论/商品评价等小型公开数据集。

② 用TF-IDF特征试试
CountVectorizer可以换成TfidfVectorizer，有时对“小样本”效果会好一点。

③ 分词优化/自定义情感词典
可以考虑在jieba里自定义“情感词典”，让模型更好识别褒贬词。

④ 多跑几次（不同random_state）
由于数据极少，有时换一下随机种子分法，结果可能会更好（但本质还是样本太少）。

优化方法会在进阶脚本中实现。
"""

