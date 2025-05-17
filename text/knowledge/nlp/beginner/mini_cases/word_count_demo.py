"""
word_count_demo.py

功能：
    统计中文（或英文）文本中的高频词，输出出现频率最高的前10个词。
    帮助理解NLP中的“词频统计”与基础文本处理流程。
用法：
    直接运行脚本，输出高频词及其频数。
依赖库：
    collections, re（标准库，无需额外安装）

输入：
    text = "文本字符串"（可直接在代码内修改为自己的文本）

输出说明：
    - “高频词 Top 10”：每个词及其出现次数，按出现频率从高到低排序。

作用说明：
    - 常用于文本分析的预处理、关键词提取、文本概览等。
    - 为后续分词、特征工程、文本建模等提供基础数据。
"""


# 示例文本，可替换为自己读入txt的内容
text = """
人工智能正在改变世界。AI可以自动分析数据、生成文本、识别图片。NLP是AI的一个重要分支，
主要关注自然语言的理解与生成。随着ChatGPT等大模型的出现，自然语言处理越来越火热。
"""

# 文本简单预处理（去标点，统一小写）
import re
text = re.sub(r"[，。！？、.]", " ", text)   # 替换中文标点为空格
text = text.lower()

# 用split分词（英文更准，中文仅为demo，推荐用jieba）
words = text.split()

# 词频统计
from collections import Counter
counter = Counter(words)

# 输出高频词
print("高频词 Top 10:")
for word, freq in counter.most_common(10):
    print(f"{word}: {freq}")

# 结果示例
# 高频词 Top 10:
# 人工智能正在改变世界: 1
# ai可以自动分析数据: 1
# 生成文本: 1
# 识别图片: 1
# nlp是ai的一个重要分支: 1
# 主要关注自然语言的理解与生成: 1
# 随着chatgpt等大模型的出现: 1
# 自然语言处理越来越火热: 1
