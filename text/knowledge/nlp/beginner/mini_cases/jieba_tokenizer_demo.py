"""
jieba_tokenizer_demo.py

功能：
    用jieba库对中文文本进行分词，输出分词后的词列表。
    演示NLP中文分词的常用API和分词模式。
用法：
    直接运行脚本，打印精确模式分词结果和全模式分词结果。
依赖库：
    jieba

输入：
    text = "中文文本字符串"（可自由修改）

输出说明：
    - “分词结果”：jieba分词后得到的词语列表
    - “全模式分词”：jieba.cut(text, cut_all=True)的分词列表

作用说明：
    - 中文NLP中，分词是最基础的预处理步骤。
    - 后续的关键词提取、文本分类等任务都依赖分词结果。
"""


import jieba

# 示例文本
text = "人工智能正在改变世界。NLP是自然语言处理的重要分支，近年来发展迅速。"

# 精确模式分词
words = jieba.lcut(text)

print("分词结果：")
print(words)

# 也可以用全模式、搜索引擎模式试试
print("\n全模式分词：")
print(list(jieba.cut(text, cut_all=True)))

# 也可以用全模式、搜索引擎模式试试
print("\n搜索引擎模：")
print(list(jieba.cut_for_search(text)))
