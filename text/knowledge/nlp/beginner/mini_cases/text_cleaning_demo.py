"""
text_cleaning_demo.py

功能：
    实现文本清洗常见操作，如去标点、统一大小写、去数字、去特殊符号等。
    体验NLP数据预处理的第一步。
用法：
    直接运行脚本，打印原始文本和清洗后的文本。
依赖库：
    re（标准库，无需安装）

输入：
    text = "原始文本字符串"

输出说明：
    - “原始文本”：未处理前的原始内容
    - “清洗后文本”：去除杂质、适合后续分析的文本

作用说明：
    - 文本清洗是所有NLP项目的数据准备基础，可显著提升后续分词与建模效果。
"""

import re

# 示例文本（可自行更换）
text = "2024年，人工智能（AI）正在改变世界！数据量高达100TB。开发者@OpenAI, 你怎么看？#AI#NLP"

print("原始文本：")
print(text)

# 1. 去除标点和特殊符号（保留汉字、英文字母和数字空格）
text_no_punct = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text)

# 2. 去除数字（如有需要可选）
text_no_num = re.sub(r"\d+", "", text_no_punct)

# 3. 统一为小写
text_lower = text_no_num.lower()

# 4. 合并多余空格
cleaned_text = re.sub(r"\s+", " ", text_lower).strip()

print("\n清洗后文本：")
print(cleaned_text)

# 输出示例
# 原始文本：
# 2024年，人工智能（AI）正在改变世界！数据量高达100TB。开发者@OpenAI, 你怎么看？#AI#NLP
#
# 清洗后文本：
# 年 人工智能 ai 正在改变世界 数据量高达 tb 开发者 openai 你怎么看 ai nlp
