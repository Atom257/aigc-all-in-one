"""
wordcloud_demo.py

功能：
    对中文文本分词并统计高频词，用 wordcloud 包可视化成词云图。
    体验NLP数据可视化，发现文本高频关键词的分布。
用法：
    直接运行脚本，会弹出词云图片窗口，也可保存为图片文件。
依赖库：
    jieba, wordcloud, matplotlib
    pip install jieba wordcloud matplotlib

输入：
    text = "中文文本字符串"（可自由修改）

输出说明：
    - 显示一张“词云”图片，高频词字体越大。
    - 也可以保存为 wordcloud.png 文件。

作用说明：
    - 用于文本探索、报告展示、内容摘要、热点词发现等NLP任务。

实践扩展
    - 可以换成自己的评论、新闻、小说文本，看看关键词分布。
    - 用去停用词后的分词结果（和前面脚本结合），效果更好。
    - 词云形状和配色都可自定义，参见 wordcloud 官方文档：https://github.com/amueller/word_cloud。


"""

import jieba
import matplotlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# 1. 示例文本（可自由替换）
text = (
    "人工智能正在改变世界，AI、NLP、大模型和多模态成为研究热点。"
    "数据、算法、算力是AI发展的三大支柱。"
    "机器学习、深度学习在文本、图像、语音等领域取得了突破性进展。"
    "未来，AI将在教育、医疗、金融、自动驾驶等行业深度应用。"
)

# 2. 中文分词
words = jieba.cut(text)
text_seg = " ".join(words)

# 3. 生成词云（推荐用思源黑体/微软雅黑等中文字体）
wc = WordCloud(
    font_path="msyh.ttc",       # Windows可用，Mac/Linux建议换成合适的字体
    width=800, height=400, background_color="white"
).generate(text_seg)

# 4. 显示词云
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("词云示例")
plt.show()

# 5. 可选：保存词云到文件
wc.to_file("wordcloud.png")
print("词云已保存为 wordcloud.png")
