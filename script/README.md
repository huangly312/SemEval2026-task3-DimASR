本文件夹内用于存放一些代码脚本，如 数据统计、可视化处理等

。/script_output 文件夹用于脚本输出

## 脚本文件说明

### ./va_distribution_analysis.py

分析所有语言的train_alltasks.jsonl文件中的VA字段分布情况

后续会考虑补充不同语言/不同领域的VA分布分析

### ./token_length_statistic.py

分析所有语言的train/dev数据经过bert-base-multilingual-cased tokenizer后的token长度分布情况

分语言,分领域,分train/dev,绘制token长度分布直方图