import numpy as np
import pandas
import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#path
data_path = './date'
train_file = './data/train.json'
test_file = './data/test1.json'

df = pd.read_json(train_file)
print(df.keys())
print(df['id'].shape)
print(df['content'].shape)
print(df['output'].shape)

hate = 'hate'
Nhate = 'non-hate'

# 定义解析函数：分词+存储分词结果
def parse_output(output):
    quadruples = output.split('[SEP]')  # 使用字符串 '[SEP]' 作为分隔符
    results = []
    for quad in quadruples:
        quad = quad.strip()  # 去除前后空格
        if quad.endswith('[END]'):  # 检查是否以 [END] 结尾
            quad = quad[:-5].strip()  # 去除 [END]
        parts = quad.split('|')  # 使用 | 分割四元组
        if len(parts) == 4:  # 确保四元组有四个部分
            target, argument, group, hateful = [part.strip() for part in parts]
            results.append({
                'Target': target,
                'Argument': argument,
                'Targeted Group': group,
                'Hateful': hateful
            })
    return results

solved_output = list(map(parse_output, df['output']))
print(solved_output[0])

# 构建结构化数据集
def build_structured_data(df):
    structured_data = []
    for idx, row in df.iterrows():
        content = row['content']
        output = row['output']
        quadruples = parse_output(output)
        for quad in quadruples:
            structured_data.append({
                'id': row['id'],
                'content': content,
                'Target': quad['Target'],
                'Argument': quad['Argument'],
                'Targeted Group': quad['Targeted Group'],
                'Hateful': quad['Hateful']
            })
    return pd.DataFrame(structured_data)

# 示例
structured_df = build_structured_data(df)
print(structured_df.columns)

def is_hate(output_data):
    hate_count = 0
    Nhate_count = 0
    for hateful_value in output_data['Hateful']:  # 遍历 'Hateful' 列
        if hateful_value == 'hate':
            hate_count += 1
        elif hateful_value == 'non-hate':
            Nhate_count += 1
    return [hate_count, Nhate_count]

# 示例
print('hate_count:', is_hate(structured_df)[0])
print('non-hate_count:', is_hate(structured_df)[1])

#数据编码
label_encoder_group = LabelEncoder()
label_encoder_hateful = LabelEncoder()

structured_df['Targeted Group Encoded'] = label_encoder_group.fit_transform(structured_df['Targeted Group'])
structured_df['Hateful Encoded'] = label_encoder_hateful.fit_transform(structured_df['Hateful'])

print(structured_df[['Targeted Group', 'Targeted Group Encoded']].drop_duplicates())
print(structured_df[['Hateful', 'Hateful Encoded']].drop_duplicates())
