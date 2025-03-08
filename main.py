import numpy as np
import pandas
import numpy
import jieba
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#path
data_path = './date'
train_file = './data/train.json'
test_file = './data/test1.json'

df_train = pd.read_json(train_file)
df_test = pd.read_json(test_file)


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

solved_train_output = list(map(parse_output, df_train['output']))


print(solved_train_output[0])

# 构建结构化数据集
def build_structured_data(df_train):
    structured_data = []
    for idx, row in df_train.iterrows():
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
train_structured_df = build_structured_data(df_train)

print(train_structured_df.columns)


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
print('hate_count:', is_hate(train_structured_df)[0])
print('non-hate_count:', is_hate(train_structured_df)[1])


#数据编码
label_encoder_group = LabelEncoder()
label_encoder_hateful = LabelEncoder()

train_structured_df['Targeted Group Encoded'] = label_encoder_group.fit_transform(train_structured_df['Targeted Group'])
train_structured_df['Hateful Encoded'] = label_encoder_hateful.fit_transform(train_structured_df['Hateful'])



print(train_structured_df[['Targeted Group', 'Targeted Group Encoded']].drop_duplicates())
print(train_structured_df[['Hateful', 'Hateful Encoded']].drop_duplicates())


print(train_structured_df.isnull().sum())
print(train_structured_df.duplicated().sum())

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import jieba

# 假设 structured_df 已经正确定义
# structured_df = ...

# 多标签编码
mlb = MultiLabelBinarizer()
targeted_group_encoded = mlb.fit_transform(train_structured_df['Targeted Group'].str.split(', '))
targeted_group_df = pd.DataFrame(targeted_group_encoded, columns=mlb.classes_)
train_structured_df = pd.concat([train_structured_df, targeted_group_df], axis=1)

# 分词
train_structured_df['content_tokenized'] = train_structured_df['content'].apply(lambda x: ' '.join(jieba.lcut(x)))
train_structured_df['Target_tokenized'] = train_structured_df['Target'].apply(lambda x: ' '.join(jieba.lcut(x)))
train_structured_df['Argument_tokenized'] = train_structured_df['Argument'].apply(lambda x: ' '.join(jieba.lcut(x)))


# 打印结果
print(train_structured_df.head())

