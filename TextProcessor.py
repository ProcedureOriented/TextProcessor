from audioop import mul
from difflib import diff_bytes
from heapq import merge
import os
import math
import datetime

import jieba
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, LSTMCell, LSTM, Bidirectional
from tensorflow.keras.layers import TextVectorization

def generate_word(series: pd.Series, clearfy=False, name=None):
    # 统计词频
    jieba.load_userdict('dict/userdict.txt')

    text = ' '.join(series.dropna().tolist())
    words = jieba.lcut(text)
    wordcount = pd.DataFrame(words,columns=['word'])
    wordcount['count'] = 1
    wordcount = wordcount.groupby('word').count()
    wordcount = wordcount.sort_values('count', ascending=False).reset_index()

    if name:
        wordcount = wordcount.rename(columns={'count': name})

    if clearfy:
        wordcount = wordcount[~wordcount['word'].str.contains(r'[\s]', regex=True)]    # 去掉空白字符
        wordcount = wordcount[~wordcount['word'].str.contains(r'[\d]', regex=True)]    # 去掉数字
        # wordcount = wordcount[~wordcount['word'].str.contains(r'[a-zA-Zａ-ｚＡ-Ｚ], regex=True')]    # 去掉字母
        wordcount = wordcount[~wordcount['word'].str.contains(r'^[！-～!-~]$', regex=True)]    # 去掉全半角，包括字母
        wordcount = wordcount[~wordcount['word'].str.contains(r'[\W]', regex=True)]    # 去掉特殊字符
    
    wordcount.reset_index(drop=True, inplace=True)
    return wordcount

def generate_dict(wordcount: pd.DataFrame, threshold: int=0, merge: bool=False, mergedict: pd.DataFrame=pd.DataFrame(), plot=True):
    # 生成字典
    dictionary = pd.DataFrame()
    dictionary['len']   = wordcount['word'].str.len()
    dictionary['count'] = wordcount.drop(columns=['word']).sum(axis=1)
    dictionary['word']  = wordcount['word']

    if merge & ~mergedict.empty:
        dictionary = pd.merge(dictionary, mergedict, on='word', how='outer')
        dictionary['count'] = dictionary[['count_x','count_y']].sum(axis=1)
        dictionary = dictionary.drop(columns=dictionary.filter(like='_', axis=1).columns)
        dictionary['len'] = dictionary['word'].str.len()
        dictionary = pd.concat([dictionary[['len','count','word']], dictionary.drop(columns=['word','len','count'])], axis=1)
        dictionary = dictionary.astype({'count': int})

    dictionary.sort_values(by='count', ascending=False, inplace=True)
    dictionary.reset_index(drop=True, inplace=True)

    print('dictionary size (counts >= %d):' %threshold, sum(dictionary['count']>=threshold))
    if plot:
        dictionary[dictionary['count']>=threshold]['count'].plot.hist(bins=30)

    return dictionary[dictionary['count']>=threshold]

def clean_freq(series: pd.Series, wordlist: list=None):
    # 去除词表里的频率词
    if wordlist:
        for word in wordlist:
            series = series.str.replace(r'%s' %word, '', regex=True)
    return series

def clean_text(series: pd.Series, wordlist: list=None, fastmode: bool=False):
    # 增强的清洗文本并分词
    series = series.str.replace('　', ' ')
    series = series.str.replace(r'\n', ' ', regex=True)
    series = clean_freq(series, wordlist)

    if fastmode:
        text = '\n'.join(series)
        text = ' '.join(jieba.lcut(text))
        text = text.replace(' \n ', '\n')
        newseries = pd.Series(text.split('\n'))
        if len(series) == len(newseries):
            series = newseries
            del newseries
        else:
            print('Not equal!')
            return newseries
    else:
        series = series.apply(lambda x: ' '.join(jieba.lcut(x)))

    series = series.str.replace(r'[a-zA-Zａ-ｚＡ-Ｚ.\d]+', ' ', regex=True)
    series = series.str.replace(r'[\d:：：,，。?？!！“”"()（）…、+_\-%％]', ' ', regex=True)
    # series = series.str.replace(r'^[:：：,，。?？!！]', '', regex=True)
    # series = series.str.replace(r'[:：：,，。?？!！]$', '', regex=True)
    series = series.str.replace(r'[ ]+', ' ', regex=True)
    series = series.str.replace(r'[ ]+$', '', regex=True)
    series = series.str.replace(r'^[ ]+', '', regex=True)
    return series

def pre_classify(text: pd.Series, dictionary: pd.DataFrame, feature: str, value: list, absolute=False):
    # 初步分类：词典
    result = pd.DataFrame(text)
    # split value and keywords
    for i in value:
        print('Processing:', i)
        wordlist = dictionary[dictionary[feature] == i]['word'].to_list()
        result[i] = 0
        for word in wordlist:
            result.loc[text.str.contains(word), i] += 1

    # calculate
    result[feature] = result[value].apply( lambda x: ''.join( [str(i) for i in x[x==x.max()].index] ), axis=1)
    if absolute:
        result['score'] = result[value].apply( lambda x: x.max() - x.sort_values().iloc[-2], axis=1)
    else:
        result['score'] = result[value].apply( lambda x: 1- (x.sort_values().iloc[-2])/x.max(), axis=1)
    # data['score'] = data[value].apply( lambda x: 1- (x.sort_values().iloc[-2])/x.mean() ) # for multiclassify

    return result

def plot_score(result: pd.DataFrame, feature: str, classes: list, wide: int):
    # 绘制分数分布
    long = math.ceil( len(classes)/wide )
    plt.figure(figsize=(wide*5, long*4))

    strtype = (result[feature].dtype == 'object')

    if result['score'].dtype == 'int64':
        for i, c in enumerate(classes):
            plt.subplot(long, wide, i+1)
            if strtype:
                bins = len(result[result[feature]==str(c)]['score'].unique())
                result[result[feature]==str(c)]['score'].plot.hist(bins=bins if bins <=30 else 30)
            else:
                bins = len(result[result[feature]==c]['score'].unique())
                result[result[feature]==c]['score'].plot.hist(bins=bins if bins <=30 else 30)
            plt.xlabel(c)
    else:
        for i, c in enumerate(classes):
            plt.subplot(long, wide, i+1)
            if strtype:
                result[result[feature]==str(c)]['score'].plot.hist(bins=20)
            else:
                result[result[feature]==c]['score'].plot.hist(bins=20)
            plt.xlabel(c)
    
    plt.tight_layout()

def sparse_class(series: pd.Series, order: list=None, classname: str=None):
    if not order:
        order = series.unique().tolist()
        order.sort()
    if not classname:
        classname = series.name

    if series.dtype == 'object':    # for multi-tags or single tag in string
        result = pd.concat([(series.str.contains(c)).rename(classname+str(c)) for c in order], axis=1)
    else:   # for single tag in numbers
        result = pd.concat([(series == c).rename(classname+'_'+str(c)) for c in order], axis=1)

    return result.astype('int')

def use_default_model(
    X: pd.Series,
    y: pd.Series,
    multi: bool = False,
    vocab_size: int = 10000,
    sequence_length: int = 20,
    embedding_dim: int = 32,
    biLSTM = False
    ):
    # 独热
    yname = y.name
    y: pd.DataFrame = sparse_class(y)

    # 词向量化
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
        )
    vectorize_layer.adapt(X.to_list())

    # 日志和时间戳
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(' ', '-')
    timestamp = timestamp.replace(':', '')
    timestamp = timestamp[:15]
    timestamp
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs_%s/%s' %(yname, timestamp))

    # 默认模型
    if biLSTM:
        model = Sequential([
            vectorize_layer,
            Embedding(vocab_size, embedding_dim, name="embedding"),
            # Bidirectional(LSTM(32, return_sequences=True)),
            Bidirectional(LSTM(16)),
            Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            Dense(y.shape[1])
        ])
    else:
        model = Sequential([
            vectorize_layer,
            Embedding(vocab_size, embedding_dim, name="embedding"),
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'),
            Dense(y.shape[1])
            ])

    # 编译
    model.compile(
        optimizer='adam',
        # optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )

    # 训练
    model.fit(
        X.sample(frac=1, random_state=666),
        y.sample(frac=1, random_state=666).to_numpy(),
        validation_split=0.3,
        # shuffle=True,
        epochs=5,
        callbacks=[tensorboard_callback]
        )

    return model

def classify(model, data: list, classes: list, feature_name: str,multi: bool=False):
    softmax_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    result = softmax_model.predict(data)
    result = pd.DataFrame(result, columns=classes)
    if multi or type(classes[0])==str:
        result[feature_name] = result[classes].apply( lambda x: ''.join( [str(i) for i in x[x==x.max()].index] ), axis=1)
    else:
        result[feature_name] = result[classes].apply( lambda x: x.idxmax(), axis=1)
    result['score'] = result[classes].max(axis=1)

    return result

def batch_classify(model, data: pd.Series, classes: list, feature_name: str, multi: bool=False, batch: int=100000):
    rlist = []
    for i in range( math.ceil(len(data)/batch) ):
        start = i*batch
        end = min((i+1)*batch, len(data))
        print('proccesing:', i)
        r = classify(model, data.iloc[start:end].to_list(), classes, feature_name, multi)
        rlist.append(r)
        
    result = pd.concat(rlist)
    result.index = data.index
    return result

def compare(left: pd.DataFrame, right: pd.DataFrame, feature: str):
    idx = (left[feature] != right[feature])
    difference = pd.concat([left[idx], right[idx]], axis=1)
    return difference, idx

def new_sample(data: pd.DataFrame, result: pd.DataFrame, feature: str, sample: int or float=1):
    output = pd.DataFrame()
    sortresult = result.sort_values('score', ascending=False)
    if 0< sample <= 1:
        for f in result[feature].unique():
            n = int(len(sortresult[sortresult[feature] == f]) * sample)
            output = pd.concat([output, sortresult[sortresult[feature] == f].iloc[:n]])
    elif sample>1:
        for f in result[feature].unique():
            output = pd.concat([output, sortresult[sortresult[feature] == f].iloc[:sample]])
    del sortresult
    output = pd.concat([data.iloc[output.index], output], axis=1)
    return output