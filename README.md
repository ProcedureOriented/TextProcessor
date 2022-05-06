# TextProcessor
基于半监督和循环增强的文本分类流程的NLP工具实现——工具部分  

## 函数说明

### generate_word
generate_word(series: pd.Series, clearfy=False, name=None)  
接受一个pandas.Series，返回全部词语的词频统计。clearfy清除字母和符号等，name为返回列重命名，可用于年度统计。  

### generate_dict
generate_dict(wordcount: pd.DataFrame, threshold: int=0, merge: bool=False, mergedict: pd.DataFrame=pd.DataFrame(), plot=True)  
接受一个pandas.DataFrame，返回包括词语，词长度和词频的DataFrame作为词典，以merge标记和mergedict接受可合并的原有词典。threshold为返回词语的频率阈值，plot为绘制词语分布的标记。  

### clean_freq
clean_freq(series: pd.Series, wordlist: list=None)  
接受pandas.Series和一个停用词列表，返回一个清除过停用词的Series。  

### clean_text
clean_text(series: pd.Series, wordlist: list=None, fastmode: bool=False)  
clean_freq的进一步封装，接受pandas.Series和一个停用词列表。返回清除停用词和特殊符号的分词后Series。fastmode使得处理稍快但可能不稳定。  

### pre_classify
pre_classify(text: pd.Series, dictionary: pd.DataFrame, feature: str, value: list, absolute=False)  
根据词频的初步分类。接受一个pandas.Series和标记后的词典DataFrame，feature指定分类维度，value指定维度中的类别，absolute=True时返回直接相减的评分。  

### plot_score
plot_score(result: pd.DataFrame, feature: str, classes: list, wide: int)
接受分类结果，维度名，维度类别和子图宽度，输出每一类别的数量分布。  

### sparse_class
sparse_class(series: pd.Series, order: list=None, classname: str=None)  
接受单列的分类结果pandas.Series，输出稀疏分类结果（独热变量），order指定顺序，classname为维度类别的前缀，默认使用Series.name+'_'。  

### use_default_model
use_default_model(X: pd.Series, y: pd.Series, multi: bool = False, vocab_size: int = 10000, sequence_length: int = 20, embedding_dim: int = 32, biLSTM = False)  
接受训练集的X（已分词）和y，词向量化，并编译模型，训练，返回训练后的模型。biLSTM=True使用带双向LTSM层的模型。  

### classify
classify(model, data: list, classes: list, feature_name: str,multi: bool=False)  
接受模型和要分类的数据列表、分类维度类别，维度名，返回分类结果。  
【一次接受数据太多会造成无响应，请使用batch_classify】  

### batch_classify
batch_classify(model, data: pd.Series, classes: list, feature_name: str, multi: bool=False, batch: int=100000)  
将classify传入的数据列表分批，此时需要传入Series以输出和传入相同索引的序列，batch指定单次分类的数据量，默认为100000。  

### compare
compare(left: pd.DataFrame, right: pd.DataFrame, feature: str)  
自检循环后使用，比较原结果和新结果，返回结果不同的行索引和合并的索引结果以查看或保存。  

### new_sample
new_sample(data: pd.DataFrame, result: pd.DataFrame, feature: str, sample: int or float=1)  
增量循环后使用，接受分类样本和结果，维度名称，结果大小，输出评分最高的部分。当sample在(0,1]时，返回每类别比例前sample * 100 %的结果，当sample大于1时返回每类别前sample数量的结果。  