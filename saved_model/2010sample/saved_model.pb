??
?'?&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02unknown8??
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	?N*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
n

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name120617*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_116073*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	?N*
dtype0
?
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_16/kernel/m
?
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_17/kernel/m
?
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	?N*
dtype0
?
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_16/kernel/v
?
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_17/kernel/v
?
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_4Const*
_output_shapes	
:?N*
dtype0*??
value??B???NB增长B业绩B年B被B月B日B评级B的B股B将B涨停B提示B亿B公司B最新B股东B净利B高B快讯B推荐B遭B股权B大B提升B或B买入B不B增持B同比B预期B减持B行业B股份B逾B亿元B	净利润B中国B盈利B中B发展B投资B有望B万B倍B大幅B基金B沪B增B板块B上市B资产B项目B强势B股价B停牌B受益B在B指B新B地产B亏损B再B维持B与B涉嫌B因B重组B机构B超B转让B调查B震荡B下跌B拟B元B业务B强烈推荐B和B期货B风险B违规B是B万元B继续B产品B预测B近B源B	上半年B	董事长B未来B门B增发B产能B有B可B称B图B利好B能力B大盘B为B空间B企业B关注B汽车B仍B价格B资金B持续B	证监会B刺激B万股B收购B套现B反弹B控股B成B调整B预增B快速B	子公司B点B证券B利润B政策B跌停B预计B新股B大涨B重大B期B上海B	地产股B注入B污染B大跌B扩张B看好B大增B上涨B跌B收入B影响B	一季度B国际B后B面临B产业B一B未B成长B等B暴跌B估值B受B质疑B内幕交易B交易B三B获B房价B融资B领涨B事件B多B股市B销售B拉升B遭遇B楼市B只B前B诉讼B加速B问题B人B连续B涨幅B波动B期待B对B快速增长B已B】B【B股票B今日B事项B向B	三季度B行情B地B推动B进入B手机B今年B走强B调控B经济B纠纷B涨B领跌B异常B两B爆发B煤炭B了B概念B国内B迎来B及B下降B科技B个B放量B立案B高管B降价B短期B银行B股指B午后B事故B两市B释放B	房地产B	停复牌B需求B发行B点评B	毛利率B符合B回落B回升B谁B板B公开B至B低B整合B前景B节能B带来B主业B战略B全年B给予B个股B下B压力B值得B优势B启动B我B建设B价值B走势B实现B稳定增长B医药B跳水B	概念股B于B新高B复牌B布局B上B上调B抛售B再度B深B明显B报B日起B处罚B起B品牌B公布B最B销量B融合B家B升级B上升B筹划B集体B土地B长期B机会B三大B较B经营B电网B计划B强劲B地块B重挫B造假B稀土B年报B激励B首次B来源B智能B小幅B做B披露B关联B好B损失B拖累B欲B技术B油价B达B下载B提高B退市B指数B延续B去年B增速B冲击B引发B资源B访问B王B景气B新疆B带动B复苏B三网B走低B游资B	总经理B良好B白酒B电子B原油B顶B消费B助推B方案B增加B后市B部分B跌破B走B案B通过B否认B陷B转型B B五成B如何B可能B出口B谨慎B订单B罚B收益B打造B券商B	目标价B居首B金融B董事B变更B全面B稳定B整体B政府B早盘B好文B免费B拿B冲B控制B	增长点B危机B难B	亿美元B	王亚伟B引爆B募资B其他B主力B下滑B飙升B规划B平稳B增强B合同B又B到B	黄光裕B苏宁B明年B信披B	下半年B非B加息B—B策略B保持B高端B国美B区域B创B传闻B之B驱动B背后B暴涨B二B盘中B海外B江苏B封B即B明B季报B因素B房企B回应B	发改委B不足B公司业绩B竞得B专家B预亏B涨价B	成品油B全线B要B融券B突破B稳健B逆势B违法B化工B纺织B私募B电器B巨大B即将B核查B加快B整改B基地B降B转B电力B支撑B动力B债务B	前三季B降低B成本B异动B半年B首日B茅台B冲高B	人民币B业B辞职B数据B四B巨亏B小B下挫B担保B下调B虚假B点睛B尾盘B从B联网B看B会B研报B严重B《B潜力B资产重组B补贴B建议B参股B领域B赔偿B火眼B曝B》B	证监局B破发B无B利空B出现B出售B趋势B药业B泄密B料B市B高位B通胀B改善B买B天B致B来B表现B短线B目标B	新一轮B绿B显著B天津B低迷B钢价B进军B行B疯狂B最大B	开发商B原B	进一步B商品B南京B低开B	铁矿石B投B恒指B年内B深陷B快B央企B增厚B重点B价B责令B记者B罚款B矿业B提价B买卖B稳步B成为B低于B中海B世博B网络B	权重股B推进B抄底B四成B规模B存在B刑拘B量B逆市B进行B美国B物B油B半日B千万B今B也B开发B	产业链B炒作B年度B制造B信托B解禁B水泥B有色B昨B明日B操作B揭秘B逼近B送B约B建B定向B长虹B设备B视频B	敢死队B三成B大地B变B午评B内B金价B操纵B并购B国家B警示B平台B山东B上市公司B风电B	竞争力B确定B卖B分析B回调B不会B上车B追捧B试点B系B最高B农行B以上B超过B翻番B暂停B	天然气B商业B为何B一个B言论B大宗B现B新低B广东B就B起诉B广阔B全国B财务B炒B弱势B签B浙江B导致B占B停产B营销B打开B两大B都B水平B明确B	投资者B投资收益B成立B出台B遭大B营收B电信B扩大B一季B难题B港股B核心B新兴产业B拓展B传B服务B实施B以B今起B移动B申请B开始B三年B高铁B赚B警惕B澄清B家电B实力B四大B主要B唐骏B减排B万吨B效应B处理B召开B六成B较大B购B荐B模式B季度B光伏B二次B索赔B提供B实际B南航B需B航运B组图B是否B情况B借B中航B世界B贷款B让B要求B获批B监管B有色金属B旅游B提前B取消B发布B	十二五B结构B竞争B溃坝B	深交所B正B完成B分红B期指B推B促B限售B评论B走高B信息B介入B获利B柴油B抛B定价B媒体B保障B两年B上扬B重庆B还B营业B积极B相关B活跃B昨日B提速B并B退出B美元B承诺B快评B存B原因B分B什么B	二季度B高开B近期B生产B河北B均B双重B历史B募投B软件B谴责B罪B福建B看点B电视B牛股B总裁B助B仅B云南B香港B行政处罚B月份B打压B小时B客车B升值B全球B钢材B质量B派B撤销B接盘B房产B低估B首季B通报B股民B特别B新增B实业B出逃B自主B美B炒股B曝光B成交B多重B多家B	中小板B下周B进口B环境B机遇B十大B借壳B亮点B产量B两倍B一度B附股B步入B本周B曾B工作B地方B发力B助力B两成B七成B更B拐点B总B逢B路B环比B显现B时B持有B律师B引领B外资B员工B发生B再次B上行B董秘B能B得B出B再现B五大B高新B领先B	第三季B格局B中心B一倍B陈述B调研B破B巨头B不断B	不合格B阶段B认定B有限B库存B	四季度B加剧B进展B拟定B外延B制药B内幕B举行B	触摸屏B神秘B泡沫B态势B应B	副总裁B一年B鸿基B调B联信B空调B消息B时间B	强势股B市值B依然B举报B东航B飙涨B远洋B过B成功B召回B零售B金B迅速B观望B翻倍B永益B	成长性B制度B低价B但B亿股B下行B采购B逐步B还是B窄幅B破产B看盘B率先B强B对外B	增长期B增幅B回归B南车B铁路B通缉B超级B而B纸业B涉B恢复B你B渠道B地震B加大B份额B临时B金属B很B安全B	国资委B团队B商业地产B冻结B优质B考验B经理B现货B独立B我国B客户B周期B名B协议B关于B令B	谢亚龙B能否B申银万国B爆炒B斥资B工业B基本B区间B出炉B八成B陷入B	银广夏B调价B给B稳步增长B直线B权证B四川B造纸B路演B贡献B浮亏B比B散户B收回B拘B央视B品种B千亿B再创B中铝B迎B行贿B百亿B爆炸B支付B提振B拍卖B	多元化B均线B回暖B去B高送B	锂电池B重卡B通信B蹊跷B能源B累计B第二B煤价B持续增长B招股B引B女B地位B发电B酒B置业B网友B结束B煤矿B没B汀江B度B官员B军工B人事变动B顺利B隐忧B研发B应对B年底B定增B升B	再融资B价格上涨B酝酿B通道B理财B烟台B潮B污水B式B增资B合并B合作B出货B中兴B路翔B说明B管理B盘点B疫苗B每日B正在B次B支持B所B市长B	小盘股B审慎B大单B处于B后续B受累B决定B传媒B上攻B一周B门槛B酒店B股本B结果B氨纶B毛利B新政B扭亏B崛起B好转B壹桥B商场B十年B北B保险B争夺B链B车B赴B财经B线B每B欧洲B楼盘B暴增B日线B收盘B巨额B吨B号B配股B苗业B短B热点B烟花B火线B油田B没有B底部B利益B重创B车企B身陷B责任B警方B解读B装备B	董事会B杀跌B文化B拉动B投产B印度B	任志强B降温B获得B紧急B系统B等待B用B瓶颈B每股B挂牌B强烈B啤酒B只股B化B分歧B具有B井喷B亏B轨道B被疑B落户B苹果B直接B盘B正式B条件B权重B机制B掀B抢筹B打B	国务院B出手B亿购B下乡B食品B集运B销售收入B说B股改B翻红B渗漏B持股B打击B当B国企B变脸B发展前景B	半年报B中远B限产B铜B重要B航线B王石B港元B永磁B来临B报道B实B央行B味精B只是B双雄B	分析师B	价齐升B之争B万辆B转向B质押B被查B火灾B步伐B新材B居前B商标B却B	加油站B内部B供应B传统B三个B一线B部门B设计B股东大会B物流B	活跃股B有限公司B扩产B房B吗B合肥B刚玉B具备B信心B	互联网B风波B钱B转换B荣信B煤B新闻B展开B失守B分拆B光B仓B中报B下游B三线B隐瞒B部B连B车市B生物B港B死B成员B形成B寻找B副B分化B优化B九成B举牌B中性B一字B闲置B锂电B蓝筹B老鼠B网上B票据B盘整B玉米B沙特B新品B推出B挑战B巨量B受贿B发B博弈B创投B分享B其B会议B	举报人B主营B	万美元B	预增超B钢厂B酿酒B选择B运行B方向B定位B失踪B合理B双轮B出资B企稳B亚运B	世博会B设B被捕B荒B第B社会B电脑B海西B死亡B旺季B应用B周三B向上B卷入B为主B金股B重整B追踪B进驻B超跌B赚钱B解决B网B用户B潜伏B木业B最后B拉B才B恐慌B待B弹性B周五B周B双双B凸显B促进B主营业务B三只B·B青岛B钢企B	融资券B老总B简介B疑B由B港口B收涨B攀升B套牢B塑料B均价B地王B国资B	反倾销B剑B减少B兑现B高价B飘红B造成B老B置B综合B类B	管理层B目前B	电动车B整理B或成B延期B升温B出让B六B风格B	零部件B重罚B遇B费用B	责任人B缩水B猜想B狂B	爆发式B溢价B	消费者B民营B比例B战B建材B基础B城建B垄断B储备B信号B低位B企B仓位B青睐B身B联手B者B突然B真相B	深国商B泄漏B河南B投入B急速B强封B开拓B	基本面B叫停B华润B分配B	分公司B减产B关口B五B中药B	中海油B	两高管B不能B靠B静待B钢B道B轮胎B走弱B航班B置换B盛宴B百元B电池B灾区B潜在B架B捐款B悬念B广钢B	工信部B宣传B安徽B块B困境B力度B全部B依旧B不改B三季B锁定B钾肥B酒类B辰州B触及B药品B续B红包B粤B管道B欧元B最受B旗下B放缓B急B双B保持稳定B低碳B人气B产品价格B予B中银国际B两个B集中B重仓B走向B豪赌B被套B良机B终止B精机B离职B社保B着B电气B现涨B玉树B汽油B望B有所B普遍B撤离B探底B拒绝B投建B或超B成就B已经B十B内容B乐观B主席B	金融股B违反B运输B运价B负债B观察B纪要B禾B瞒报B百货B版B涉及B日本B效果B捧B扎堆B开B封死B家族B宁波B加仓B分钟B何时B人员B产销B	交易所B齐B重工B进B聚焦B缘何B砸盘B百万B用地B现身B	次新股B查处B来自B春节B方式B改革B换B拆迁B想象B总局B巨资B定B如期B大成B地铁B	因涉嫌B	关联方B公司股票B保证B中央B食品饮料B逆转B连塑B购买B西安B股派B终端B细菌B	盛新材B监测B涨势B棉花B栋梁B方源B提B抛弃B	巴菲特B差钱B岁B完善B契机B大户B友谊B华富B利润分配B出任B冰箱B上演B三洋B三月B铜业B通知B适当B转增B股领B结构调整B细分B突出B	确定性B破解B甩卖B曲美B旺盛B放弃B改制B拍B开通B左右B富豪B家纺B客B审批B安排B学历B妖股B大肆B大佬B多头B反B	利润率B公路B传言B亿城B	二手房B专访B上游B限购B铝B退税B边际B	经销商B精选B神话B期间B改正B扫描B带走B官方B国B	商用车B周四B周一B发现B参与B印刷B公募B充分B信宜B人士B乳业B中山B两地B一天B鲁信B频现B锂B迷雾B连阳B跑B资格B试水B证实B认为B系列B种B男子B热B火电B激活B汽车行业B欺诈B指标B把B扶持B张B城B	地产商B卖出B加工B副总B再陷B内地B六大B光电B介绍B九龙B中投B两难B两会B三倍B万亿B限价B附近B跨越B走出B该B行为B自己B胶B联盟B考虑B维权B确认B现金B温州B派息B治理B款B案件B接受B惧B	快车道B市民B存疑B失败B大举B多数B增近B商B合资B叶檀B反复B加强B催生B促销B会否B业内B铜矿B金顶B配置B这B补助B疑似B电价B演绎B游戏B海参B浮出B注资B没收B江泥B新兴B推迟B接手B投向B批B扭亏为盈B我们B性B待解B广永B幕后B天润B处B售价B周报B受惠B入驻B住宅B亿建B	上交所B	银监会B重重B配售B转移B跟随B起飞B规范B补涨B蒸发B落空B联想B红B确保B电力行业B王者B	潜规则B法院B核准B机电B更名B曲线B携手B接任B持B开放B建仓B延伸B岳阳B局长B宽幅B实行B套利B	商务部B周二B后期B吃B厂商B医药行业B医疗B利益输送B减仓B内参B优惠B伊春B今市B人事B云B二线B丰田B专业B金利B重回B超募B贵州B规避B第三季度B破位B物价B源车B材B服装B暂B	新大新B斥B接近B押宝B披B投机B强化B引擎B幅度B审核B好于B天价B	多晶硅B困局B合约B	可转债B受理B区B北汽B加入B净B冠军B再起B借款B借力B保山B	交易日B之后B东铝B题材B难逃B锌B遭二B购房B见B	螺纹钢B节后B深度B淡季B民航B	母公司B李B有待B搬迁B搁浅B推手B指跌B护盘B	房产税B忙B形势B大火B古汉B半月B到来B准备B内生B全B信贷B住B亿吨B京B互动B争议B两天B	世界杯B上药B	一体化B青海B陈晓B问责B重现B通州B迹象B轿车B货币B被迫B被否B	葡萄酒B航母B腐败B综述B细则B纷纷B窗口B空头B砸B研究B略超B电子商务B湘财B深航B淘汰B流通B水电B民企B有关B显B数字B改B操盘B拼入B存款B大豆B地区B在线B同行B台湾B另类B受阻B	发动机B	出厂价B众B	企业家B享受B中邮B不明B高层B隐现B过剩B谈判B询价B	蓝筹股B考察B罗莱家纺B缩量B组合B第三B空难B私人B离场B皮B登场B燃气B	煤化工B激增B漩涡B滑坡B海鸟B海上B暴利B晨会B显示B故事B携B投诉B执行B打折B我市B彩电B巴西B对话B密集B圈地B国酒B回B厂B单位B包装B	制造业B到底B农药B入市B	亿港元B主线B三花B黑马B高盛B需要B限电B门店B	销售额B铜价B重B还有B转变B车展B	路线图B超市B货币政策B诚信B计算B	范德均B航天B自B罚单B稳B	电子书B激情B次日B梦想B有效B最快B掏空B批准B扩容B意外B情B建成B广告B尚未B将成B富华B官司B外围B型B圈钱B囤地B吸B反转B反击B华芳B华神B华为B	千万元B	刘益谦B内需B八大B	产业化B主题B中线B专项B丑闻B不佳B下线B上众B领导B靓丽B重启B账户B被判B蓄势待发B落幕B苏州B竞拍B监事B疲软B生物制药B特色B物业B	爆发性B煤炭行业B焦炭B点名B湘鄂B	流动性B活动B油气B杀入B星马B无碍B方B收跌B振兴B情绪B带B已成B家电行业B奶粉B垃圾B国产B	商品房B变化B变动B受到B原料B医改B力诺B兼并B兰州B全天B保B	使用权B使用B体现B仰融B他B亚洲B二期B书记B乏力B乌龙B两日B专利B一汽B一日B鳄B领跑B陕西B铺路B金德B重估B进程B购物B谷歌B谨防B解密B虚增B薪酬B药B至少B	结构性B紧缩B竞购B省B电煤B电厂B理性B玄机B独家B爱B照片B添B涨超B洽购B汇率B欧盟B欧B榜B条B月底B昆明B推升B接替B排放B彰显B平均B	市盈率B居民B奠定B天利B坚决B四只B商城B哪些B变身B发行股票B华B力挺B农村B内外B修复B侵权B佳豪B休市B二代B乱象B主任B两次B不止B不是B上证B七大B一致B限制B郑B	迪士尼B轮B跌近B资金占用B谈B计提B蓄势B落后B	营业部B	茅台酒B	胡锦涛B网站B网游B缺乏B简称B简报B科冕B福州B疑为B生物医药B煤企B海洲B	洗衣机B注意B泄露B水面B早报B无法B排名B拟发B担忧B投行B扩建B房贷B	战略性B意见B	恢复性B忧虑B循环B年产B带头B展B实质B存货B太阳B天威B处置B困扰B呈现B各B南通B单季B	半年线B前夜B到期B俄B	供应商B二度B	中长期B上任B齐鲁B高档B重磅B里B酒窖B身家B赛迪B贸易B贵B财政B评选B警告B补B英华B美股B缺陷B缺口B结算B组织B精诚B类股B秘密B离奇B监管部门B电动B狂飙B	潜力股B消防B海尔B波音B概率B	李荣融B	本地股B最牛B暴露B暂时B早评B惹B开建B店B广西B年期B干部B带活B	富豪榜B宝龙B姚明B增值B坐庄B国土B售B周末B同业B卷土重来B升势B副总经理B出厂价格B	内蒙古B关键B关系B公关B先行B像B便宜B使B京东B五年B主流B两月B两任B专注B不可B须B隐患B险资B降幅B陆续B阿胶B锦化B钢圈B部委B遭罚B选股B连锁B转战B身价B豪宅B评估B解除B西北B补偿B虎年B若B自家B美利纸业B终结B碳B石B	知情人B真正B百点B百年B登陆B炼油B棉价B案例B架空B暴雨B易主B早B整固B摆脱B措施B招行B成熟B怎么B必须B	异动股B开市B建立B年线B平板B工行B山寨B属实B封杀B定局B	大学生B夏新B坚持B坚守B国阳B	嘉士伯B合成B变数B协会B出局B兴民B假B修正B位B优秀B以来B	乘用车B主导B中签B严打B不再B＋B鼎工B首份B领衔B领军B零B难改B铁矿B配套B遇阻B	运营商B运作B较强B越B资源整合B贱卖B财险B财报B	负责人B谋B表示B行权B行政B节前B组建B纯B站B	窗口期B突击B程序B磁材B	监事会B生活B爆B	深成指B波及B沪市B江B民间B板指B杀B本B改造B指收B报价B抢B承认B慈善B急跌B庄家B巩固B岘B局面B尴尬B富B威胁B大摩B大关B备受B吸引B合法B	发展期B及其B占用B加码B力拓B初现B几何B决战B再冲B免税B债券B	俱乐部B体系B	价格战B	争夺战B买家B主因B	为什么B中服B个人B不清B不愿B下属B黑洞B首B风暴B面积B霸王B雪花B闽B错B重装B重拳B配合B遏制B速度B辆B身份B跳B超出B资料B论坛B要闻B补跌B自救B级B竣工B税收B矿B知名B现象B现实B独董B炼厂B清理B深泰B涨跌B	涨停板B涌入B	消费税B海盗B止损B旱灾B效益B收市B投票B托管B微幅B年薪B差B局B富龙B	富士康B媒体报道B太B天下B大型B复合B	增长率B囤B啤B商务B呼吁B呈B吸纳B台B	可能性B变相B变局B反攻B卡B动B力B利率B共同B入股B	保证金B	俄罗斯B住房B亮相B五一B买房B中卫B严厉B两桶B两家B东碳B下探B上报B上周B三板B一股B鲁丰B首发B预售B闪电B长远B金矿B金汇B部署B道歉B造船B远B过度B资讯B货运B解释B被告B腾讯B胜B职工B职务B考试B绿豆B签下B秦川发展B种子B眼看B真B电动汽车B	物业税B照明B火箭B温和B清晰B涉案B济柴B油运B每吨B梦B桑德B村民B时日B日评B	方舟子B新建B斯B敲定B排除B拒B折射B承压B或现B意在B开征B开工B废水B年末B工厂B属B展望B层B小盘B导演B宗B奔B	太阳鸟B天发B大陆B大学B大会B图文B国庆B回顾B告别B参加B势头B前夕B公B僵局B	催化剂B储量B	信用卡B	供应链B传奇B会见B亦B	下周一B上马B三次B黑色B高潮B骗B驶入B首选B频频B难掩B长沙B钢材价格B金额B重申B部长B逼B逐渐B退B迅猛B输血B轴承B跳空B跌势B走进B证监B认购B认证B表态B蓉B落马B芯片B船B腾飞B胡润B聚友B细节B糖价B米克B禾欣B福特B现场B环境污染B犹存B特征B火速B火爆B澳B湖B民族B欣网B松动B松下B权益B期价B	有利于B最具B暴富B晶源B日报B日发B施工B方面B方兴B收B揭开B揭B接力B挺进B	持续性B折B抗B抓B技巧B	成长期B成商B惨遭B徘徊B建厂B延长B广场B	己二酸B工具B	工业园B峰会B就是B尚B少B小非B导航B完美B威远B套B天气B	大规模B	大盘股B	大幅度B复制B	国际化B唯一B哪里B	吞吐量B各地B变频B发射B反对B华晨B半B化肥B划转B八B入B健康B做大B仓库B人为B京沪B产权B	九龙山B临近B丰富B中华B东方B下架B上网B上书B三联B一起B一成B黑B鲸吞B风B频繁B面对B除权B银泰B重返B迷局B转债B身亡B超微B起火B负责B请B	话语权B记B解析B西单商场B西B被诉B蔬菜B董B花旗B统一B纳入B粮油B筹码B	第三方B移送B科龙B码头B监控B生死B生存B	现金流B	燃料油B	煤气化B热钱B热情B烟台万华B洗牌B泥潭B比较B正常B检查B查B构筑B末日B有意B明朗B昆明制药B新民B效率B收低B接连B持仓B拉开B投放B抑制B把握B惊天B惊喜B微涨B当地B张铜B应该B年会B平米B干旱B山体B履行B大通B大福B大力B备战B壳牌B国外B嘉宝B哈飞B咨询B吉利B史上B可期B受罚B参观B去向B厂房B卡车B卖场B卖地B募集B动漫B前任B出击B公诉B	公务员B停B促成B供货B何去何从B产B乘客B久B中兵B；B	高价股B飞B频发B预示B	零售业B长航B长B铝价B重视B重新B透露B连环B轮番B车型B路径B跟风B赔B资质B	贸易商B负面B诸多B误读B绿城B绝地B经编B纠结B管B筹资B究竟B税B租房B矿山B目录B电机B率B状告B版权B激烈B源于B游艇B流通股东B流出B法B江西B橡胶B机床B本土B	时间表B时期B无效B整车B	收益率B推高B找B成新B惊现B恐B忽悠B建行B底B常态B带领B希腊B已现B尾声B尚待B	家乐福B宣判B客机B宏观B安信B宁夏B天量B天期B大降B大战B外汇B外B基建B坚挺B国腾B回报B四维B四宗B和解B吸金B各方B古井B受伤B双鹤B双方B危险B华丽B医院B勒令B	加拿大B准备金率B净赚B关闭B入场B先B	元器件B元凶B做空B偏转B保护B供求B作为B佛山B低吸B任B件B亿阳B亿拍B人物B	中诚信B中称B两周B万金B万台B一触即发B麦加B马云B首个B风光B颓势B	领头羊B项B非法B	集装箱B难度B陈B钛白B重心B重压B避免B运营B运力B迈进B	过山车B软B车祸B跟踪B走软B赌B	财政部B证据B讨论B触底B解决方案B视点B西飞B获准B股基B翻绿B绝对B紧张B精彩B精品B精化B竞相B立案侦查B确立B盛大B盘初B疑问B理财产品B理想B珠海B玻纤B玻璃B狂欢B牛市B牙膏B牌B燃料B澳洲B	潘石屹B游客B渐显B渐B清洁B液晶B济钢B洛玻B注册B法规B油脂B	汽柴油B民事B欧债B核能B标的B	杜双华B村B	李东生B机组B机票B暗战B暂缓B时机B无忧B无奈B无力B撞B换帅B指涨B抱团B或是B想B悬疑B思路B得到B开启B延误B广夏B年前B工作人员B山钢B	封口费B宽带B委员B	大面积B大象B	大牛股B大旗B多方B声学B基层B坚称B地价B地下B团购B同时B叶片B发飙B	发电量B双钱B	原料药B单B千元B包B勘探B别墅B初B刚B	出租车B冯仑B内销B典型B具B公里B	信息化B作用B伪造B产销两旺B交银B交投B主导产品B中标B中捷B中小B两项B世纪B三天B万购B一级B一号B黄海B魅力B高新技术B驳回B首度B馆B难以B随B钢构B	金九银B配B郑糖B逼宫B造B追究B追加B远东B迎接B转身B	车流量B跻身B超额B	购房者B购得B调查结果B请辞B详解B诉B蔓延B落实B获益B药物B背景B股神B老大B置入B突袭B禁入B研判B矛盾B瞬间B直指B盲目B百事B白糖B疯B疑云B电子信息B用于B理工B焦煤B淘金B海虹B油品B欺骗B横盘B查明B期权B有利B更新B星辉B早间B无线B整治B	散货船B改变B收高B收到B撬动B推广B	控制权B授信B换手B拥有B担心B押注B	抗生素B承担B托B打破B所致B必读B微软B当前B并举B工人B寻底B实录B孙B委托B夸大B外贸B复兴B培育B国栋B围绕B回购B啥B	哈尔滨B周边B司法B只板B受压B发债B压制B占款B卖矿B单边B半数B午间B化工行业B劲增B前期B初步B刘B出来B农林牧渔B入主B债基B倒挂B倒B信B何方B们B	亚运会B争相B九九B中小企业B东源B业内人士B上诉B	上下游B三菱B万套B一流B一审B一定B＊B鼓励B黑莓B魔咒B高效B饲料B餐盒B风机B面板B隐情B降级B阻力B防B阅读B长阳B	铁公鸡B重实B配方B遇难B逐季B连高B违纪B迈入B辉立B车模B跨国B趋紧B豁免B议案B规则B蓝星B英国B芜湖B自查B自有B胶业B纯利B稽查B积极关注B秒杀B种业B破灭B矿区B瞄准B看空B相对B申万B现跌B现价B猜测B	炒房客B炒房B澳大利亚B渐近B清算B深入B涉足B消化B海盛B浮现B济南B流失B流入B法律B江铜B欧美B查询B未获B有人B替代B景区B明天B旭飞B无人B新区B	收获期B摘帽B掀起B换血B拟建B招保B抛盘B抓住B技改B扩充B打新B手B意愿B总价B应声B庄B将建B宣布B审计B安源B夫妇B大量B大赛B大蒜B大考B	大手笔B大幕B大幅提高B处分B国有B品质B周刊B吸筹B可观B变盘B受挫B反思B南海发展B卖股B华星B华孚色纺B华业B力保B割肉B农民B	内生性B停职B供需B何以B伤B亿万富翁B人民日报B二号B事故责任B乐B主板B主动B主B中行B中源B	中国式B东海B一飞冲天B一类B一只B	一个月B－B黑豹B高调B高点B高于B首都B飞行B预B陷阱B	阶段性B阵营B阴霾B阴影B	重仓股B造富B追讨B连豆B这些B达标B辽宁B辟谣B输B车间B趋于B	资源税B诡异B诈骗B藏B营运B节奏B艰难B联华B联动B绑架B线螺B紫砂B米B箱体B管制B笔B空壳B空中B短线交易B白电B登记B略B电话B电源B甩B生猪B甘肃B理由B煤电B	煤层气B澄B清洗B液晶面板B涨近B消除B消失B浮盈B流产B洋B泉州B	水污染B水B权B暗示B晨报B春B旱情B	日复牌B无量B方法B	新机遇B放B攻略B收紧B撤退B控盘B掘金B抢占B折戟B扫货B扛B所得B房市B房地产股B必B心理B很大B征税B引资B异军突起B年来B帝国B岛B山下B尽快B宏源B完毕B字B如B夯实B	天津市B	大事记B多项B多空B多名B	增值税B境外B国都B	国家级B国家电网B困难B回吐B四月B嘉联B商用B商家B同步B变成B	受益者B原油价格B十月B化解B加油B剥离B制约B制剂B创造B创近B刑事拘留B出去B兼B养殖B共B公益B公众B全日B免征B依托B低调B低点B	会计师B以旧换新B代表B仍然B	亿拿地B人保B产品销售B亏本B二八B主权B中票B两股B两只B	世界级B不超B三股B三农B一次B一样B一人B高达B高度B高峰B	题材股B频出B	预增近B难免B	闽福发B闯关B遭受B道路B逼空B通报批评B适合B追B近东B过半B	边缘化B轻卡B躁动B	跨越式B起航B质检B	豆浆机B谋求B评B设立B论B西湖B装置B表B螺纹B范围B花儿B节日B艘B	股逆市B股将B联袂B联众B翻身B美女B罢免B维稳B经典B粮食B	第三次B站上B突发B穿B空白B禾嘉B	研讨会B盘前B百姓B疲弱B瑞银B玉源B狂买B牵出B濒临B深水B海螺B海地B沦为B沉重B汽车销量B江湖B毒B每年B每天B榆林B楼面B构建B李锂B	李启红B期钢B日程B旅客B文件B数量B救B收官B摩根B揭牌B提醒B掌控B	换手率B持平B担B承接B戴帽B成渝B悲观B恒丰B总部B总体B怎样B微跌B微利B开花B帝B希望B	家央企B宝石B始末B妻子B如此B奠基B大庆B大于B多个B外盘B增收B增大B基B图谋B	国内外B回避B四个B喝B否决B向下B同B可逢B取得B发威B厚积薄发B	占有率B	博览会B协调B包头B勿B劲升B加盟B	副局长B制B到位B别B冲刺B关B全力B免职B	先行者B借机B体验B体B优先B代理B仍存B	亿投建B亿投B亮丽B二局B	事务所B中银B中润B个人简历B东盛B东光B	上杭县B三鹿B三类B三普B七B一种B一月B一批B一场B首旅股份B首批B	领导者B面B震动B雏形B集结B隐藏B陈家B长线B长期投资B长假B	长三角B钨B量价B郑州B避险B通车B近日B	近三成B辽通B轻轨B轻B车流B跟B购物中心B财务指标B贡B谋划B触B观点B见证B西工B被动B行踪B蜕变B蛋糕B蒙牛B	董事局B落定B落子B萎缩B获有B药企B	股派现B绯闻B结盟B结合B组团B纪录B紫B紧盯B	签合同B答疑B竞标B税率B秀B	福建省B破坏B矿产B看淡B看待B	百亿元B略有B甲醇B生变B燃B漏洞B混战B	消息面B测试B	流通股B洪灾B津城B波段B油荒B	沪综指B	沃尔沃B水价B	民航局B比重B比肩B正面B标杆B	柴油机B查封B林忠B林业B来袭B机B	月销量B最贵B最差B暨B星河B昆B日至B日照B无关B新盘B故障B收窄B搞B揽B探秘B掌门B挤压B挖掘B按B指南B拟以B护航B折价B承建B批捕B扑朔迷离B房屋B	戴姆勒B战争B或涉B成谜B悄然B急涨B快乐B必将B强大B开采B开局B延边B平衡B席位B已致B屡B尚需B寻B对手B密切B宽松B	实质性B宏观调控B宏源证券B安防B学习B	季节性B天伦B大额B	大输液B大笔B大戏B大厦B大势B多元B处理结果B声B坚毅B商贸B周评B吐B	后遗症B后劲B可望B可控B受宠B发文B发掘B发售B反应B双沟B厅B卸任B南B卓越B协查B华锐B千年B十足B十一B	制造商B判断B	创始人B分别B几乎B典范B其中B公寓B公交B倍增B保密B保定B	低成本B传说B亿买B人数B亲属B产生B亚吉B事实B书B之谜B中秋B中汇B不要B不变B上杭B上方B万手B齐聚B黑幕B鲁润B	食用油B食用B风云B领B顶级B韩B	零售价B集资B集合B隐形B险企B降至B防范B防御B链条B	银行业B重生B重击B	郭京毅B邯宝B追涨B连跌B进度B进场B这么B近百B迈向B过程B过后B达到B边B辞任B转股B转盈B	资金面B购车B购地B购入B订购B警察B解百B解冻B见底B装修B蓝海B	获利盘B药价B苗圩B船员B腾挪B股票交易B股早B股受B	股东会B老人B缺失B缓慢B维修B经验B绍兴B简单B筹B筑底B福建南纸B神光B短板B	短期内B看多B瘦身B	略低于B留意B电表B电力设备B	生物质B	生产线B瑞达B王勇B猫腻B猛增B狙击B状况B	特莱斯B牵连B爱心B煲B点燃B炉料B潜入B渐成B洗盘B法律法规B法师B沿海B汇众B汇丰B水落石出B水务B氟B比赛B死鱼B歌B欠款B楼B架构B李莉B	李毅中B本月B期市B	服务商B	有助于B月末B更换B暗藏B普跌B日钢B新规B整顿B放大B接班B接入B挺B拟向B招B拍出B抽检B抽B扩B	成交额B憧憬B悲剧B急挫B急拉B思考B忧B德国B归来B弱市B引起B引导B引入B底气B庆丰B平淡B	差异化B尿素B	小家电B将发B对决B富人B家央B客运B官B完工B娱乐B女子B女人B套保B天一B	大部分B	大起底B大盘走势B大气B壳B基指B埋单B圣元B围城B喜人B哪般B哪B命运B呼之欲出B启示B台州B可怕B	发起人B参展B厂家B博盈B协B华南B动摇B加B前瞻B制冷B利源B创出B列为B几个B军品B再添B兼备B共享B充足B	充电站B停止B借助B侵犯B供给B供B作者B体检B伺机B	仇子明B亿利B人生B五丰B二级B事B之一B中国证监会B中国移动B两度B两岸B东方宾馆B业主B不宜B不同B不休B上缴B三重B三精B三宗B万人B一路B．B鼎立B鸡肋B首家B首例B飞乐B	风云榜B	领跑者B预警B预喜B顺势B青B集合竞价B限期B长征B锦华B错误B鑫富B金牌B金丰投资B	重灾区B那些B逃税B送股B连发B边缘B	输油管B转机B	转基因B车险B车辆B	跌停板B趁机B起舞B赎回B豆粕B调低B解散B解B角逐B西游B袭击B袭B补充B	衍生品B虚报B获刑B致歉B	自然人B自杀B	自动化B股王B股再B联B考核B翻B美林B罕见B网购B缺钱B缺位B缠身B绵世B给力B绒业B经B纸B纳B	紧箍咒B	索罗斯B简历B破裂B省市B目的B皖江B电站B电B琼花B理念B珠展B	珍珠粉B现状B现在B特大B爆料B燕郊B燃控B热潮B	炼油厂B	炒房团B漏油B	港交所B渝B渐行B渐入佳境B清退B深市B深化B淡水B涌动B海B济铁B流域B洼地B	泥石流B河谷B沱牌B江钻B	汇添富B气B	欧元区B根源B标签B	柳传志B查出B杀手B机密B未涉B未必B有限责任B最佳B曲酒B普天B春季B昔日B明晰B时隔B日系B无限B无门B施压B	斯迪尔B救援B攻守B收储B攀钢B搁置B	提款机B推介B	控股权B掌握B授权B抵押B披星B抗旱B抉择B扣划B打包B所有B成果B成本上升B	徐可强B徐B归属B开户B开店B建造B延迟B底价B并非B并未B年终B巨变B左手B巅峰B岁末B山B	尹明善B寻求B寄望B容易B家房B	审计署B	实验室B	实名制B定调B完整B学生B季B孕育B女性B奇迹B	大比拼B大桥B大小B大事B增至B基调B坚瑞B坚定B国债B因大B回补B四年B售后B周年B告急B合B号线B右手B只有B变革B取代B反映B双规B友谊股份B	原高管B原油期货B压顶B十亿B	十一五B势B	副省长B利用B判刑B切换B出路B凸现B几B减值B减B冶炼B	农民工B军团B再续B内涝B关铝B全资B先进B兆山B充裕B	储备库B偏高B偏弱B偏低B债B	倒计时B保荐B作B何在B传化B会计B优良B休克B价涨B价格下降B亨B	产销量B五家B五宗B了解B买股B九B之路B之星B丰B中将B中外B严查B东B不过B不跌反涨B不确定性B不惜B下午B三种B三川B三千B三九医药B一行B一块B高涨B高压B额度B预告B降薪B阵痛B阴跌B阴谋B阴云B闪亮B销毁B铜城B金环B量产B	重组股B	采矿权B酒业B配件B避税B	通行费B适时B适度B连连B连拉B近半B较为B车轮B跨B跌跌B超标B资金紧张B	费用率B费B购置B货车B贤成B谋变B详细B评为B警报B视察B	见光死B	草甘膦B	节假日B舟曲B腰斩B	股大涨B职员B终极B终B纯碱B系三B管理效率B策划B突显B空袭B空B称其B	福布斯B神经B社区B破净B眼中B看市B百度B白色B	电视剧B生意B生B特种B牵涉B熄火B煤机B焦化B灾害B灰色B演讲B游B清B深证B液晶电视B洽谈B	洋垃圾B泰复B汾酒B汽车产业B	氧化铝B气氛B榜单B楼价B梯队B框架B样本B松绑B	李彬海B术B本地B期铜B有戏B最终B最低B暂无B星B	易方达B昌河B时富B时刻B新药B新房B	敏感期B放心B攻击B收复B支援B播报B撤资B撤B摘星B搭B提出B损益B捂盘B振荡B指责B拯救B拟设B拆B抗跌B扩展B打响B打入B手软B户B成霖B成发B感受B意欲何为B意向B意义B惊人B悉数B恒强B总额B总监B急冲B态度B德棉B微车B微博B徐工B征B	影响力B	彩电业B	当事人B强者B	张志忠B开高B开出B	座谈会B底线B已过B已成定局B工大B山焦B就业B将现B审议B宜B	定价权B它B套房B失事B	天花板B天线B天海B大王B大放异彩B外滩B增长势头B增势B堪忧B基因B城管B国兴B四倍B商机B唱B哄抬B和源B含B向东B名流B合建B合康B吁B口B受困B受制B发酵B发生爆炸B反腐B双节B双良B双反B及时B单价B华山B华人B包机B包揽B动态B动作B加紧B	力拓案B制定B制作B利息B利于B判决B列车B列入B分钱B出席B凯基B凭证B几近B几年B准入B	冤大头B再超B再发B内房B共建B八倍B光纤B偿还B停车B做好B假摔B	债权人B借口B倒逼B信批B保值B体育B位置B休闲B以及B	代理商B人民B亩B交割B五洲B亏损企业B	二十年B买地B九发B乘B之间B之旅B之前B	中资股B中资B中证B	中药材B中方B严控B严惩B两种B两宗B两名B东银B不高B不满B不了B上限B三高B三日B三佳B万份B七月B七喜B一直B齐星B黄B	高房价B	高开高B高开低走B饱满B飞天B飞信B飚B风力B额B频遭B	领跌股B	预增王B顾问B韩国B靴子B霸主B雪上加霜B集B陶瓷B附属B阻碍B阻击B	铁道部B钻井B钢市B鑫B重拾B郁亮B	避风港B遍地开花B透视B选B追问B追责B连遭B这个B返航B	近九成B轮船B车载B身影B跳槽B距离B跌超B跃进B足坛B足B越南B赛格B赔付B费率B负B谜B	诺基亚B评述B评价B访谈B讯B认识B	观察报B要点B裁定B被部B蕴含B蓝天B	药监局B药店B自称B自然B股有B耗资B缓解B	绩优股B绩优B	统计局B结论B经历B细数B纵深B红利B篇B筹建B	等离子B笔试B笔夫B站稳B立足B立体B窘境B窃密B	空高开B积分B秒B	科技股B离开B离B禁售B禁令B矿价B	石家庄B短缺B知道B知识产权B真实B直面B直言B直奔B直击B盯上B白领B登顶B电子器件B田B用电B环评B王牌B猛B独B状态B特钢B燃油B熊市B炒令B滞后B源自B湘潭B渗透B	深加工B涉嫌犯罪B流向B泰富B法庭B沟通B沉默B	汽车业B汽贸B汇改B水质B	民航业B民众B殃及B此B检修B核实B杨B	李旭利B有点B有助B暗流B早市B旧账B无偿B	旅游业B方圆B教育B教B	政策底B支承B换届B捐B捂地B拘留B拓B抵制B报名B抢眼B抛压B	批发价B手握B房票B房地产商B截至B成绩B愈演愈烈B恶B总结B	总公司B必要B征集B形态B归B强拉B弊端B开业B序幕B年份B干预B师B巴黎B	展览会B居B尘埃落定B小额贷款B将会B寿险B	家电业B家庭B审查B学会B如愿B女友B奥飞B奥B奖金B奇瑞B头B夭折B	太子奶B天兴B大航B多点B多久B外地B复出B壁垒B增多B	城镇化B	型基金B均衡B图解B国税B国祥B国寿B围猎B囤货B嘉瑞B	啤酒花B商住B品B和谐B吹响B名酒B同比增加B合计B各路B可以B受限B受审B双底B友利B去年同期B卖房B卖壳B协助B千B医疗器械B医生B医保B	北海港B动能B前途B初显B分离B击退B减弱B冷链B决策B冲动B农化B农B	公安厅B八月B充电B元旦B停业B	候选人B信任危机B保险公司B保守B佣金B作秀B体制B低谷B传递B传播B传承B休整B任职B任务B仪表B令人B仍处B产业政策B交易中心B亟待B五菱B五月B五只B事业B争取B买进B买盘B之王B之下B举措B主体B中国富豪B中介B个人资料B东盛科技B不振B不如B不力B上年B三七B万亩B七星B一期B一手B一大B一哥B齐头并进B齐升B麻烦B鹏华B	鸡西市B魅影B高清B首证B首富B饮B飙B	风向标B颠覆B	领军者B预言B	预增股B顺义B面纱B面向B难言B隐身B降临B	陈景河B陈年B附B阶梯B防守B	阅读器B闪耀B长阴B长丰B	镁合金B错杀B银华B铊B铃木B鑫龙B金果B重建B重出江湖B	邹晓春B遭查B逝世B通货膨胀B通B逐年B透支B选举B退房B迷茫B迟迟B连阴B连涨B连年B连封B连创B过去B辅导B路在何方B起点B走俏B赢B赛跑B资源优势B账款B	责任险B财务费用B财务总监B豇豆B豆油B谜团B调节B调仓B误导B误区B设施B议价B认同B	警戒线B规定B	西气东B街头B	行贿罪B血液制品B蓝图B菜价B英雄B英股B花B芯B	节能灯B舞B自身B自购B	自来水B能源管理B股送B股份公司B耐心B翘楚B群起B罪名B罚金B网民B网吧B缩短B绿化B终于B纪实B简讯B签约B筑信B笼罩B第四B突遭B空方B空客B科工B离岛B福利B禁止B硅谷B破冰B	石膏板B相信B盛润B盘面B皮宝B百科B病菌B疯涨B	电解铝B电纸B申花B瓜分B特点B牵动B父子B燕子B煤炭企业B炼成B点报B火B激进B潜质B源发B	清明节B深幅B淡出B	涌金系B	消费品B海隆B	浩宁达B洛阳B波澜B油轮B汽配B汇兑B欢乐B	桐君阁B	案一审B栋B构成B	权之争B本溪B未现B期限B望京B服装出口B有序B有力B最烂B普涨B晒B明代B无虞B旗舰B	方高管B数控B数亿B敦沛B救主B放行B收获B攀B	擦边球B操控B提名B提交B推行B推定B排污B挪用B	挖掘机B	指标股B指早B择机B拟售B	招股书B拉高B拉闸B拉开帷幕B押B	技术性B扭转B扣件B打扮B手中B所持B	所得税B房子B房地产业B或许B或致B成行B成型B	成两市B惹怒B惠普B惊艳B悬B您B怪圈B德银B形同虚设B强于B张股B引进B建中B度日B底牌B并入B带业B市委B已达B	工作组B工业用地B崩溃B展示B	尤洛卡B封装B对象B对接B对冲B富翁B富力B密码B宽带接入B定制B宏碁B安阳B学院B存托B子B嫌B威尔B	委员会B奋力B夺冠B失火B夫妻B天广B天后B大限B大腕B外籍B复杂B复地B	基金业B基数B基于B培训B埋伏B坐B在建B在京举行B圆满B国际原油B国诚B团B因何B	回报率B喷吹B喜忧参半B喊冤B商行B商圈B	唱主角B唐山B哥B同意B同志B合适B	可诉讼B另B变频空调B	变速箱B受让B受损B发挥B发帖B反倾销税B压缩B压B	卫生部B博客B华龙B	华尔街B华中B千点B十股B劫B动荡B前高B刷新B利B初裁B刘坚B列B刑责B击穿B出走B出尽B出事B几成B净值B决裂B冲关B冬季B农网B农场B再出B内讧B兼具B入围B先生B兆瓦B僵持B储B偿债B停售B做强B值B倒卖B信诚B保费B保有B保价B侵吞B依赖B作业B余热B余B低如B估计B传将B众多B休眠B	伊拉克B价齐B代B亿立方米B人才B人均B产品升级B五倍B之道B之痛B之战B丰东B中财B中联B中止B	中广核B中子B两人B	专用车B专家建议B与否B不当B不利B不减B	上证所B三项B三问B三地B万设B万平方米B万力B	万人次B一波B一枝独秀B一条B一德B一半B★B齿轮B黑金B黄陈B黄酒B	黄晓明B	麦考林B鸡蛋B	魏家福B高额B高科B骤降B骤增B验证B	驱动力B驰援B风雨B风声B鞍山B革命B静候B集群B集成B雅砻B雄震B难挡B院B陈俭B阿里巴巴B阿继B防止B队伍B锡矿B铸钢B银联B铁腕B钒B金融公司B	重金属B重金B采用B郭B郑步B遂昌B逻辑B逞强B通吃B透明B逃离B逃亡B退地B退休B追债B迷踪B迷航B迟到B违规行为B违约B远超B远离B远期B过高B辞去B较劲B轻仓B转正B转化B路上B走漏B资不抵债B贿赂B	贵金属B购矿B货物B账面B	负债率B豪掷B谢幕B诱因B试验B	试探性B证大B	证券法B	证券报B	讨说法B认可B触发B西旱B西布B装饰B	被究责B行列B血拼B蜂拥B花钱B色B良性B致使B至今B	自行车B自动B自力B能够B	胰岛素B背靠B背离B股齐B股遭B股转B股成B	股发力B肉鸡B耀眼B美克B绽放B结转B经贸B经观B	经营性B	组合拳B纷争B精粹B简述B简评B筹钱B第二季度B竟B立法B突遇B税费B秘籍B科健B秋季B禁B祸B矿企B真金B相投B相当B相互B直逼B直播B盗版B皮革B白猫B痴心B界B男孩B	电视台B电石B电企B瓶B理解B班B珠峰B环节B	王老吉B王国B猛涨B独特B狂跌B狂潮B牛B牌照B燕京B煤制B热销B点击B炒家B湛江B湖山B渎职B清空B清白B海域B浪B流行B泥沼B法国B沽B油污B油企B沪深股市B求解B汀B气体B歼B正道B棱光B桥B株洲B极限B杠杆B李宁B	机顶盒B本轮B朝B最近B	替罪羊B曲明B智能手机B春运B昆仑B日益B日渐B日内B无氟B文章B数月B教授B救赎B敌B放贷B支线B	操盘手B撞死B	摩托车B揭示B揭幕B掷B控股公司B接棒B探路B探索B	排行榜B据称B指微B指导B挂B拷问B拓宽B抽查B投保B抓紧B执掌B打乱B成分B慢B意图B惹祸B惠民B惊魂B悲情B恶意B恶化B恐难B恐陷B必然B心态B微调B得而复失B影视B当时B弄潮B开闸B开车B开庭B延报B廊坊B座B广昌B幻影B席卷B市委书记B已有B崩盘B局部B封基B对方B家用B实践B宏宝B宇通B学校B	存三大B字头B妖B如果B女儿B奖励B夺魁B夺回B夺B夹击B头号B太空B太平B天胶B天天B	大跳水B	大跃进B大礼B大楼B大减B大众B多少B	处罚金B增产B基民B坍塌B均值B国际航运B国际期货B国浩B回家B四板B商铺B	商标权B商会B商业银行B商业模式B唐钢B	唐山大B呼唤B	周期性B吹B启B吉炭B吉B合金B合格B各国B司机B台风B另有隐情B	变压器B变卖B发起B发股B双龙B双管齐下B厂区B危局B印尼B卫视B卡夫B卖空B协同效应B华讯B华商B十点B十字路口B	十字星B	十余年B动车B动物B动工B动力电池B加重B	办公室B力促B剖析B	前高后B	利用率B初见成效B刚果B刑事责任B凶猛B准B净利润率B再造B内蒙B内幕消息B内在B养老B兵B兴建B关停B六年B全球股市B	全流通B全新B入账B光环B傍B	傅成玉B停工B信誉B信用B保驾护航B保护主义B保安B	保卫战B便B侨丰B佳B作出B	余万元B何处B低空B传中B伟大B优于B任命B价高B价量B以内B	什么样B交B五股B争B买点B买单B九天B之惑B之困B义煤B义乌B主攻B临B丰厚B	中邮系B中通B中进B中试B中福B中力B个人简介B两类B	业务量B世基B	世博园B不降反升B	不锈钢B下注B下沉B下发B上榜B上位B上交B三驾B三罪B三期B三方B三度B三家B三名B三人B	万吨级B	一日游B一夜B一举B～B龙B鲁阳B高成长性B	高弹性B马车B马B首市B饮料B飞跃B风采B领袖B	领航者B预减B顾B音频B非常B雨B雅致B障碍B隐含B限B阔绰B问B闪现B长百B锻件B锦上添花B银行存款B	铝板带B钢丝B	钓鱼岛B金融危机B	金融业B金系B金元B重上B采访B	邓普顿B邀B道德B遇冷B逾期B造车B选购B选址B逃逸B追诉B违建B进步B还贷B运动B输美B	输大市B轻松B轮动B转手B转回B路线B跨界B跟进B趋缓B趋稳B趋B超越B趁B	赵伟平B走上B赣B资费B贴B贯通B贪污B货源B败诉B财源B豆类B谭B谣言B	调查组B诺安B请求B说法B语音B试探B访B设厂B讲述B记忆B记录B	计算机B解码B解套B角色B	规模化B要约B裁员B血B蚁族B虚开B薄膜B落地B落B获大B草根B苏物B艺术B艳照B船队B	航运业B航海B舞弊B膜B脸色B脱帽B胡B胜景B股迎B	股行情B股冲B老赖B老太B美邦B美罗B	罗伟广B网优B缺B缴款B绵阳B绩B绝招B结案B结婚B终审B组B纲要B红线B繁荣B糖B精英B精细化工B粉饰B	第四届B	第六届B	第三届B	第三代B笑B端倪B竞买B突围B空转B穷B穆迪B秦岭B	科技园B	私有化B离任B票价B碾B硅股B破题B研制B着手B真的B看涨B	看上去B直追B直冲B盯B盐化B盈科B百日B	百分点B百余B疑点B留B男人B电容B电子产品B用药B用料B	珠三角B玩B狂降B	爆发力B熔盛B	烂尾楼B炼化B炮轰B激发B潜逃B满月B满意B渔民B清仓B混合B混乱B深套B淡化B	润滑油B涤纶B涅槃B浓B流拍B活力B洪水B洪B泡汤B油站B沉寂B沈B汤B求B水运B水灾B民资B民用B民品B毫不手软B毛B每桶B止跌B欲望B榜首B检验B检方B梦碎B	桂浩明B查分B染指B果断B	板高管B松江B杭城B杜鹃B	李彦宏B机舱B机关B本币B	本土化B未解B未受B未决B未入B	服务业B有超B	有起色B月饼B月度B	曹德旺B暗藏玄机B景观B普通B显卡B昨遭B易涨B明胶B旺B	早籼稻B日趋B日记B日期B日均B	日午评B	新进展B数测B	数字化B	数十亿B敢B效仿B收费B支架B摇号B搞定B揽入B提议B	提供商B接到B接B探讨B探B排队B	掌门人B换股B按期B指低B持稳B拟斥B招生B	拖后腿B拖B拆除B抬B报收B	报复性B报喜B投运B投资规模B	投融资B抑B承载B扮靓B	扛不住B扔B打工B房山B成败B戏B意欲B惜售B惊动B患B总是B总分B忽略B念B必备B得益B征收B征地B形B当年B强调B强制B弱B张刚B引来B弃权B弃B开支B开打B建设项目B	廉租房B废钢B庆发B庆典B广陆B	广东省B平静B	干散货B干B已然B已到B已决B巨单B工银B山矿B屯河B属于B屏B局势B尾市B	尚福林B尚无B少数B少女B小麦B小跑B小跌B小心B封口B对比B对抗B对价B富昌B富国B家属B	宝能系B威力B委B奶糖B	奢侈品B失误B失实B大越B大水B大师B大市B增量B填补B堪比B基资B基站B坚实B坏账B地质灾害B地炼B在手B在于B土地储备B国通B国联B国标B	国字号B回笼资金B回笼B四家B囚徒B喜欢B品牌价值B命悬B	周生生B吸收B吴B启用B后扬B名牌B名人B同期B吉尔吉斯B合资企业B合臣B各项B各有B各大B司法机关B	史玉柱B台企B	变频器B	受让方B受灾B	受害者B发票B双料B	双引擎B双头B双向B叉车B厅级B占据B南昌B单月B华融B午B千亩B	医药业B区域规划B北海B北江B包括B势力B努力B助理B功效B办理B办B力合B	副行长B前行B剂B刷卡B	制高点B利卡B初期B切记B分销B分行B分类B分散B凭B几无B减至B凌晨B准入条件B	净资产B冲回B冲上B冰火B冬天B农户B军B写B冒烟B再迎B具吸B共振B六宗B公文B公安机关B公共B八股B八只B八五B全责B入伙B兖煤B先机B先抑B先后B借壳上市B倒退B修绩B修改B保险资金B	保险业B保存B侵害B供气B低端B伪B伟业B优选B任重道远B仪化B以下B代言B代线B仍难B	今复牌B仅剩B	亿限售B亿成B亿多B	五星级B五星B二连B二奶B争霸B争斗B九大B之险B之际B	主旋律B主张B临汾B	中长线B中茵B中中B两至B两点B两极分化B丛生B世联B专用B专户B不易B不小B不务B上学B上午B上千B上冲B上下B	三高管B三套B	三剑客B万平B万家B七只B七个B一点B一切B―B黄河B	麦当劳B鸡苗B鲁证B高手B骨头B驾车B驻扎B香水B饮食B餐具B	风电场B风投B风向B	领跌板B面世B非洲B非标B静观其变B青岛碱业B难测B隔夜B随时B附表B	阿根廷B	防御性B闹剧B间接B闯B闭幕B闪婚B长协B锦州B锋芒B铝箔B铅锌B鑫安B金子B金化B	金三角B重阳B重获B重构B重归B重型B重兵B释疑B采掘B采取B配股缴款B部副B郑棉B邳州B遭涌B遭到B遗书B逼债B	造船业B	造纸业B造就B造好B选手B逆势上涨B远虑B进账B进攻B	进出口B这样B还原B近忧B运用B过火B过户B达产B轻工B	软件园B	软件业B转送B转投B转入B践行B跨入B	趵突泉B趁势B起来B赵B走跌B赢利B贷案B购销B贬值B贫困B败走B败B财年B财务报表B谷底B	谢国忠B语录B诀窍B许可B计入B解困B	观察期B	见分晓B覆盖B表述B表演B补仓B街B	行权价B融入B融B蜱B蜗居B蒙古B	蒋洁敏B	葡萄牙B落网B	茅于轼B英皇B英B苦果B苏丹B良运B致辞B致命B自我B自仪B脱颖而出B脱硫B能效B背水一战B股配B股遇B股盘B股权结构B股今B职业B而动B	老股民B	老字号B	翻两番B翱翔B翘尾B网卡B缺席B缴税B缘于B缘B	综合体B维和B维B继任B结局B经济腾飞B	经常性B纽约B纯属B纤维B红色B红海B紧跟B紧绷B紧B精确B粽子B米价B管网B管涉B管控B签订B签署B答记者问B等于B	第八届B符B竞价B空军B税控B稍B稀缺B称王B	秦皇岛B	秘书长B私吞B	磷复肥B	碧桂园B硅片B砍B矿石B短暂B督办B眼前B看看B看法B	省运会B	直通车B直封B盛B皇帝B疗法B疑因B留下B生机B瑕疵B理赔B珠拉B环球B	王文京B玉石B率众B猪肉B猪B猝死B狼B独资B犹在B状况良好B	特高压B特区B牛气B牛散B片B	爱尔兰B燃烧B煤业B烯烃B热议B热炒B	热毒宁B热情高涨B炒成B激辩B	澳矿企B滞留B滞涨B	滑铁卢B源股B湖科B港澳B渣打B清淡B清查B清单B	混凝土B深耕B液化B	涉少报B海星B海外投资B海口B浮云B浪漫B浓厚B浅析B派发B泵业B泰阳B法巴B油管B	油气田B沦陷B汪B汤臣B汕头B汇市B水利B水分B气荒B民警B此股B欠缴B横空出世B榨菜B	楼世芳B	植入式B植入B森林B检测B梅雁B档B染料B林B极寒B松辽B杜B朵朵B未减B朝鲜B最长B最爱B最富B最多B	曹妃甸B曲轴B暗算B	智能化B智利B	普及版B普及B晨鸣B春天B昙花一现B明起B旺销B时候B旧事B日航B无望B无意B无恙B旋风B新机B新址B	斯巴鲁B断货B斩仓B料首B斗B	数千亿B救灾B救命稻草B放松B放开B改娶B支招B摘得B搭台B搅局B推盘B	推介会B掣肘B掘B掐架B捂B挖B指控B指急B	指大涨B指向B指创B拳头产品B招聘B拍得B拍地B拉近B抵消B抬轿B报复B抛股B抛空B折腾B	抗跌性B抓捕B承包B扶植B执法B打通B	打水漂B	手续费B手法B房展B战火B战机B战场B或变B成股B	成绩单B成本增加B	成本价B成效B	成思危B成下B	意大利B惩罚B惠及B悖论B恶意中伤B	恐慌性B恐吓B总统B总成B急升B心得B心B德B微电B得以B很多B待遇B待破B征程B往事B	彭小军B当当B弃儿B	开门红B开门B开战B开展B开售B延后B	康师傅B并存B年初B年关B平价B幅B常林B帮B市中B差别B左B工会B	工业品B工B岌岌可危B展台B尽显B尼日利亚B尝试B尝B尔B寻租B寡头B宾馆B	家财险B家涉B宣泄B宠儿B实拍B实惠B安装B孩子B学B嫦娥B嫌疑B	姚建华B委任B奶牛B套餐B套期B奔驰B奏效B头筹B	头等舱B失手B大门B大起大落B大步B大杨B	大峡谷B多次B多处B多地B多只B外销B外观B外媒B外出B复盘B	处方药B增利B培养B	城中村B场外B场B地标B地板B地图B地主B圣诞B圈B国安B	国土局B国人B围剿B园B因正B因信B因为B回国B回到B四季B四天B器件B商讨B商店B商品价格B	商业化B售出B哄抢B告吹B启幕B吞噬B名额B名不副实B同悦B吃紧B台阶B叫卖B叫价B叫B变为B取保候审B发表B发放B发出B双马B双翼B参考B参建B去意B厦门B原罪B压弯B卷B印B占领B占优B博时B博士B博会B博B南下B卖楼B卖单B单日B华福B半岛B升至B十日B医药公司B北美B北上B化工产品B	化妆品B募B加价B副董事长B剧烈B	前高管B前进B前后B削减B刻意B创世B划拨B刑事B切入B	分数线B出错B	几十亿B	几内亚B减肥B减缓B减税B减价B减亏B	准备金B净化B冷清B冷B	冶炼厂B再起波澜B内定B兼修B兴德B	兴奋点B关税B共识B六家B	公积金B公民B	公安部B公司债券B八旬B党委B	光通信B先斩后奏B儿童B催化B储地B偿付能力B停飞B偏强B假地B假冒B倾销B借贷B倒戈B保持良好B保供B	侵权案B依靠B供大于求B何B位居B伦敦B伤亡B传动B会签B会合B伊朗B价格调整B价差B价位B代持B今生B仅售B亿售B人大常委会B人大B京津B产险B产地B	产品线B交织B	交流会B交技B	交割日B亚B	井喷式B事发B	事业部B争锋B争购B争做B了结B乱局B九月B之手B之中B主管部门B	主战场B为宜B中美关系B中瑞思创B中澳B个性B个别B丧失B严格B	两重天B两舱B两翼B两条B两拓B两千B	两位数B丢B东亚B业界B不破B	不知情B不知B不得B不少B不乏B下移B	下一代B上风B上阵B	三部曲B三足鼎立B三条B三分之一B三亚B七倍B一片B一架B一对B一家B一套B一则B一位B龙王B齐飞B齐发B	黑龙江B黑嘴B黄石B鲁B鱼B	高送配B高设B	高端化B高温B高校B高企B骗取B骑士B马来西亚B马丁B餐饮B食品行业B飞涨B飞抵B风水B颇B	领导力B预盈B顶风B音乐B靠前B静观B	零售商B雅鲁藏布B难阻B难消B难料B陷全B除息B	陕西省B	陈金霞B	陈榕生B	陈发树B阿里B阵脚B阳谋B阳晨B防伪B长短B镜花水月B锡B错失B	锂离子B销往B	销售量B铸造B铬B铁B钼价B	钻空子B钱景B钟情B钗B鏖战B金饰B金银B金融中心B金蝉脱壳B金砖B金华B重拳出击B重在B酿造B酷睿B酱香B配额B	配股价B	郑少东B邯钢B邮轮B那B邀请B遭拒B遗弃B遇挫B造价B通航B逐鹿B逆B退烧B退场B追赶B迭B迪康B	连锁店B连破B	进口车B近年B	近一成B迈出B迈亚B过期B辩解B输送B轻骑B转网B转攻B	转折点B转产B车用B身体B踊跃B跑马B足协B	越来越B超额完成B赶超B	赵本山B赴美B走向世界B赣州B资B贪官B质地B货B账B财产B财B负面影响B豪华B谜局B谋杀B调升B调任B诱惑B详情B诚聘B诉至B证明B证B	设计师B论文B论剑B许家B讲B认股权证B订造B触屏B解约B觊觎B视B觅B	西班牙B西宁B襄樊B	表演队B表彰B补税B街道B衍生B行长B虫B薪水B蓟县B蒲县B落成B	营业税B萎靡B莫名B莫B荆州B苗药B节B船厂B航B舆论B致死B自由B自掏腰包B自制B自产自销B脉冲B脉B肯定B股飙B肆虐B耕耘B老牌B老师B翻脸B	群英会B群体B美国政府B罪遭B罢工B网罗B网店B网帖B缺少B缴B缩小B缥缈B编织B编码B缓B绿鞋B绩增B经济学家B终裁B终将B索道B索尼B	系统性B精装B粮价B箔B筛B	第四次B第四季度B	笔记本B端B竞逐B立方B立即B窝案B窗B窃取B穿越B空售B空前B稠州B程度B称雄B称为B积压B科苑B	福建厅B禁用B神坛B社论B碰B硬盘B	研究员B矿难B矿长B短融B着力B相继B直航B直泻B直拉B盛行B盛世B监督B皆B	百视通B百花B百倍B白浪B痕迹B疑虑B疑涉B画饼B男B电量B电子行业B	电子盘B电子刊物B电力公司B甲流B用脚B	用电量B生命B甘当B珍珠B现有B环B特殊B	特保案B牵手B爱情B爆炸事件B	熊玲瑶B煤炭资源B	煤制油B焦B热衷B点金B点火B炸药B炯B灾后B火车B演B漂亮B满仓B满B滚筒B湿法B湾B渐入B	清一色B深鸿B深渊B淘B海勤B	浙江省B派现B洪良B	洋品牌B泽华B注B波B	泡沫化B泛滥B	法人股B沿B治B油砂B河B沥青B沙黾B污水处理B江钨B永远B氰胺B气化B	毒奶粉B段B殴打B残酷B死刑B步B正果B欧阳B欢喜B欠债B横向B模拟B概况B楼板B棋局B棉花价格B检出B桩B桥梁B	案开审B桂林B根本B核B	样本股B标本B柳州B	柯希平B极具B板材B板收B杯B村镇B	李冰冰B	李克强B	李书福B权力B权利B	杀手锏B机车B本质B本色B服饰B最迟B最为B更正B更加B曲折B暴力B	暴利税B暧昧B暂未B景顺B显威B昭示B	星期二B易B时速B旧B日子B日产B既B无须B无需B无果B	旗舰店B族B	新纪元B新海B新开B新年B断言B断流B断供B斗争B文明B数字电视B数名B	数千元B教材B敌人B政治B放宽B攻势B改进B	改扩建B撼动B撤职B撇清B摸底B摸B摄影B携号B搅动B提防B提门B提案B推向B接待B	排量车B	排头兵B掐B掉B授予B损害B挤B按年B指澄B	指收涨B指报B指引B指冲B拼B招金B	拉锯战B拉低B担任B抵B抱B抢先B投身B投资银行B技不如人B批零B批量B批评B扰乱B扬州B打算B打假B才能B手段B手机游戏B所涉B	房祖名B	房央企B戴B截止B或降B成色B成板B成形B慷慨B慰问B感动B惨案B惨B惠天B惊B患者B恶性竞争B息率B恩怨B恒基B恐成B恐怕B总量B	总动员B急需B必拓B微B	得益于B得利B往来B彻底B当日B强强B	弃权票B异议B开设B建部B建材行业B廉价B康B底盘B应有B广B并进B并行B年轻B年代B平板电视B帽子B	巴曙松B已近B	巨无霸B巨幅B工资B工地B工作细则B	工业用B	工业区B峰回路转B山航B履历B	居首位B	局局长B尽B	小股民B小散B	导火索B对待B对华B寄语B宽B家长B家底B实益B实战B实力雄厚B宝盈B宝商B定论B完全B完B宋城B安然无恙B安B宇航B学费B学者B学位B季节B字号B子虚乌有B嫁接B	娄勤俭B委内瑞拉B妥协B奢侈B套住B奔腾B头炮B失衡B失效B失控B失意B失去B	央企将B太罗B太光B太低B大道B大跌眼镜B大胆B	大红包B大片B大展B大家B大力发展B大兴B多年B多倍B	多万元B外宣B	外地人B复活B处理完毕B增点B基药B城区B城乡B埋B型钢B场馆B圈定B国内航线B回答B回眸B回收B回去B四起B四股B四级B四种B嘉宾B嘉实B售楼B和平B呼和浩特B周线B启程B	含金量B吧B吓B后方B吉安B合规B各种B	叶志翔B史B	可行性B	只金股B	另一面B口腔B	变速器B取向B发电机组B	发布会B发展潜力B发动B反驳B去世B原糖B原始B原奶B原副B原则B厂长B卧轨自杀B南疆B南沙B南宁B单独B单一B卓创B华证B	半年度B	半小时B千里B	千万吨B十八B包销B勇士B势在必行B劳工B劲B	助推器B动静B动机B动向B力量B力求B力推B剩女B剩余B前奏B前世B别碰B	创纪录B创始B创下B刚需B	刘积仁B分配利润B分立B	分水岭B分子B出游B出境B凯马B凯诺B减轻B减半B准地B冲锋B农牧B	写字楼B再启B	内银股B内涵B兽药B养生B	养殖业B其妻B关门B共存B六月B公章B公平B公安B八年B入侵B党委书记B光荣B光盘B充满B元股B元老B做局B做客B偏离B假期B假日B债市B借钱B借道B借势B倒腾B信测B保留B侵占B	依存度B供电B供热B作案B余额B	何立峰B似B传输B会议纪要B会员B优酷B企将B价格政策B价格指数B价创B仪器B代价B仓储B从未B仍为B今早B	今年底B今后B今冬B仅为B	亿险资B亿设B亿短B亿度B亿地B亿在B人间蒸发B人造B人选B人人B亮剑B享B产出B产值B交流B亚军B五日B	五方面B五折B互搏B二套B事后B乳品B九点B九年B之选B之忧B之变B丽水B主题公园B主角B主管B	主沉浮B主打B中融B中纺B	中秋节B中油B中州B中创B中企B两高B两江B两幅B两块B两位B两万B东部B东盟B业务收入B世贸B世家B且战且退B且B专项资金B专营B	专利权B与源B不该B不见B不行B不淡B	不明朗B不成B不均B不了了之B下课B下禁B	下影线B上访B	上海滩B上望B	上市日B	三连亏B三湘B三分B万点B	万千瓦B七年B一辆B一箭双雕B一步B一张B一处B一品B一体BⅣB鼎B黑色星期B黄氏家族B黄块B	黄俊钦B麻雀B鸿海B高金B高过B	高管频B高息B高官B高喊B高估B骗贷B驻B驶进B驶向B驱逐B	马红漫B马力B首现B首演B首位B首付B饼干B	饮用水B食糖B食物B食B飘香B风帆B风光不在B额外B频变B颇丰B	领风骚B	领导人B预订B预览B预提B音箱B韩朝B	面面观B面孔B非诚B露面B震撼B雪山B集权B	集成商B	集中度B	雅居乐B雄伟B难敌B难撼B难产B险守B降近B	陈晓慧B	陈志武B	陈必安B	陈天桥B陈健B阐述B闲钱B	闲置地B问鼎B长因B锗B	链断裂B	银行卡B银十B银B	铜版纸B铜市B铁证B钣B	钢结构B钢制B钟信B钛B针对B鑫科B鉴定B金融服务B金穗B金秋B金条B	金刚石B重钢B重税B重点项目B重演B重提B重复B重啤B里面B醉酒B酮B酒香B酒瓶B酒企B配送B配楼B都市B郴州B部队B	部党组B	郑康豪B遭拘B遭一B遗体B通缩B逊B选秀B	选择性B	选拔赛B退货B退潮B退位B	退二进B迷你B迫在眉睫B	迪斯尼B连板B连夜B进退B还会B返回B近来B近在眼前B过节B过年B过冬B过关B输液B输油管道B载客B轻遭B轻罚B轻描淡写B轻型B转为B轨迹B车价B蹦极B跳楼B路产B跨境B跌价B足球B超高B	起征点B赶底B赶B赢嘉B赠送B	赔偿案B贼B贵族B购机B	质监局B质变B财团B贡高B	负增长B豪饮B豪言B豆奶B谦B谎言B谈拢B	谈不拢B	调整期B调度B调减B调入B课程B读B诸暨B	说明会B	说了算B误判B详细资料B诠释B话费B话B试B评测B	证券业B议程B讨要B计价B警民B警力B言B触礁B观B见好就收B见义勇为B要素B	西门子B西芒B裸奔B被控B衰退B补血B	补助金B补丁B行者B行动B血战B虽B虹桥B虚构B藏药B落下B	营收超B营养B	营业额B营B萝卜B	菜篮子B荷兰B药材B荣誉B荣威B荒置B	范一飞B	苹果皮B	苦肉计B苦B若隐若现B苏B花费B花落谁家B花瓶B花炮B节节B节约B航油B致力B自营B	自给率B自传B自买B膨胀B腹地B腰B脆弱B能量B能山B胜出B背光B	股大元B股可B肉B聪明B联营B职责B	耿佃杰B耳机B耐特B耍B	考察团B老公B	老佛爷B群雄B群益B	美欣达B美景B美容B羊B罚没B罗莱B缺德B缰绳B缓和B缓刑B综指B	维生素B维柯B继显B绣球B统计B绝技B绚丽B绕过B结伴B经适B	经纪人B经济社会B绊倒B终盘B线下B纸浆B纸价B纳米B纪念B纪委B紧缺B	索芙特B索尔B索取B糖市B精耕细作B精煤B粮B箱B管理费用B管材B管持B管业B算账B算盘B筹备B	第五届B第五B	第三个B竞赛B竞投B竞B	立方米B突增B突变B空单B	稽查局B稳中有升B税务B称大B称号B积聚B秩序B秘笈B科学B秋天B私B离婚B福B祥龙B神B社B示范B磷B硬伤B硝铵B硝烟B硅B石墨B短信B知钱B知识B知名品牌B看台B看到B看中B省钱B省委书记B省外B省内B盼B相纸B相应B相B直销B直营B直供B盯人B盛大网络B盘后B盘口B益阳B盈转B盈余B盈B皇家B百灵B百威B百亿美元B百万富翁B百B	白马股B白马B白狐B痰B病B疗效B	略高于B畅B男女B男友B电解B	电视机B	电容式B申龙B申报B甲乙B甩掉B生长B生疑B生物质能B甚为B	甘氨酸B瓷器B璀璨B理顺B球迷B环境影响B环境保护B玫瑰B玩具B王征B	王建宙B猛烈B狱中B独揽B独占B狙杀B狂热B特权B特B牵扯B	爱立信B	爱国者B熟B照进B煤油B煤化B热清B烦恼B点球B火热B激战B潍坊B演示B满足B	满负荷B源头B湿地B湘B添彩B淹没B混淆B深挖B液态B涨薪B涨升B涉钾B涂料B海龙B海水B	海富通B浴火重生B	浦江镇B济B浅谈B流程B流浪B流动资金B活源B泡泡B法航B法则B泓B沽售B河源B河流B沪胶B汽车电机B汪洋B	污水池B池B	江苏省B汉堡B汇聚B永久B水稻B水皮B	水电站B水源B气田B民船B民爆B	民政部B民意B民情B民工B	毛泽东B比翼双飞B每月B每周B	残疾人B殆尽B死者B死缓B正酣B正股B正当竞争B正名B横行B横扫B	模块化B	楼忠福B桶B根本原因B标价B柳B林地B	林业局B极大B	杨元庆B条款B条例B村长B李鬼B	李稻葵B	李春汶B杀死B本部B未知B未果B未改B	木地板B木B服务提供商B服务供应商B有钱B有无B有吸B有何B最穷B最深B最有B最早B更迭B更要B更好B曝遭B暴涨暴跌B暗降B暖风B暑期B普B晋级B显露B显象B	昨复牌B	星期五B易人B	明年初B明后B昆山B旱区B早已B日志B日常B日信B旋涡B	方向性B	方便面B	新闻纸B新车B新湖B新港B新宠B新型B	新品种B	新台阶B新务B新军B断B	文化节B文化产业B整装待发B敲诈B数据中心B数年B敦促B散货B散B	教育部B教师B	救援队B政经B	政策性B政协委员B放假B攻克B改投B改口B改判B改为B收集B支出B支B擅自B撤侨B撞车B撑起B摩托B摇篮B	摆乌龙B揭黑B揩油B插上B提起B掩B推倒重来B控制能力B控B接轨B	接班人B探访B探明B掀翻B捡漏B损伤B捕捞B捕捉B挽B挫B挪移B挑B指日可待B持久B拨开B	拦路虎B拥B拟将B	招股价B拘捕B拖欠B拒贷B拒收B拍案B拉美B拆卸B拆借B担责B抵御B抬高B报表B报案B抢滩B	抢反弹B抚顺B抗议B抗癌B投资信托B	投行部B投标B抑或B抄袭B技术升级B承受B扭曲B扩程B扩散B	扩张期B	打白条B手足B手续B手术B手持B房租B	房产商B戴尔B或迎B或谋B或临B我省B成套B感觉B	意外险B惹火B想像B惨淡B惠州B情理之中B情妇B悬殊B悄悄B恶果B恶劣B恰逢B恐将B总包B怪B性质B急病B急欲B怕B	怎么办B怀疑B怀揣B快递B快车B志在B必看B必有B心思B	德隆系B德赛B微妙B得失B	徐留平B徐州B待定B征途B征用B征战B形象B录B当期B	强降雨B强推B强拆B弥补B张茵B	张维迎B	张悟本B张勇B弃购B开辟B开演B开标B开学B开刀B开低B开会B	建造师B建文B建国B康日B康佳B应用软件B应付B庇护B广药B并重B年间B年换B年度人物B年增B年净B	平方米B平坦B常州B常委B帷幕B带队B帐B布子B布B	市政府B	市政协B	市占率B	巴斯夫B	巴拿马B	已无钱B巨鳄B巨增B巧合B左颖B工艺B工信B川B岷电B屯兵B属性B展翅B居里B居住用地B	尾矿库B	尼罗河B	尼日尔B尺寸B尚远B尚有B	小马拉B小区B尊重B将促B封顶B封至B	导火线B对阵B密谋B宿命B容忍B家锁B家润B家庭理财B家店B家乡B宣告B客人B实用B宜宾B定期B	宗庆后B宏观政策B安凯B安保B守望B守住B守B宅地B学子B	孙建波B婚姻B	威灵顿B如虎添翼B如常B如实B	好日子B奶酪B	奶制品B女生B奏B夺取B夺下B头发B失血B失算B央票B央企拿地B太钢B天相B天津海关B天大B	大逆转B大车B	大秦线B大潮B大款B大哥B大发B	大卖单B大升B夜航B夜市B多股B多种B多多B多位B多万B外购B外衣B外渗B外来B外币B	外媒称B外国B复航B填权行情B堵B基石B坪石B坠亡B地面B地皮B	地价款B圆满结束B国货B国策B	国家队B国军B国储B固话B固定B围观B囤酒B团伙B	因信批B	回马枪B回炉B回流B回复B四类B四折B四度B四国B喊B啖B商飞B商界B唱响B售罄B售后服务B哈B员B告一段落B听B吞并B吝啬B	后首案B后见B后来居上B后悔B后台B名片B名家B名厅B	同质化B同样B同日B合营B合生B	合作方B	合伙人B合一B吆喝B	叶荣添B右翼B	史铁生B	史立荣B台长B可行B可操作性B可持续性B可持B可低B叫板B只卖B	口水战B变迁B变故B	受钢价B受控B发潮B	发家史B	发审委B反向B	双胞胎B	双增长B友邦B又现B参设B原酒B印记B印后B卡特彼勒B卡位B占近B博鳌B博纳B	博物馆B单品B华菱B	半潜式B半壁B升涨B升幅B十分B十倍B十个B	区域性B北斗B	化工业B包袱B包围B勿扰B势必B	动车组B劣质B加薪B办法B力挽狂澜B力争B前台B削弱B券B到货B利器B利剑B判B初露端倪B初具B创展B则B	刘晓忠B划定B划入B切勿B切割B分派B分会B刁扬B出面B	出货量B出色B出招B准确B	净流入B	净亏损B净亏B	决策层B决心B冲突B冲垮B冰岛B	冯小刚B冠捷B军舰B冒充B再战B再受B内饰B内衣B内核B冀B共计B共推B共促B	兰先德B六折B	六个月B六个B	公安局B公司债务B公允B	公交车B八家B	八千万B八一B全身而退B全市B入局B光缆B	先驱者B充沛B元年B允许B儿子B	储气库B傍上B	傅育宁B偷逃B假如B倾斜B倾向B	倡议书B	信披存B信心危机B保金B	保荐人B	保监会B保牌B保壳B依B供水B供暖B	佳兆业B作品B作价B	体育馆B	传媒业B传前B伙伴B优异B众人B休B伏击B	伊立浦B伊始B	企业化B仿真B份B仪征B以为B代工B他人B从良B从何而来B从事B从中B从严B仍待B仍具B今晨B今明两年B仅增B亿造B	亿澳元B亿扩B亿平方米B	亿吨级B亿取B	亿业园B人间B京郊B享有B产融B交锋B交警B	交易会B亡羊补牢B	亚马逊B亚视B	五连阴B五类B五天B五个B互B	亏损额B	二甲醚B二月B二季B事情B争食B争执B乱世英雄B乱B买矿B九折B九只B九倍B乙醇B乔洪B乐园B之殇B之内B之上B义务B么B久拖B丽江B为求B为先B为了B临停B	中铝力B中途B中资银行B中芯B中立B中游B	中消协B中毒B中旬B中断B中拟B中建B	中国区B	中北因B中农B中共B个贷B严令B	两部委B两起B两元B两亿B	两三年B东征B业园B不符B不敢B不怕B不安B不妨B不及B不卖B不到B不久B下水B下方B下放B下个B上空B上移B上瘾B上火B	三重门B三至B	三级跳B三甲B三亿B三九B丈夫B万部B万象B万立方米B万拍B万到B万保B	万亿元B丁磊B一艘B一站B一生B一波三折B	一次性B一控B一战B一座B	一年期B一吨B一厂B一元B一亿B一个半月B／B＂B﹡B	龚家龙B龙岩B齐跌B齐涨B齐整B	鼎工受B黑榜B黑户B黑客B	黑匣子B	黎明前B黄松B黄昏B	黄俊灿B鹏飞B	高风险B高陶B高附加值B高烧B高潮迭起B高淳B	高峰期B高尔夫球场B骨干B骤减B骗购B	骑车人B驾B驻华B驴B驱B	马自达B	马纯济B	马晓东B	马士基B	马化腾B	香饽饽B香B首降B首涨B首款B首单B饮水B饭店B饥渴B餐桌B餐B食谱B	食用菌B食物中毒B	飞虎队B	风生水B风流B风情B频B领先地位B预演B预报B预判B顾虑B顺风B页岩B韩泰B鞭子B鞋B非议B	非理性B青春B露出B震幅B震区B雷锋B雷人B零碎B零时B集装箱运输B集聚B雅虎B雅培B雄踞B雄厚B难遏B难抵B难成B难变B难估B隆重举行B隆重B陷入困境B险情B险B降雪B降职B	陈少丹B	附加费B	陀罗尼B阻止B阻挡B阻挠B	阻力位B阴领B阴阳B	阴谋论B阴B队B阀门B闽粤B闽发证券B闹B间谍B间B门罚B长风B长跑B长期趋势B长丝B锤头B错过B锆B锁B链接B铸就B银根B银川B银基B	银华系B	铝合金B	铝加工B	铜精矿B	铁娘子B钽价B钼矿B钻杆B钨钼B钢板B钢坯B	钢价续B金鼎B金融时报B金盛B金创B金九B量升B野心B重蹈B重燃B重掌B重塑B重伤B里程B酿B酱酒B酒都B郭勇B	郎咸平B那样B那么B避重就轻B避谈B遭热B遭标B遭旧B遭性B遭压B遥遥无期B	遥控器B遗忘B遗产B道博B遇上B逮捕B	造船厂B造型B造势B速冲B速B通往B通关B递延B逐一B透水B逆变B适房B适应B退还B	退税率B追高B追随B追逐B追送B追溯B迫降B迫切B迫B迪拜B迟缓B迟报B连锁反应B连线B连接B连平B连带责任B远逊B	远超板B	进口量B还要B还债B	近几年B运维B	运煤船B运城B迎新B迈科B迈B过快B过分B过于B达成B边控B	输变电B轻装上阵B轻易B软肋B轮轴B轮流B轮回B转购B转给B转淡B车牌B车商B	车保帅B	躲猫猫B身兼B	身份证B踩线B踏空B踏板B踏上B	跷跷板B践B跨省B跨桥B跟着B距B跑车B跌穿B跌出B跃居B跃升B趴B趟B超配B	超常规B起锚B起价B	赵汉忠B走平B赛马B赛B	赔偿款B赏B	资金流B	贿赂案B贾康B贷款风险B贷B贵过B	购雷普B购进B购自B购油B贪B货款B货仓B败局B财经频道B财政收入B财务人员B贝恩B豪B谷B谱B	谭华杰B	谢百三B谜题B谋建B谋利B调查小组B	调味品B谁动B课B诸城B说谎B	说漏嘴B诱发B	误操作B误伤B误传B诚通B诚品B	试运行B试图B诉判B	评审团B证券时报B访谈实录B	设计院B设局B论证B论市B	讲故事B讲座B订金B计费B警钟B誓言B触电B触动B角力B角B觉醒B觉察B视频会议B视听B规矩B规律B覆辙B西单B表现出色B表情B补缺B补救B	补偿款B衣服B衡水B行程B行政复议B血本无归B	血制品B蠢蠢欲动B螺B	融资难B蝶阀B蝶变B蜂蜜B蜂拥而至B蛰伏B虫草B虚设B虚惊一场B虚实B虎虎生威B虎B蕴藏B	蔡宇清B	蔚少辉B蓄力B蒜B落选B营销整合B营造B萎靡不振B获补B药监B药用B药方B药厂B荣膺B荣波B荆棘丛生B茶B	茅台镇B	范冰冰B英才B英大B苏酒B花落B节能产品B节油B	艺术家B	艺术品B艺人B航钛B舒适B舆论监督B	致癌物B致癌B至关重要B自贡B自发B	腐败案B脱手B脱开B脚步B胶州B	胡茂元B胞兄B胜诉B胎动B育种B股补B股票投资B股海B股期B股息B股堪B股中B肝脏B肝素B	肉制品B肇事B职场B职位B耐火B考生B	老龄化B老挝B老婆B翅膀B群殴B群众B美国公司B署B罗沙B网点B网传B网上商城B缺货B缺油B缴纳B缘由B缔造B缄默B	综合性B续演B继B绞杀B绝话B绝缘B绝密B绝大部分B绝境B绝佳B绕道B	结算价B经销B经营风险B	经营权B经营收入B经纪B	经理人B	经济体B经幢B终成B细说B线路B线缆B线索B线图B	纺织业B纷呈B纵容B红旗B红军B红光B纠缠B紧随B紧逼B	索赔案B系统集成B系因B	糖酒会B糖业B粉针B粉丝B类似B	米其林B簇拥B箭在弦上B管终B答问B第十二届B	第二日B	第九届B	第三类B	第七届B笑纳B笑称B笑傲B章鱼B	章宏斌B	章子怡B竟然B竞合B竞争对手B竖线B窟窿B窄B	空置房B空置B空穴来风B空域B	空城计B	稳定性B稳增B稳固B税负B稍超B稀有金属B移至B称谓B称系B称独B称暂B称受B积累B	秦海璐B秦晓B租金B租车B科技成果B秋冬B离境B福原B祸起B祸及B神龙B神股B	祁玉民B社长B	社科院B礼包B磷矿B磁B硝烟弥漫B研讨B砍价B矿产资源B石渣B石波B石头B短纤B知情B瞠目结舌B瞄上B眼光B真诚B	真功夫B真假B看重B省级B	相结合B相约B相撞B相同B相似B相亲B	直营店B直扑B盯紧B盛会B盘涨B盘式B盘局B盘子B盘升B盗B盖茨B盖楼B	监管局B监理B益华B盈预B盈利模式B	百货类B	百富榜B百名B百只B	白酒业B	白菜价B登月B癌症B疲态B疯子B疑遭B疑未B疑有B疑是B疑惑B留有B留任B留人B男篮B电讯B电荒B	电磁炉B电梯B电工B	申万称B甩货B用工B生猛B生存之道B	生命线B	生力军B甜头B甚B甄别B瑞信B理论B	理事会B珠宝B珠B珍惜B	现阶段B	现货价B现售B现任B王毅B	王家岭B玉柴B猜B	独立性B独有B狂澜B犹如B	特大型B特价机票B特价B牵头B物超所值B牟利B牛黄B	牛磺酸B版图B父母B父亲B爆裂B	爆炸案B爆仓B熊鲜B	熊维平B焦虑B焊割B烯B热烈B	热像仪B烫伤B烧B烟草B点迭B点石成金B点位B	炸药包B炭素B炒金B炒楼B灿烂B灵活B激素B澳矿B潜行B潜心B漫漫B漫步B漫天要价B演进B演练B	演变成B演变B漏洞百出B	满洲里B满城B滞胀B游戏规则B港区B港务B温岭B	温家宝B	渠道商B渔船B渐远B清扬B清华B	添加剂B深夜B	深发展B淘米B淀粉B液压B	液化气B涨涨B涟钢B涛涛B涉黄B涉原B涉信B消灭B	海珍品B	海拉尔B海底B海岛B海军B	海内外B浩然B浦江B浙系B浙报B浙大B派送B	洽谈会B	活雷锋B	活跃度B活板B活B洪都B洪涝灾害B洪峰B洪因B津B洗浴B洗B	泼冷水B泰康B泰国B泰亚B注重B注水B注册商标B波折B沾B治疗B治污B油库B油井B沸腾B	河南省B沦落B没说B沙洲B沙尘B沈城B	沃达丰B	沃尔玛B汽车玻璃B江西省委B江煤B江津B汛情B汇理B求生B永丰B永B水域B水产B水上B氛围B气量B气候变化B民营企业B民润B	民政局B民安B	毛主席B比增B母亲B死罪B此起彼伏B此前B正极B歌华B欢迎B次终B欠税B欠B模块B梦魇B	梦工厂B档次B案发B桃子B格斗B核阀B校车B校友B树B标段B	柴达木B柜台B染红B果真B构想B极端B杰夫B杯具B来水B	李莉称B	李绍德B李昆B李家B	李国旺B权威人士B杂志B杀猪B机长B机器B本科B	本月底B	末班车B末期B未结B	未确定B未批B未定B未及B未予B期金B期货经纪B	期货价B期货交易B	期货业B期盼B期现B期望B期内B朝阳B朝华B望超B望成B服装行业B服务行业B有毒B	有机硅B有机B月初B	最高奖B最重B最美B	最大值B最优B替B	曼哈顿B更大B更具B曝出B暴雪B暴怒B暴B暗送B暗箱B暂告B	智囊团B智B景洪B普明B晨讯
??
Const_5Const*
_output_shapes	
:?N*
dtype0	*??
value??B??	?N"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_194080
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_<lambda>_194085
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?;
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
?
layer_with_weights-0
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures*
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
'
1
2
3
 4
!5*
'
0
1
2
 3
!4*
* 
?
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

'serving_default* 
;
(_lookup_layer
)	keras_api
*_adapt_function*
?

embeddings
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
?

kernel
bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
?

 kernel
!bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses*
?
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem{m|m} m~!mv?v?v? v?!v?*
'
1
2
3
 4
!5*
'
0
1
2
 3
!4*
* 
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEembedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_16/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_16/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_17/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_17/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 
* 
* 
7
Rlookup_table
Stoken_counts
T	keras_api*
* 
* 

0*

0*
* 
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 

 0
!1*

 0
!1*
* 
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUE	Adam/iter>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE
Adam/decay?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

i0
j1*
* 
* 
* 
* 
* 
* 
* 
R
k_initializer
l_create_resource
m_initialize
n_destroy_resource* 
?
o_create_resource
p_initialize
q_destroy_resource_
tableVlayer_with_weights-0/layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	rtotal
	scount
t	variables
u	keras_api*
H
	vtotal
	wcount
x
_fn_kwargs
y	variables
z	keras_api*
* 
* 
* 
* 
* 
* 
* 
hb
VARIABLE_VALUEtotalIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEcountIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

r0
s1*

t	variables*
jd
VARIABLE_VALUEtotal_1Ilayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_1Ilayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

v0
w1*

y	variables*
??
VARIABLE_VALUEAdam/embedding/embeddings/mWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_16/kernel/mWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/dense_16/bias/mWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_17/kernel/mWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/dense_17/bias/mWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/embedding/embeddings/vWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_16/kernel/vWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/dense_16/bias/vWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/dense_17/kernel/vWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUEAdam/dense_17/bias/vWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
#serving_default_sequential_11_inputPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCall#serving_default_sequential_11_input
hash_tableConstConst_1Const_2embedding/embeddingsdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_193694
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOpConst_6*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_194194
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateMutableHashTabletotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/mAdam/embedding/embeddings/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_194279??
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_194012

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?p
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193888

inputsS
Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_8_string_lookup_8_equal_y3
/text_vectorization_8_string_lookup_8_selectv2_t	4
!embedding_embedding_lookup_193867:	?N9
'dense_16_matmul_readvariableop_resource:6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?embedding/embedding_lookup?Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2\
 text_vectorization_8/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_8/StaticRegexReplaceStaticRegexReplace)text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_8/StringSplit/StringSplitV2StringSplitV20text_vectorization_8/StaticRegexReplace:output:0/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_8/StringSplit/strided_sliceStridedSlice8text_vectorization_8/StringSplit/StringSplitV2:indices:0=text_vectorization_8/StringSplit/strided_slice/stack:output:0?text_vectorization_8/StringSplit/strided_slice/stack_1:output:0?text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_8/StringSplit/strided_slice_1StridedSlice6text_vectorization_8/StringSplit/StringSplitV2:shape:0?text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handle7text_vectorization_8/StringSplit/StringSplitV2:values:0Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_8/string_lookup_8/EqualEqual7text_vectorization_8/StringSplit/StringSplitV2:values:0,text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/SelectV2SelectV2.text_vectorization_8/string_lookup_8/Equal:z:0/text_vectorization_8_string_lookup_8_selectv2_tKtext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/IdentityIdentity6text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_8/RaggedToTensor/Const:output:06text_vectorization_8/string_lookup_8/Identity:output:0:text_vectorization_8/RaggedToTensor/default_value:output:09text_vectorization_8/StringSplit/strided_slice_1:output:07text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_193867Atext_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/193867*+
_output_shapes
:?????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/193867*+
_output_shapes
:??????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????s
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_8/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_16/MatMulMatMul(global_average_pooling1d_8/Mean:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp^embedding/embedding_lookupC^text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

*__inference_embedding_layer_call_fn_193953

inputs	
unknown:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_192925s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193455
sequential_11_input
sequential_11_193434
sequential_11_193436	
sequential_11_193438
sequential_11_193440	'
sequential_11_193442:	?N&
sequential_11_193444:"
sequential_11_193446:&
sequential_11_193448:"
sequential_11_193450:
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallsequential_11_inputsequential_11_193434sequential_11_193436sequential_11_193438sequential_11_193440sequential_11_193442sequential_11_193444sequential_11_193446sequential_11_193448sequential_11_193450*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_192964?
softmax_4/PartitionedCallPartitionedCall.sequential_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_193308q
IdentityIdentity"softmax_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namesequential_11_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
$__inference_signature_wrapper_193694
sequential_11_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_192849o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namesequential_11_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
r
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_193973

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_softmax_4_layer_call_and_return_conditional_losses_193308

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_17_layer_call_fn_194002

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_192957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
.__inference_sequential_13_layer_call_fn_193431
sequential_11_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_193387o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namesequential_11_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?i
?
"__inference__traced_restore_194279
file_prefix8
%assignvariableop_embedding_embeddings:	?N4
"assignvariableop_1_dense_16_kernel:.
 assignvariableop_2_dense_16_bias:4
"assignvariableop_3_dense_17_kernel:.
 assignvariableop_4_dense_17_bias:&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: B
/assignvariableop_14_adam_embedding_embeddings_m:	?N<
*assignvariableop_15_adam_dense_16_kernel_m:6
(assignvariableop_16_adam_dense_16_bias_m:<
*assignvariableop_17_adam_dense_17_kernel_m:6
(assignvariableop_18_adam_dense_17_bias_m:B
/assignvariableop_19_adam_embedding_embeddings_v:	?N<
*assignvariableop_20_adam_dense_16_kernel_v:6
(assignvariableop_21_adam_dense_16_bias_v:<
*assignvariableop_22_adam_dense_17_kernel_v:6
(assignvariableop_23_adam_dense_17_bias_v:
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysB]layer_with_weights-0/layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_16_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense_16_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_17_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense_17_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:10RestoreV2:tensors:11*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 _
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_adam_embedding_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_16_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_16_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_17_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_17_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_adam_embedding_embeddings_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_16_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_16_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_17_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_17_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?<
?
__inference__traced_save_194194
file_prefix3
/savev2_embedding_embeddings_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysB]layer_with_weights-0/layer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesBIlayer_with_weights-0/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-0/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-0/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *)
dtypes
2		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?N::::: : : : : ::: : : : :	?N:::::	?N::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?N:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
;
__inference__creator_194017
identity??
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name120617*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
/
__inference__initializer_194040
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
.__inference_sequential_11_layer_call_fn_193147
text_vectorization_8_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_193103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_8_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
.__inference_sequential_11_layer_call_fn_193723

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_192964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?C
?
__inference_adapt_step_193946
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
ς
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193597

inputsa
]sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleb
^sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:sequential_11_text_vectorization_8_string_lookup_8_equal_yA
=sequential_11_text_vectorization_8_string_lookup_8_selectv2_t	B
/sequential_11_embedding_embedding_lookup_193575:	?NG
5sequential_11_dense_16_matmul_readvariableop_resource:D
6sequential_11_dense_16_biasadd_readvariableop_resource:G
5sequential_11_dense_17_matmul_readvariableop_resource:D
6sequential_11_dense_17_biasadd_readvariableop_resource:
identity??-sequential_11/dense_16/BiasAdd/ReadVariableOp?,sequential_11/dense_16/MatMul/ReadVariableOp?-sequential_11/dense_17/BiasAdd/ReadVariableOp?,sequential_11/dense_17/MatMul/ReadVariableOp?(sequential_11/embedding/embedding_lookup?Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2j
.sequential_11/text_vectorization_8/StringLowerStringLowerinputs*#
_output_shapes
:??????????
5sequential_11/text_vectorization_8/StaticRegexReplaceStaticRegexReplace7sequential_11/text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite u
4sequential_11/text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
<sequential_11/text_vectorization_8/StringSplit/StringSplitV2StringSplitV2>sequential_11/text_vectorization_8/StaticRegexReplace:output:0=sequential_11/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Bsequential_11/text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Dsequential_11/text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Dsequential_11/text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
<sequential_11/text_vectorization_8/StringSplit/strided_sliceStridedSliceFsequential_11/text_vectorization_8/StringSplit/StringSplitV2:indices:0Ksequential_11/text_vectorization_8/StringSplit/strided_slice/stack:output:0Msequential_11/text_vectorization_8/StringSplit/strided_slice/stack_1:output:0Msequential_11/text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Dsequential_11/text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_11/text_vectorization_8/StringSplit/strided_slice_1StridedSliceDsequential_11/text_vectorization_8/StringSplit/StringSplitV2:shape:0Msequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Osequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Osequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
esequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastEsequential_11/text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
gsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastGsequential_11/text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
osequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeisequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
osequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
nsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdxsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0xsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ssequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterwsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0|sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
nsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastusequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxisequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
osequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2vsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0xsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulrsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumksequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumksequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
rsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountisequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0zsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
lsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
gsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumysequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
psequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
lsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
gsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ysequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2]sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleEsequential_11/text_vectorization_8/StringSplit/StringSplitV2:values:0^sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8sequential_11/text_vectorization_8/string_lookup_8/EqualEqualEsequential_11/text_vectorization_8/StringSplit/StringSplitV2:values:0:sequential_11_text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
;sequential_11/text_vectorization_8/string_lookup_8/SelectV2SelectV2<sequential_11/text_vectorization_8/string_lookup_8/Equal:z:0=sequential_11_text_vectorization_8_string_lookup_8_selectv2_tYsequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
;sequential_11/text_vectorization_8/string_lookup_8/IdentityIdentityDsequential_11/text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
?sequential_11/text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
7sequential_11/text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
Fsequential_11/text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor@sequential_11/text_vectorization_8/RaggedToTensor/Const:output:0Dsequential_11/text_vectorization_8/string_lookup_8/Identity:output:0Hsequential_11/text_vectorization_8/RaggedToTensor/default_value:output:0Gsequential_11/text_vectorization_8/StringSplit/strided_slice_1:output:0Esequential_11/text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
(sequential_11/embedding/embedding_lookupResourceGather/sequential_11_embedding_embedding_lookup_193575Osequential_11/text_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*B
_class8
64loc:@sequential_11/embedding/embedding_lookup/193575*+
_output_shapes
:?????????*
dtype0?
1sequential_11/embedding/embedding_lookup/IdentityIdentity1sequential_11/embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_11/embedding/embedding_lookup/193575*+
_output_shapes
:??????????
3sequential_11/embedding/embedding_lookup/Identity_1Identity:sequential_11/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
?sequential_11/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
-sequential_11/global_average_pooling1d_8/MeanMean<sequential_11/embedding/embedding_lookup/Identity_1:output:0Hsequential_11/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
,sequential_11/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_11/dense_16/MatMulMatMul6sequential_11/global_average_pooling1d_8/Mean:output:04sequential_11/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_11/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/dense_16/BiasAddBiasAdd'sequential_11/dense_16/MatMul:product:05sequential_11/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
sequential_11/dense_16/ReluRelu'sequential_11/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
,sequential_11/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_11/dense_17/MatMulMatMul)sequential_11/dense_16/Relu:activations:04sequential_11/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_11/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/dense_17/BiasAddBiasAdd'sequential_11/dense_17/MatMul:product:05sequential_11/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
softmax_4/SoftmaxSoftmax'sequential_11/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitysoftmax_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^sequential_11/dense_16/BiasAdd/ReadVariableOp-^sequential_11/dense_16/MatMul/ReadVariableOp.^sequential_11/dense_17/BiasAdd/ReadVariableOp-^sequential_11/dense_17/MatMul/ReadVariableOp)^sequential_11/embedding/embedding_lookupQ^sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2^
-sequential_11/dense_16/BiasAdd/ReadVariableOp-sequential_11/dense_16/BiasAdd/ReadVariableOp2\
,sequential_11/dense_16/MatMul/ReadVariableOp,sequential_11/dense_16/MatMul/ReadVariableOp2^
-sequential_11/dense_17/BiasAdd/ReadVariableOp-sequential_11/dense_17/BiasAdd/ReadVariableOp2\
,sequential_11/dense_17/MatMul/ReadVariableOp,sequential_11/dense_17/MatMul/ReadVariableOp2T
(sequential_11/embedding/embedding_lookup(sequential_11/embedding/embedding_lookup2?
Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?i
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193277
text_vectorization_8_inputS
Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_8_string_lookup_8_equal_y3
/text_vectorization_8_string_lookup_8_selectv2_t	#
embedding_193262:	?N!
dense_16_193266:
dense_16_193268:!
dense_17_193271:
dense_17_193273:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2p
 text_vectorization_8/StringLowerStringLowertext_vectorization_8_input*#
_output_shapes
:??????????
'text_vectorization_8/StaticRegexReplaceStaticRegexReplace)text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_8/StringSplit/StringSplitV2StringSplitV20text_vectorization_8/StaticRegexReplace:output:0/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_8/StringSplit/strided_sliceStridedSlice8text_vectorization_8/StringSplit/StringSplitV2:indices:0=text_vectorization_8/StringSplit/strided_slice/stack:output:0?text_vectorization_8/StringSplit/strided_slice/stack_1:output:0?text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_8/StringSplit/strided_slice_1StridedSlice6text_vectorization_8/StringSplit/StringSplitV2:shape:0?text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handle7text_vectorization_8/StringSplit/StringSplitV2:values:0Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_8/string_lookup_8/EqualEqual7text_vectorization_8/StringSplit/StringSplitV2:values:0,text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/SelectV2SelectV2.text_vectorization_8/string_lookup_8/Equal:z:0/text_vectorization_8_string_lookup_8_selectv2_tKtext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/IdentityIdentity6text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_8/RaggedToTensor/Const:output:06text_vectorization_8/string_lookup_8/Identity:output:0:text_vectorization_8/RaggedToTensor/default_value:output:09text_vectorization_8/StringSplit/strided_slice_1:output:07text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0embedding_193262*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_192925?
*global_average_pooling1d_8/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_192859?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_16_193266dense_16_193268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_192941?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_193271dense_17_193273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_192957x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^embedding/StatefulPartitionedCallC^text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_8_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_192925

inputs	*
embedding_lookup_192919:	?N
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_192919inputs*
Tindices0	**
_class 
loc:@embedding_lookup/192919*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/192919*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
r
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_192859

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193311

inputs
sequential_11_193284
sequential_11_193286	
sequential_11_193288
sequential_11_193290	'
sequential_11_193292:	?N&
sequential_11_193294:"
sequential_11_193296:&
sequential_11_193298:"
sequential_11_193300:
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_193284sequential_11_193286sequential_11_193288sequential_11_193290sequential_11_193292sequential_11_193294sequential_11_193296sequential_11_193298sequential_11_193300*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_192964?
softmax_4/PartitionedCallPartitionedCall.sequential_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_193308q
IdentityIdentity"softmax_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
a
E__inference_softmax_4_layer_call_and_return_conditional_losses_193898

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
-
__inference__destroyer_194045
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
)__inference_dense_16_layer_call_fn_193982

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_192941o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_1940259
5key_value_init120616_lookuptableimportv2_table_handle1
-key_value_init120616_lookuptableimportv2_keys3
/key_value_init120616_lookuptableimportv2_values	
identity??(key_value_init120616/LookupTableImportV2?
(key_value_init120616/LookupTableImportV2LookupTableImportV25key_value_init120616_lookuptableimportv2_table_handle-key_value_init120616_lookuptableimportv2_keys/key_value_init120616_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^key_value_init120616/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2T
(key_value_init120616/LookupTableImportV2(key_value_init120616/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?

?
.__inference_sequential_11_layer_call_fn_192985
text_vectorization_8_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_192964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_8_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
F
*__inference_softmax_4_layer_call_fn_193893

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_193308`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
__inference__creator_194035
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_116073*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?i
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193212
text_vectorization_8_inputS
Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_8_string_lookup_8_equal_y3
/text_vectorization_8_string_lookup_8_selectv2_t	#
embedding_193197:	?N!
dense_16_193201:
dense_16_193203:!
dense_17_193206:
dense_17_193208:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2p
 text_vectorization_8/StringLowerStringLowertext_vectorization_8_input*#
_output_shapes
:??????????
'text_vectorization_8/StaticRegexReplaceStaticRegexReplace)text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_8/StringSplit/StringSplitV2StringSplitV20text_vectorization_8/StaticRegexReplace:output:0/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_8/StringSplit/strided_sliceStridedSlice8text_vectorization_8/StringSplit/StringSplitV2:indices:0=text_vectorization_8/StringSplit/strided_slice/stack:output:0?text_vectorization_8/StringSplit/strided_slice/stack_1:output:0?text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_8/StringSplit/strided_slice_1StridedSlice6text_vectorization_8/StringSplit/StringSplitV2:shape:0?text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handle7text_vectorization_8/StringSplit/StringSplitV2:values:0Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_8/string_lookup_8/EqualEqual7text_vectorization_8/StringSplit/StringSplitV2:values:0,text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/SelectV2SelectV2.text_vectorization_8/string_lookup_8/Equal:z:0/text_vectorization_8_string_lookup_8_selectv2_tKtext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/IdentityIdentity6text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_8/RaggedToTensor/Const:output:06text_vectorization_8/string_lookup_8/Identity:output:0:text_vectorization_8/RaggedToTensor/default_value:output:09text_vectorization_8/StringSplit/strided_slice_1:output:07text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0embedding_193197*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_192925?
*global_average_pooling1d_8/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_192859?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_16_193201dense_16_193203*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_192941?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_193206dense_17_193208*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_192957x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^embedding/StatefulPartitionedCallC^text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:_ [
#
_output_shapes
:?????????
4
_user_specified_nametext_vectorization_8_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
-
__inference__destroyer_194030
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
.__inference_sequential_13_layer_call_fn_193502

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_193311o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
.__inference_sequential_13_layer_call_fn_193525

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_193387o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_dense_16_layer_call_and_return_conditional_losses_193993

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
.__inference_sequential_11_layer_call_fn_193746

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_193103o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_192957

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_embedding_layer_call_and_return_conditional_losses_193962

inputs	*
embedding_lookup_193956:	?N
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_193956inputs*
Tindices0	**
_class 
loc:@embedding_lookup/193956*+
_output_shapes
:?????????*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/193956*+
_output_shapes
:??????????
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ς
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193669

inputsa
]sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleb
^sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:sequential_11_text_vectorization_8_string_lookup_8_equal_yA
=sequential_11_text_vectorization_8_string_lookup_8_selectv2_t	B
/sequential_11_embedding_embedding_lookup_193647:	?NG
5sequential_11_dense_16_matmul_readvariableop_resource:D
6sequential_11_dense_16_biasadd_readvariableop_resource:G
5sequential_11_dense_17_matmul_readvariableop_resource:D
6sequential_11_dense_17_biasadd_readvariableop_resource:
identity??-sequential_11/dense_16/BiasAdd/ReadVariableOp?,sequential_11/dense_16/MatMul/ReadVariableOp?-sequential_11/dense_17/BiasAdd/ReadVariableOp?,sequential_11/dense_17/MatMul/ReadVariableOp?(sequential_11/embedding/embedding_lookup?Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2j
.sequential_11/text_vectorization_8/StringLowerStringLowerinputs*#
_output_shapes
:??????????
5sequential_11/text_vectorization_8/StaticRegexReplaceStaticRegexReplace7sequential_11/text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite u
4sequential_11/text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
<sequential_11/text_vectorization_8/StringSplit/StringSplitV2StringSplitV2>sequential_11/text_vectorization_8/StaticRegexReplace:output:0=sequential_11/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Bsequential_11/text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Dsequential_11/text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Dsequential_11/text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
<sequential_11/text_vectorization_8/StringSplit/strided_sliceStridedSliceFsequential_11/text_vectorization_8/StringSplit/StringSplitV2:indices:0Ksequential_11/text_vectorization_8/StringSplit/strided_slice/stack:output:0Msequential_11/text_vectorization_8/StringSplit/strided_slice/stack_1:output:0Msequential_11/text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Dsequential_11/text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Fsequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential_11/text_vectorization_8/StringSplit/strided_slice_1StridedSliceDsequential_11/text_vectorization_8/StringSplit/StringSplitV2:shape:0Msequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Osequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Osequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
esequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastEsequential_11/text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
gsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastGsequential_11/text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
osequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeisequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
osequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
nsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdxsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0xsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ssequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterwsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0|sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
nsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastusequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxisequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0zsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
osequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2vsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0xsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulrsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumksequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumksequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
qsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
rsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountisequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0zsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
lsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
gsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumysequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
psequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
lsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
gsequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ysequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0msequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0usequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2]sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleEsequential_11/text_vectorization_8/StringSplit/StringSplitV2:values:0^sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
8sequential_11/text_vectorization_8/string_lookup_8/EqualEqualEsequential_11/text_vectorization_8/StringSplit/StringSplitV2:values:0:sequential_11_text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
;sequential_11/text_vectorization_8/string_lookup_8/SelectV2SelectV2<sequential_11/text_vectorization_8/string_lookup_8/Equal:z:0=sequential_11_text_vectorization_8_string_lookup_8_selectv2_tYsequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
;sequential_11/text_vectorization_8/string_lookup_8/IdentityIdentityDsequential_11/text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
?sequential_11/text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
7sequential_11/text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
Fsequential_11/text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor@sequential_11/text_vectorization_8/RaggedToTensor/Const:output:0Dsequential_11/text_vectorization_8/string_lookup_8/Identity:output:0Hsequential_11/text_vectorization_8/RaggedToTensor/default_value:output:0Gsequential_11/text_vectorization_8/StringSplit/strided_slice_1:output:0Esequential_11/text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
(sequential_11/embedding/embedding_lookupResourceGather/sequential_11_embedding_embedding_lookup_193647Osequential_11/text_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*B
_class8
64loc:@sequential_11/embedding/embedding_lookup/193647*+
_output_shapes
:?????????*
dtype0?
1sequential_11/embedding/embedding_lookup/IdentityIdentity1sequential_11/embedding/embedding_lookup:output:0*
T0*B
_class8
64loc:@sequential_11/embedding/embedding_lookup/193647*+
_output_shapes
:??????????
3sequential_11/embedding/embedding_lookup/Identity_1Identity:sequential_11/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
?sequential_11/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
-sequential_11/global_average_pooling1d_8/MeanMean<sequential_11/embedding/embedding_lookup/Identity_1:output:0Hsequential_11/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
,sequential_11/dense_16/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_11/dense_16/MatMulMatMul6sequential_11/global_average_pooling1d_8/Mean:output:04sequential_11/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_11/dense_16/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/dense_16/BiasAddBiasAdd'sequential_11/dense_16/MatMul:product:05sequential_11/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
sequential_11/dense_16/ReluRelu'sequential_11/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
,sequential_11/dense_17/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential_11/dense_17/MatMulMatMul)sequential_11/dense_16/Relu:activations:04sequential_11/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_11/dense_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_11/dense_17/BiasAddBiasAdd'sequential_11/dense_17/MatMul:product:05sequential_11/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????w
softmax_4/SoftmaxSoftmax'sequential_11/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentitysoftmax_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^sequential_11/dense_16/BiasAdd/ReadVariableOp-^sequential_11/dense_16/MatMul/ReadVariableOp.^sequential_11/dense_17/BiasAdd/ReadVariableOp-^sequential_11/dense_17/MatMul/ReadVariableOp)^sequential_11/embedding/embedding_lookupQ^sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2^
-sequential_11/dense_16/BiasAdd/ReadVariableOp-sequential_11/dense_16/BiasAdd/ReadVariableOp2\
,sequential_11/dense_16/MatMul/ReadVariableOp,sequential_11/dense_16/MatMul/ReadVariableOp2^
-sequential_11/dense_17/BiasAdd/ReadVariableOp-sequential_11/dense_17/BiasAdd/ReadVariableOp2\
,sequential_11/dense_17/MatMul/ReadVariableOp,sequential_11/dense_17/MatMul/ReadVariableOp2T
(sequential_11/embedding/embedding_lookup(sequential_11/embedding/embedding_lookup2?
Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Psequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?i
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_192964

inputsS
Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_8_string_lookup_8_equal_y3
/text_vectorization_8_string_lookup_8_selectv2_t	#
embedding_192926:	?N!
dense_16_192942:
dense_16_192944:!
dense_17_192958:
dense_17_192960:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2\
 text_vectorization_8/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_8/StaticRegexReplaceStaticRegexReplace)text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_8/StringSplit/StringSplitV2StringSplitV20text_vectorization_8/StaticRegexReplace:output:0/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_8/StringSplit/strided_sliceStridedSlice8text_vectorization_8/StringSplit/StringSplitV2:indices:0=text_vectorization_8/StringSplit/strided_slice/stack:output:0?text_vectorization_8/StringSplit/strided_slice/stack_1:output:0?text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_8/StringSplit/strided_slice_1StridedSlice6text_vectorization_8/StringSplit/StringSplitV2:shape:0?text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handle7text_vectorization_8/StringSplit/StringSplitV2:values:0Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_8/string_lookup_8/EqualEqual7text_vectorization_8/StringSplit/StringSplitV2:values:0,text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/SelectV2SelectV2.text_vectorization_8/string_lookup_8/Equal:z:0/text_vectorization_8_string_lookup_8_selectv2_tKtext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/IdentityIdentity6text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_8/RaggedToTensor/Const:output:06text_vectorization_8/string_lookup_8/Identity:output:0:text_vectorization_8/RaggedToTensor/default_value:output:09text_vectorization_8/StringSplit/strided_slice_1:output:07text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0embedding_192926*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_192925?
*global_average_pooling1d_8/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_192859?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_16_192942dense_16_192944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_192941?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_192958dense_17_192960*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_192957x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^embedding/StatefulPartitionedCallC^text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?p
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193817

inputsS
Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_8_string_lookup_8_equal_y3
/text_vectorization_8_string_lookup_8_selectv2_t	4
!embedding_embedding_lookup_193796:	?N9
'dense_16_matmul_readvariableop_resource:6
(dense_16_biasadd_readvariableop_resource:9
'dense_17_matmul_readvariableop_resource:6
(dense_17_biasadd_readvariableop_resource:
identity??dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?embedding/embedding_lookup?Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2\
 text_vectorization_8/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_8/StaticRegexReplaceStaticRegexReplace)text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_8/StringSplit/StringSplitV2StringSplitV20text_vectorization_8/StaticRegexReplace:output:0/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_8/StringSplit/strided_sliceStridedSlice8text_vectorization_8/StringSplit/StringSplitV2:indices:0=text_vectorization_8/StringSplit/strided_slice/stack:output:0?text_vectorization_8/StringSplit/strided_slice/stack_1:output:0?text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_8/StringSplit/strided_slice_1StridedSlice6text_vectorization_8/StringSplit/StringSplitV2:shape:0?text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handle7text_vectorization_8/StringSplit/StringSplitV2:values:0Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_8/string_lookup_8/EqualEqual7text_vectorization_8/StringSplit/StringSplitV2:values:0,text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/SelectV2SelectV2.text_vectorization_8/string_lookup_8/Equal:z:0/text_vectorization_8_string_lookup_8_selectv2_tKtext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/IdentityIdentity6text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_8/RaggedToTensor/Const:output:06text_vectorization_8/string_lookup_8/Identity:output:0:text_vectorization_8/RaggedToTensor/default_value:output:09text_vectorization_8/StringSplit/strided_slice_1:output:07text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_193796Atext_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding/embedding_lookup/193796*+
_output_shapes
:?????????*
dtype0?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/193796*+
_output_shapes
:??????????
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????s
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_average_pooling1d_8/MeanMean.embedding/embedding_lookup/Identity_1:output:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_16/MatMulMatMul(global_average_pooling1d_8/Mean:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp^embedding/embedding_lookupC^text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2?
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
+
__inference_<lambda>_194085
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?i
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193103

inputsS
Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleT
Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	0
,text_vectorization_8_string_lookup_8_equal_y3
/text_vectorization_8_string_lookup_8_selectv2_t	#
embedding_193088:	?N!
dense_16_193092:
dense_16_193094:!
dense_17_193097:
dense_17_193099:
identity?? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2\
 text_vectorization_8/StringLowerStringLowerinputs*#
_output_shapes
:??????????
'text_vectorization_8/StaticRegexReplaceStaticRegexReplace)text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite g
&text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
.text_vectorization_8/StringSplit/StringSplitV2StringSplitV20text_vectorization_8/StaticRegexReplace:output:0/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
4text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
6text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
6text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
.text_vectorization_8/StringSplit/strided_sliceStridedSlice8text_vectorization_8/StringSplit/StringSplitV2:indices:0=text_vectorization_8/StringSplit/strided_slice/stack:output:0?text_vectorization_8/StringSplit/strided_slice/stack_1:output:0?text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
6text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
8text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
8text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
0text_vectorization_8/StringSplit/strided_slice_1StridedSlice6text_vectorization_8/StringSplit/StringSplitV2:shape:0?text_vectorization_8/StringSplit/strided_slice_1/stack:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0Atext_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Wtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast7text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast9text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShape[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdjtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
etext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreateritext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ntext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
`text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastgtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMax[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
atext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2htext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0jtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMuldtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum]text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
ctext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
dtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincount[text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ltext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
btext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
^text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Ytext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2ktext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0_text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0gtext_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Otext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handle7text_vectorization_8/StringSplit/StringSplitV2:values:0Ptext_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
*text_vectorization_8/string_lookup_8/EqualEqual7text_vectorization_8/StringSplit/StringSplitV2:values:0,text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/SelectV2SelectV2.text_vectorization_8/string_lookup_8/Equal:z:0/text_vectorization_8_string_lookup_8_selectv2_tKtext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
-text_vectorization_8/string_lookup_8/IdentityIdentity6text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:?????????s
1text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
)text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
8text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor2text_vectorization_8/RaggedToTensor/Const:output:06text_vectorization_8/string_lookup_8/Identity:output:0:text_vectorization_8/RaggedToTensor/default_value:output:09text_vectorization_8/StringSplit/strided_slice_1:output:07text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
!embedding/StatefulPartitionedCallStatefulPartitionedCallAtext_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0embedding_193088*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_192925?
*global_average_pooling1d_8/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_192859?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0dense_16_193092dense_16_193094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_192941?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_193097dense_17_193099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_192957x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^embedding/StatefulPartitionedCallC^text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2?
Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2Btext_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Δ
?

!__inference__wrapped_model_192849
sequential_11_inputo
ksequential_13_sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handlep
lsequential_13_sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value	L
Hsequential_13_sequential_11_text_vectorization_8_string_lookup_8_equal_yO
Ksequential_13_sequential_11_text_vectorization_8_string_lookup_8_selectv2_t	P
=sequential_13_sequential_11_embedding_embedding_lookup_192827:	?NU
Csequential_13_sequential_11_dense_16_matmul_readvariableop_resource:R
Dsequential_13_sequential_11_dense_16_biasadd_readvariableop_resource:U
Csequential_13_sequential_11_dense_17_matmul_readvariableop_resource:R
Dsequential_13_sequential_11_dense_17_biasadd_readvariableop_resource:
identity??;sequential_13/sequential_11/dense_16/BiasAdd/ReadVariableOp?:sequential_13/sequential_11/dense_16/MatMul/ReadVariableOp?;sequential_13/sequential_11/dense_17/BiasAdd/ReadVariableOp?:sequential_13/sequential_11/dense_17/MatMul/ReadVariableOp?6sequential_13/sequential_11/embedding/embedding_lookup?^sequential_13/sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2?
<sequential_13/sequential_11/text_vectorization_8/StringLowerStringLowersequential_11_input*#
_output_shapes
:??????????
Csequential_13/sequential_11/text_vectorization_8/StaticRegexReplaceStaticRegexReplaceEsequential_13/sequential_11/text_vectorization_8/StringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
Bsequential_13/sequential_11/text_vectorization_8/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
Jsequential_13/sequential_11/text_vectorization_8/StringSplit/StringSplitV2StringSplitV2Lsequential_13/sequential_11/text_vectorization_8/StaticRegexReplace:output:0Ksequential_13/sequential_11/text_vectorization_8/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
Psequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Rsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Rsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Jsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_sliceStridedSliceTsequential_13/sequential_11/text_vectorization_8/StringSplit/StringSplitV2:indices:0Ysequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice/stack:output:0[sequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice/stack_1:output:0[sequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Rsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Tsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Tsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Lsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1StridedSliceRsequential_13/sequential_11/text_vectorization_8/StringSplit/StringSplitV2:shape:0[sequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack:output:0]sequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_1:output:0]sequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
ssequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastSsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
usequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastUsequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
}sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapewsequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
}sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
|sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProd?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreater?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
|sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCast?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
{sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxwsequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
}sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
{sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
{sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMul?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumysequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumysequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountwsequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
zsequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
usequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
~sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
zsequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
usequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0{sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0?sequential_13/sequential_11/text_vectorization_8/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
^sequential_13/sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2ksequential_13_sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_table_handleSsequential_13/sequential_11/text_vectorization_8/StringSplit/StringSplitV2:values:0lsequential_13_sequential_11_text_vectorization_8_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
Fsequential_13/sequential_11/text_vectorization_8/string_lookup_8/EqualEqualSsequential_13/sequential_11/text_vectorization_8/StringSplit/StringSplitV2:values:0Hsequential_13_sequential_11_text_vectorization_8_string_lookup_8_equal_y*
T0*#
_output_shapes
:??????????
Isequential_13/sequential_11/text_vectorization_8/string_lookup_8/SelectV2SelectV2Jsequential_13/sequential_11/text_vectorization_8/string_lookup_8/Equal:z:0Ksequential_13_sequential_11_text_vectorization_8_string_lookup_8_selectv2_tgsequential_13/sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Isequential_13/sequential_11/text_vectorization_8/string_lookup_8/IdentityIdentityRsequential_13/sequential_11/text_vectorization_8/string_lookup_8/SelectV2:output:0*
T0	*#
_output_shapes
:??????????
Msequential_13/sequential_11/text_vectorization_8/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Esequential_13/sequential_11/text_vectorization_8/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"????????       ?
Tsequential_13/sequential_11/text_vectorization_8/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensorNsequential_13/sequential_11/text_vectorization_8/RaggedToTensor/Const:output:0Rsequential_13/sequential_11/text_vectorization_8/string_lookup_8/Identity:output:0Vsequential_13/sequential_11/text_vectorization_8/RaggedToTensor/default_value:output:0Usequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice_1:output:0Ssequential_13/sequential_11/text_vectorization_8/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:?????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
6sequential_13/sequential_11/embedding/embedding_lookupResourceGather=sequential_13_sequential_11_embedding_embedding_lookup_192827]sequential_13/sequential_11/text_vectorization_8/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*P
_classF
DBloc:@sequential_13/sequential_11/embedding/embedding_lookup/192827*+
_output_shapes
:?????????*
dtype0?
?sequential_13/sequential_11/embedding/embedding_lookup/IdentityIdentity?sequential_13/sequential_11/embedding/embedding_lookup:output:0*
T0*P
_classF
DBloc:@sequential_13/sequential_11/embedding/embedding_lookup/192827*+
_output_shapes
:??????????
Asequential_13/sequential_11/embedding/embedding_lookup/Identity_1IdentityHsequential_13/sequential_11/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:??????????
Msequential_13/sequential_11/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
;sequential_13/sequential_11/global_average_pooling1d_8/MeanMeanJsequential_13/sequential_11/embedding/embedding_lookup/Identity_1:output:0Vsequential_13/sequential_11/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:??????????
:sequential_13/sequential_11/dense_16/MatMul/ReadVariableOpReadVariableOpCsequential_13_sequential_11_dense_16_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
+sequential_13/sequential_11/dense_16/MatMulMatMulDsequential_13/sequential_11/global_average_pooling1d_8/Mean:output:0Bsequential_13/sequential_11/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;sequential_13/sequential_11/dense_16/BiasAdd/ReadVariableOpReadVariableOpDsequential_13_sequential_11_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
,sequential_13/sequential_11/dense_16/BiasAddBiasAdd5sequential_13/sequential_11/dense_16/MatMul:product:0Csequential_13/sequential_11/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)sequential_13/sequential_11/dense_16/ReluRelu5sequential_13/sequential_11/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
:sequential_13/sequential_11/dense_17/MatMul/ReadVariableOpReadVariableOpCsequential_13_sequential_11_dense_17_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
+sequential_13/sequential_11/dense_17/MatMulMatMul7sequential_13/sequential_11/dense_16/Relu:activations:0Bsequential_13/sequential_11/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
;sequential_13/sequential_11/dense_17/BiasAdd/ReadVariableOpReadVariableOpDsequential_13_sequential_11_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
,sequential_13/sequential_11/dense_17/BiasAddBiasAdd5sequential_13/sequential_11/dense_17/MatMul:product:0Csequential_13/sequential_11/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_13/softmax_4/SoftmaxSoftmax5sequential_13/sequential_11/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x
IdentityIdentity)sequential_13/softmax_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp<^sequential_13/sequential_11/dense_16/BiasAdd/ReadVariableOp;^sequential_13/sequential_11/dense_16/MatMul/ReadVariableOp<^sequential_13/sequential_11/dense_17/BiasAdd/ReadVariableOp;^sequential_13/sequential_11/dense_17/MatMul/ReadVariableOp7^sequential_13/sequential_11/embedding/embedding_lookup_^sequential_13/sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2z
;sequential_13/sequential_11/dense_16/BiasAdd/ReadVariableOp;sequential_13/sequential_11/dense_16/BiasAdd/ReadVariableOp2x
:sequential_13/sequential_11/dense_16/MatMul/ReadVariableOp:sequential_13/sequential_11/dense_16/MatMul/ReadVariableOp2z
;sequential_13/sequential_11/dense_17/BiasAdd/ReadVariableOp;sequential_13/sequential_11/dense_17/BiasAdd/ReadVariableOp2x
:sequential_13/sequential_11/dense_17/MatMul/ReadVariableOp:sequential_13/sequential_11/dense_17/MatMul/ReadVariableOp2p
6sequential_13/sequential_11/embedding/embedding_lookup6sequential_13/sequential_11/embedding/embedding_lookup2?
^sequential_13/sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2^sequential_13/sequential_11/text_vectorization_8/string_lookup_8/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namesequential_11_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193479
sequential_11_input
sequential_11_193458
sequential_11_193460	
sequential_11_193462
sequential_11_193464	'
sequential_11_193466:	?N&
sequential_11_193468:"
sequential_11_193470:&
sequential_11_193472:"
sequential_11_193474:
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallsequential_11_inputsequential_11_193458sequential_11_193460sequential_11_193462sequential_11_193464sequential_11_193466sequential_11_193468sequential_11_193470sequential_11_193472sequential_11_193474*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_193103?
softmax_4/PartitionedCallPartitionedCall.sequential_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_193308q
IdentityIdentity"softmax_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namesequential_11_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193387

inputs
sequential_11_193366
sequential_11_193368	
sequential_11_193370
sequential_11_193372	'
sequential_11_193374:	?N&
sequential_11_193376:"
sequential_11_193378:&
sequential_11_193380:"
sequential_11_193382:
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_193366sequential_11_193368sequential_11_193370sequential_11_193372sequential_11_193374sequential_11_193376sequential_11_193378sequential_11_193380sequential_11_193382*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_193103?
softmax_4/PartitionedCallPartitionedCall.sequential_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_softmax_4_layer_call_and_return_conditional_losses_193308q
IdentityIdentity"softmax_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_save_fn_194064
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?

?
D__inference_dense_16_layer_call_and_return_conditional_losses_192941

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
.__inference_sequential_13_layer_call_fn_193332
sequential_11_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	?N
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsequential_11_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*'
_read_only_resource_inputs	
	*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_193311o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namesequential_11_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_1940809
5key_value_init120616_lookuptableimportv2_table_handle1
-key_value_init120616_lookuptableimportv2_keys3
/key_value_init120616_lookuptableimportv2_values	
identity??(key_value_init120616/LookupTableImportV2?
(key_value_init120616/LookupTableImportV2LookupTableImportV25key_value_init120616_lookuptableimportv2_table_handle-key_value_init120616_lookuptableimportv2_keys/key_value_init120616_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: q
NoOpNoOp)^key_value_init120616/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2T
(key_value_init120616/LookupTableImportV2(key_value_init120616/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
W
;__inference_global_average_pooling1d_8_layer_call_fn_193967

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *_
fZRX
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_192859i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
__inference_restore_fn_194072
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
sequential_11_input8
%serving_default_sequential_11_input:0??????????
	softmax_42
StatefulPartitionedCall_1:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures"
_tf_keras_sequential
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
C
1
2
3
 4
!5"
trackable_list_wrapper
C
0
1
2
 3
!4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
"non_trainable_variables

#layers
$metrics
%layer_regularization_losses
&layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_13_layer_call_fn_193332
.__inference_sequential_13_layer_call_fn_193502
.__inference_sequential_13_layer_call_fn_193525
.__inference_sequential_13_layer_call_fn_193431?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193597
I__inference_sequential_13_layer_call_and_return_conditional_losses_193669
I__inference_sequential_13_layer_call_and_return_conditional_losses_193455
I__inference_sequential_13_layer_call_and_return_conditional_losses_193479?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_192849sequential_11_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
'serving_default"
signature_map
P
(_lookup_layer
)	keras_api
*_adapt_function"
_tf_keras_layer
?

embeddings
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem{m|m} m~!mv?v?v? v?!v?"
	optimizer
C
1
2
3
 4
!5"
trackable_list_wrapper
C
0
1
2
 3
!4"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_11_layer_call_fn_192985
.__inference_sequential_11_layer_call_fn_193723
.__inference_sequential_11_layer_call_fn_193746
.__inference_sequential_11_layer_call_fn_193147?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193817
I__inference_sequential_11_layer_call_and_return_conditional_losses_193888
I__inference_sequential_11_layer_call_and_return_conditional_losses_193212
I__inference_sequential_11_layer_call_and_return_conditional_losses_193277?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_softmax_4_layer_call_fn_193893?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_softmax_4_layer_call_and_return_conditional_losses_193898?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
':%	?N2embedding/embeddings
!:2dense_16/kernel
:2dense_16/bias
!:2dense_17/kernel
:2dense_17/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_193694sequential_11_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
L
Rlookup_table
Stoken_counts
T	keras_api"
_tf_keras_layer
"
_generic_user_object
?2?
__inference_adapt_step_193946?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_embedding_layer_call_fn_193953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_layer_call_and_return_conditional_losses_193962?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?2?
;__inference_global_average_pooling1d_8_layer_call_fn_193967?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_193973?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_16_layer_call_fn_193982?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_16_layer_call_and_return_conditional_losses_193993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_dense_17_layer_call_fn_194002?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_17_layer_call_and_return_conditional_losses_194012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
j
k_initializer
l_create_resource
m_initialize
n_destroy_resourceR jCustom.StaticHashTable
Q
o_create_resource
p_initialize
q_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	rtotal
	scount
t	variables
u	keras_api"
_tf_keras_metric
^
	vtotal
	wcount
x
_fn_kwargs
y	variables
z	keras_api"
_tf_keras_metric
"
_generic_user_object
?2?
__inference__creator_194017?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_194025?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_194030?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_194035?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_194040?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_194045?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
:  (2total
:  (2count
.
r0
s1"
trackable_list_wrapper
-
t	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
v0
w1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
,:*	?N2Adam/embedding/embeddings/m
&:$2Adam/dense_16/kernel/m
 :2Adam/dense_16/bias/m
&:$2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
,:*	?N2Adam/embedding/embeddings/v
&:$2Adam/dense_16/kernel/v
 :2Adam/dense_16/bias/v
&:$2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
?B?
__inference_save_fn_194064checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_194072restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_57
__inference__creator_194017?

? 
? "? 7
__inference__creator_194035?

? 
? "? 9
__inference__destroyer_194030?

? 
? "? 9
__inference__destroyer_194045?

? 
? "? B
__inference__initializer_194025R???

? 
? "? ;
__inference__initializer_194040?

? 
? "? ?
!__inference__wrapped_model_192849R??? !8?5
.?+
)?&
sequential_11_input?????????
? "5?2
0
	softmax_4#? 
	softmax_4?????????k
__inference_adapt_step_193946JS???<
5?2
0?-?
??????????IteratorSpec 
? "
 ?
D__inference_dense_16_layer_call_and_return_conditional_losses_193993\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_16_layer_call_fn_193982O/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_dense_17_layer_call_and_return_conditional_losses_194012\ !/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_17_layer_call_fn_194002O !/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_embedding_layer_call_and_return_conditional_losses_193962_/?,
%?"
 ?
inputs?????????	
? ")?&
?
0?????????
? ?
*__inference_embedding_layer_call_fn_193953R/?,
%?"
 ?
inputs?????????	
? "???????????
V__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_193973{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
;__inference_global_average_pooling1d_8_layer_call_fn_193967nI?F
??<
6?3
inputs'???????????????????????????

 
? "!???????????????????z
__inference_restore_fn_194072YSK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_194064?S&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193212~R??? !G?D
=?:
0?-
text_vectorization_8_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193277~R??? !G?D
=?:
0?-
text_vectorization_8_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193817jR??? !3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_11_layer_call_and_return_conditional_losses_193888jR??? !3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_11_layer_call_fn_192985qR??? !G?D
=?:
0?-
text_vectorization_8_input?????????
p 

 
? "???????????
.__inference_sequential_11_layer_call_fn_193147qR??? !G?D
=?:
0?-
text_vectorization_8_input?????????
p

 
? "???????????
.__inference_sequential_11_layer_call_fn_193723]R??? !3?0
)?&
?
inputs?????????
p 

 
? "???????????
.__inference_sequential_11_layer_call_fn_193746]R??? !3?0
)?&
?
inputs?????????
p

 
? "???????????
I__inference_sequential_13_layer_call_and_return_conditional_losses_193455wR??? !@?=
6?3
)?&
sequential_11_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193479wR??? !@?=
6?3
)?&
sequential_11_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193597jR??? !3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_193669jR??? !3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_13_layer_call_fn_193332jR??? !@?=
6?3
)?&
sequential_11_input?????????
p 

 
? "???????????
.__inference_sequential_13_layer_call_fn_193431jR??? !@?=
6?3
)?&
sequential_11_input?????????
p

 
? "???????????
.__inference_sequential_13_layer_call_fn_193502]R??? !3?0
)?&
?
inputs?????????
p 

 
? "???????????
.__inference_sequential_13_layer_call_fn_193525]R??? !3?0
)?&
?
inputs?????????
p

 
? "???????????
$__inference_signature_wrapper_193694?R??? !O?L
? 
E?B
@
sequential_11_input)?&
sequential_11_input?????????"5?2
0
	softmax_4#? 
	softmax_4??????????
E__inference_softmax_4_layer_call_and_return_conditional_losses_193898\3?0
)?&
 ?
inputs?????????

 
? "%?"
?
0?????????
? }
*__inference_softmax_4_layer_call_fn_193893O3?0
)?&
 ?
inputs?????????

 
? "??????????