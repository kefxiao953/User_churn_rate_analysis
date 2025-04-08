
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#读取数据
df = pd.read_csv(r'K:\BaiduNetdiskDownload\userlostprob_train_2.csv', sep=',', encoding='utf-8')
# df.info()

#
# #=====构建新特征=======
# 转为日期型格式
df['arrive_date']=pd.to_datetime(df['arrive_date'])
df['date']=pd.to_datetime(df['date'])
# 相减得到“提前预定天数”列
df['day_advanced']=(df['arrive_date']-df['date']).dt.days
# 删除原有列
df=df.drop(['date','arrive_date'],axis=1)
#
df.describe()

# #===========异常值处理（负数）==============#
# 对特定列中的负数进行处理，填充为中位数
for col in ['deltaPrice_1', 'deltaPrice_2','min_price']:
    # 找出该列中值小于 0 的行，将这些行对应的值替换为该列的中位数
    df.loc[df[col] < 0, col] = df[col].median()

# 对特定列中的负数进行处理，填充为 0
for col in ['user_value_profit', 'profits_ctrips']:
    # 找出该列中值小于 0 的行，将这些行对应的值替换为 0
    df.loc[df[col] < 0, col] = 0

# 极值处理
for i in df.columns:
    # 找出该列中值小于该列 1% 分位数的行，将这些行对应的值替换为 1% 分位数的值
    df.loc[df[i] < np.percentile(df[i], 1), i] = np.percentile(df[i], 1)
    # 找出该列中值大于该列 99% 分位数的行，将这些行对应的值替换为 99% 分位数的值
    df.loc[df[i] > np.percentile(df[i], 99), i] = np.percentile(df[i], 99)


# #==============缺失值处理==================#
#查看各列缺失情况，并统计
df_count = df.count()
na_count = len(df) - df_count
na_rate = na_count/len(df)
#按values正序排列，不放倒序是为了后边的图形展示排列
a = na_rate.sort_values(ascending=True)
a1 = pd.DataFrame(a)
# plt.rcParams['font.sans-serif']=['SimHei']
x = df.shape[1]
fig = plt.figure(figsize=(8,12)) #图形大小
plt.barh(range(x),a1[0],color='orange',alpha=1)
plt.tick_params(axis='both',labelsize=14)
plt.xlabel('数据缺失占比') #添加轴标签
columns1 = a1.index.values.tolist() #列名称
plt.yticks(range(x),columns1)
plt.xlim([0,1]) #设置X轴的刻度范围
for x,y in enumerate(a1[0]):
    plt.text(y,x,'%.3f' %y,va='bottom')
plt.show()

# #=====缺失值删除=====
# 删除缺失值比例大于80%的行和列

# 删除缺失值比例大于80%的行和列
print('删除空值前数据维度是:{}'.format(df.shape))
df.dropna(axis=0,thresh=df.shape[1]*0.2,inplace=True)
df.dropna(axis=1,thresh=df.shape[0]*0.2,inplace=True)
print('删除空值后数据维度是:{}'.format(df.shape))

#
# 绘制数据框各列的直方图分布
df.hist(figsize=(20, 20), color='purple')
plt.show()

# 定义列表filter_mean，存储适合用均值填充缺失值的字段名
filter_mean=['business_rate_tag','business_rate_tag.1','cancle_rate_tag','reval_tag ']
# 遍历数据框df的所有列名
for i in df.columns:
    # 判断列名是否在filter_mean列表中
    if i in filter_mean:
        # 如果在，使用该列的均值填充缺失值，并直接修改原数据框
        df[i].fillna(df[i].mean(),inplace=True)
    else:
        # 如果不在，使用该列的中位数填充缺失值，并直接修改原数据框
        df[i].fillna(df[i].median(),inplace=True)

#=====极值处理=====
# 缩尾处理
for i in df.columns:
        #小于1%分位数的用1%分位数填充
    df.loc[df[i]<np.percentile(df[i],1),i]=np.percentile(df[i],1)
    # 大于99%分位数的用99%分位数填充
    df.loc[df[i]>np.percentile(df[i],99),i]=np.percentile(df[i],99)

# ===========相关性图=================
# =====用户=====
# # 定义用户特征列表，包含用于分析的用户相关特征列名
# user_features = ['visit_num_365', 'star_prefer','sid', 'price_sensitive', 'year_order_num', 'order_cancle_pv', 'order_cancle_rate', 'last_pv_gap',
#                  'last_order_gap', 'hours_half_land', 'order_pv_24h', 'total_order_num', 'avg_hotel_pv', 'h',
#                  'deltaPrice_2', 'deltaPrice_1', 'decision_habit', 'user_value_profit', 'profits_ctrips', 'cr', 'consuming_capacity', 'avg_price']
#
# # 从数据框 df 中提取用户特征列，计算这些特征之间的相关性矩阵
# user_corr = df[user_features].corr()
# # 创建一个大小为 18x14 的图形和坐标轴对象
# fig, ax = plt.subplots(figsize=(18, 14))
# # 绘制用户特征相关性矩阵的热力图
# sns.heatmap(user_corr,
#             xticklabels=True,
#             yticklabels=True,
#             square=False,
#             linewidths=.5,
#             annot=True,
#             cmap="Purples")
# plt.show()
#
# #====酒店====
# # 定义酒店信息特征列表
# hotel_features = ['hotel_cr', 'hotel_uv', 'comment_pv', 'novoters', 'cancle_rate','min_price', 'cr_tag', 'uv_tag1', 'uv_tag2', 'business_rate_tag',
#                   'business_rate_tag.1','reval_tag', 'comment__num_tag1', 'comment__num_tag2', 'cancle_rate_tag', 'novoters_tag1', 'novoters_tag2',
#                   'deltaPrice_tag2_t1','min_price_tag1','min_price_tag2', 'fisrt_order_bu', 'visit_detail_pv']
# # 计算这些酒店特征之间的相关性矩阵
# hotel_corr1 = df[hotel_features].corr()
# # 创建一个大小为18x12的图形和对应的坐标轴对象
# fig, ax = plt.subplots(figsize=(18, 12))
# # 绘制酒店特征相关性矩阵的热力图
# sns.heatmap(hotel_corr1,
#             xticklabels=True,
#             yticklabels=True,
#             square=False,
#             linewidths=.5,
#             annot=True,
#             cmap="Purples")
# # 显示绘制好的热力图
# plt.show()

#===========数据降维，PCA============
# 用户价值
c_value=['user_value_profit','profits_ctrips']
# 用户消费水平
consume_level=['avg_price','consuming_capacity']
# 用户偏好价格
price_prefer=['deltaPrice_1','deltaPrice_2']
# 酒店热度
hotel_hot=['comment_pv','novoters']
# 24小时内浏览次数最多的酒店热度
hotel_hot_pre=['comment__num_tag1','novoters_tag1']
# 24小时内浏览酒店的平均热度
hotel_hot_pre2=['comment__num_tag2','novoters_tag2']

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
df['c_value']=pca.fit_transform(df[c_value])
df['consume_level']=pca.fit_transform(df[consume_level])
df['price_prefer']=pca.fit_transform(df[price_prefer])
df['hotel_hot']=pca.fit_transform(df[hotel_hot])
df['hotel_hot_pre']=pca.fit_transform(df[hotel_hot_pre])
df['hotel_hot_pre2']=pca.fit_transform(df[hotel_hot_pre2])

df.drop(c_value,axis=1,inplace=True)
df.drop(consume_level,axis=1,inplace=True)
df.drop(price_prefer,axis=1,inplace=True)
df.drop(hotel_hot,axis=1,inplace=True)
df.drop(hotel_hot_pre,axis=1,inplace=True)
df.drop(hotel_hot_pre2,axis=1,inplace=True)
df.drop('total_order_num',axis=1,inplace=True)
print('PCA降维后数据维度是：{}'.format(df.shape))


#=====数据标准化=========
# 数据标准化
from sklearn.preprocessing import StandardScaler

y = df['label']
x = df.drop('label',axis=1)
scaler = StandardScaler()

X= scaler.fit_transform(x)


# #=====逻辑回归=======
from sklearn.model_selection import train_test_split, GridSearchCV

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state=420)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
#
#  # 实例化一个LR模型
# lr = LogisticRegression()
#  # 训练模型
# lr.fit(X_train,y_train)
#  # 预测1类的概率
# y_prob = lr.predict_proba(X_test)[:,1]
#  # 模型对测试集的预测结果
# y_pred = lr.predict(X_test)
#  # 获取真阳率、伪阳率、阈值
# fpr_lr,tpr_lr,threshold_lr = metrics.roc_curve(y_test,y_prob)
# # AUC得分
# auc_lr = metrics.auc(fpr_lr,tpr_lr)
#  # 模型准确率
# score_lr = metrics.accuracy_score(y_test,y_pred)
# print('模型准确率为:{0},AUC得分为:{1}'.format(score_lr,auc_lr))
# print('============================================================')
# print(classification_report(y_test,y_pred,labels=None,target_names=None,sample_weight=None, digits=2))
#
# #=========朴树贝叶斯===========
# from sklearn.naive_bayes import GaussianNB
#
# gnb = GaussianNB()                                                # 实例化一个LR模型
# gnb.fit(X_train,y_train)                                          # 训练模型
# y_prob = gnb.predict_proba(X_test)[:,1]                           # 预测1类的概率
# y_pred = gnb.predict(X_test)                                      # 模型对测试集的预测结果
# fpr_gnb,tpr_gnb,threshold_gnb = metrics.roc_curve(y_test,y_prob)  # 获取真阳率、伪阳率、阈值
# auc_gnb = metrics.auc(fpr_gnb,tpr_gnb)                            # AUC得分
# score_gnb = metrics.accuracy_score(y_test,y_pred)                 # 模型准确率
#
#
# print('模型准确率为:{0},AUC得分为:{1}'.format(score_gnb,auc_gnb))
# print('============================================================')
# print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
#
# #===========支持向量机============
#
# from sklearn.svm import SVC
#
# svc = SVC(kernel='rbf',C=1,max_iter=100).fit(X_train,y_train)
# y_prob = svc.decision_function(X_test)                              # 决策边界距离
# y_pred = svc.predict(X_test)                                        # 模型对测试集的预测结果
# fpr_svc,tpr_svc,threshold_svc = metrics.roc_curve(y_test,y_prob)    # 获取真阳率、伪阳率、阈值
# auc_svc = metrics.auc(fpr_svc,tpr_svc)                              # 模型准确率
# score_svc = metrics.accuracy_score(y_test,y_pred)
# print('模型准确率为:{0},AUC得分为:{1}'.format(score_svc,auc_svc))
# print('============================================================')
# print(classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2))
#
#
# #===========决策树============
# from sklearn import tree
#
# dtc = tree.DecisionTreeClassifier()                              # 建立决策树模型
# dtc.fit(X_train,y_train)                                         # 训练模型
# y_prob = dtc.predict_proba(X_test)[:,1]                          # 预测1类的概率
# y_pred = dtc.predict(X_test)                                     # 模型对测试集的预测结果
# fpr_dtc,tpr_dtc,threshod_dtc= metrics.roc_curve(y_test,y_prob)   # 获取真阳率、伪阳率、阈值
# score_dtc = metrics.accuracy_score(y_test,y_pred)
# auc_dtc = metrics.auc(fpr_dtc,tpr_dtc)
# print('模型准确率为:{0},AUC得分为:{1}'.format(score_dtc,auc_dtc))
# print('============================================================')
# print(classification_report(y_test,y_pred,labels=None,target_names=None,sample_weight=None, digits=2))
#
# #===========随机森林============
# from sklearn.ensemble import RandomForestClassifier
#
#
# rfc = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=15,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     n_jobs=-1
# )                                   # 建立随机森林分类器
# rfc.fit(X_train,y_train)
#
# # 训练随机森林模型
# y_prob = rfc.predict_proba(X_test)[:,1]                            # 预测1类的概率
# y_pred=rfc.predict(X_test)                                         # 模型对测试集的预测结果
# fpr_rfc,tpr_rfc,threshold_rfc = metrics.roc_curve(y_test,y_prob)   # 获取真阳率、伪阳率、阈值
# auc_rfc = metrics.auc(fpr_rfc,tpr_rfc)                             # AUC得分
# score_rfc = metrics.accuracy_score(y_test,y_pred)                  # 模型准确率
# print('模型准确率为:{0},AUC得分为:{1}'.format(score_rfc,auc_rfc))
# print('============================================================')
# print(classification_report(y_test,y_pred,labels=None,target_names=None,sample_weight=None, digits=2))
#
#
# #===========XGBoost============

# import xgboost as xgb
#
# dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns)
# dtest = xgb.DMatrix(X_test, feature_names=X_test.columns)
#
# # 设置xgboost建模参数
# params={'booster':'gbtree','objective': 'binary:logistic','eval_metric': 'auc',
#     'max_depth':8,'gamma':0,'lambda':2,'subsample':0.7,'colsample_bytree':0.8,
#     'min_child_weight':3,'eta': 0.2,'nthread':8,'silent':1}
#
# # 训练模型
# watchlist = [(dtrain,'train')]
# bst=xgb.train(params,dtrain,num_boost_round=500,evals=watchlist)
#
# # 输入预测为正类的概率值
# y_prob=bst.predict(dtest)
#
# # 设置阈值为0.5，得到测试集的预测结果
# y_pred = (y_prob >= 0.5)*1
#
# # 获取真阳率、伪阳率、阈值
# fpr_xgb,tpr_xgb,threshold_xgb = metrics.roc_curve(y_test,y_prob)
# auc_xgb = metrics.auc(fpr_xgb,tpr_xgb)    # AUC得分
# score_xgb = metrics.accuracy_score(y_test,y_pred)    # 模型准确率
# print('模型准确率为:{0},AUC得分为:{1}'.format(score_xgb,auc_xgb))
# print('============================================================')
# print(classification_report(y_test,y_pred,labels=None,target_names=None,sample_weight=None, digits=2))

# #==========模型比较===========
# plt.style.use('bmh')
# plt.figure(figsize=(13, 10))
#
# plt.plot(fpr_lr, tpr_lr, label='lr: {0:.3f}'.format(score_lr))  # 逻辑回归
# plt.plot(fpr_gnb, tpr_gnb, label='gnb:{0:.3f}'.format(score_gnb))  # 朴素贝叶斯
# plt.plot(fpr_svc, tpr_svc, label='svc:{0:.3f}'.format(score_svc))  # 支持向量机
# plt.plot(fpr_dtc, tpr_dtc, label='dtc:{0:.3f}'.format(score_dtc))  # 决策树
# plt.plot(fpr_rfc, tpr_rfc, label='rfc:{0:.3f}'.format(score_rfc))  # 随机森林
# plt.plot(fpr_xgb, tpr_xgb, label='xgb:{0:.3f}'.format(score_xgb))  # XGBoost
#
# plt.legend(loc='lower right', prop={'size': 25})
# plt.xlabel('伪阳率')
# plt.ylabel('真阳率')
# plt.title('ROC carve')
# plt.savefig('./模型比较图.jpg', dpi=400, bbox_inches='tight')
# plt.show()

#=======RFM用户画像======
# 字段重名
rfm = df[['last_order_gap', 'year_order_num', 'consume_level']]
rfm.rename(columns={'last_order_gap':'recency', 'year_order_num': 'frequency', 'consume_level':'monetary'},
           inplace=True)

# 利用MinMaxScaler进行归一化处理
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(rfm)
rfm = pd.DataFrame(scaler.transform(rfm), columns=['recency', 'frequency','monetary'])

# 分箱
rfm['R'] = pd.qcut(rfm["recency"], 2)
rfm['F'] = pd.qcut(rfm["frequency"], 2)
rfm['M'] = pd.qcut(rfm["monetary"], 2)

# 根据分箱情况进行编码，二分类可以直接用标签编码方式
from sklearn.preprocessing import LabelEncoder

rfm['R'] = LabelEncoder().fit(rfm['R']).transform(rfm['R'])
rfm['F'] = LabelEncoder().fit(rfm['F']).transform(rfm['F'])
rfm['M'] = LabelEncoder().fit(rfm['M']).transform(rfm['M'])


# 定义RFM模型，需要特别注意的是，R值代表距离上次消费时间间隔，值越小客户价值越高，与F和M值正好相反。
def get_label(r, f, m):
    if (r == 0) & (f == 1) & (m == 1):
        return '高价值客户'
    if (r == 1) & (f == 1) & (m == 1):
        return '重点保持客户'
    if ((r == 0) & (f == 0) & (m == 1)):
        return '重点发展客户'
    if (r == 1) & (f == 0) & (m == 1):
        return '重点挽留客户'
    if (r == 0) & (f == 1) & (m == 0):
        return '一般价值客户'
    if (r == 1) & (f == 1) & (m == 0):
        return '一般保持客户'
    if (r == 0) & (f == 0) & (m == 0):
        return '一般发展客户'
    if (r == 1) & (f == 0) & (m == 0):
        return '潜在客户'


def RFM_convert(df):
    df['Label of Customer'] = df.apply(lambda x: get_label(x['R'], x['F'], x['M']), axis=1)

    df['R'] = np.where(df['R'] == 0, '高', '低')
    df['F'] = np.where(df['F'] == 1, '高', '低')
    df['M'] = np.where(df['M'] == 1, '高', '低')

    return df[['R', 'F', 'M', 'Label of Customer']]


rfm1 = RFM_convert(rfm)
print(rfm1.head(10))

value_counts = rfm1["Label of Customer"].value_counts().values
labels = rfm1["Label of Customer"].value_counts().index
explode = [0.1, 0.1, 0.1, 0, 0, 0, 0, 0]
color = ['deepskyblue','steelblue', 'lightskyblue', 'aliceblue','skyblue', 'cadetblue', 'cornflowerblue', 'dodgerblue']
plt.figure(figsize=(10, 7))

plt.pie(x=value_counts, labels=labels, autopct='%.2f%%', explode=explode, colors=color, wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'},
        textprops={'fontsize': 12, 'color': 'black'})
plt.legend(labels, bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.7)
plt.title('客户类别细分情况')
plt.show()

# ======F-score 方差分析法(降维)====
# 导入plot_importance
from xgboost import plot_importance
import xgboost as xgb
# X 是特征数据，y 是标签数据，都是 numpy.ndarray 类型
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置xgboost建模参数
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
   'max_depth': 8,
    'gamma': 0,
    'lambda': 2,
   'subsample': 0.7,
    'colsample_bytree': 0.8,
   'min_child_weight': 3,
    'eta': 0.2,
    'nthread': 8,
   'silent': 1
}

# 使用DataFrame的列名作为特征名称，这一步是关键修改
feature_names = X_train.columns

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, feature_names=feature_names)
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)

# 画柱状图
fig, ax = plt.subplots(figsize=(15, 15))
plot_importance(bst, height=0.5, ax=ax, max_num_features=40, color='chocolate')
plt.savefig('./重要性特征图.jpg', dpi=400, bbox_inches='tight')
plt.grid(False)  # 关闭网格
plt.show()

#=========k-means=====================
#肘部法则确定K
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# 选取刻画用户的重要指标
user_feature = ['decision_habit', 'order_cancle_pv', 'order_cancle_rate',
                'consume_level','star_prefer', 'last_order_gap', 'last_pv_gap',
                'h','sid', 'c_value', 'hours_half_land', 'price_sensitive',
                'price_prefer', 'day_advanced', 'avg_hotel_pv',
                'year_order_num']
user_attributes = df[user_feature]

# 数据标准化
scaler = StandardScaler()
user_attributes = scaler.fit_transform(user_attributes)

# 肘部法则确定k值
inertia_values = []
k_range = range(1, 11)  # 尝试从1到10个聚类
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=13)
    kmeans.fit(user_attributes)
    inertia_values.append(kmeans.inertia_)  # 记录每个k对应的惯性

# 可视化
plt.plot(k_range, inertia_values, 'bx-')
plt.xlabel('聚类数k')
plt.ylabel('惯性(Inertia)')
plt.title('肘部法则确定最优k值')
plt.show()



# # 选取刻画用户的重要指标
# user_feature = ['decision_habit', 'order_cancle_pv', 'order_cancle_rate',
#                 'consume_level','star_prefer', 'last_order_gap', 'last_pv_gap',
#                 'h','sid', 'c_value', 'hours_half_land', 'price_sensitive',
#                 'price_prefer', 'day_advanced', 'avg_hotel_pv',
#                 'year_order_num']
# user_attributes = df[user_feature]
#
# # 数据标准化
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(user_attributes)
# user_attributes = scaler.transform(user_attributes)
#
# #K-Means聚类
# from sklearn.cluster import KMeans
#
# Kmeans = KMeans(n_clusters=3, random_state=13)  # 建立KMean模型
# Kmeans.fit(user_attributes)  # 训练模型
# k_char = Kmeans.cluster_centers_  # 得到每个分类的质心
# personas = pd.DataFrame(k_char.T, index=user_feature, columns=['0类', '1类', '2类'])  # 用户画像表
# print(personas)
#
# #绘制热力图
# fig, ax = plt.subplots(figsize=(5, 10))
# sns.heatmap(personas, xticklabels=True,
#             yticklabels=True, square=False,
#             linewidths=.5, annot=True, cmap="YlGnBu")
# plt.tick_params(axis='both', labelsize=14)
# plt.show()
#
# plt.figure(figsize=(9, 9))
#
# class_k = list(Kmeans.labels_)  # 每个类别的用户个数
# percent = [class_k.count(1) / len(user_attributes),
#            class_k.count(0) / len(user_attributes),
#            class_k.count(2) / len(user_attributes)]  # 每个类别用户个数占比
#
# fig, ax = plt.subplots(figsize=(10, 10))
# colors = ['chocolate','sandybrown', 'peachpuff']
# types = ['高价值用户', '低价值用户', '中等群体']
# ax.pie(percent, radius=1, autopct='%.2f%%', pctdistance=0.75,
#        colors=colors, labels=types)
# ax.pie([1], radius=0.6, colors='w')
# plt.savefig('./用户画像.jpg', dpi=400, bbox_inches='tight')
# plt.show()