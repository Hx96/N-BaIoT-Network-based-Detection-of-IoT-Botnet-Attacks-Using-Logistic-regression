import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report,f1_score
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn import svm
import itertools


def Kfold_for_TrainModel(X_train_data, y_train_data):
    fold = KFold(len(X_train_data),5,shuffle = False)

    # 正则化前面的C 参数
    c_params = [0.01, 0.1, 1, 10, 100]
    #这块生成一个DataFrame 用来保存不同的C参数，对应的召回率是多少
    result_tables = pd.DataFrame(columns = ['C_parameter','Mean recall score'])
    result_tables['C_parameter'] = c_params
    j = 0
    for c_param in c_params:
        print('-------------------------------------------')
        print('C参数为：',c_param)
        print('-------------------------------------------')
        print('')

        recall_list = []
        for iteration, indices in enumerate(fold,start=1):
            #采用l1正则化
            lr = LogisticRegression(C=c_param, penalty = 'l1',max_iter=10)
            #indices[0] 保存的是这个k=5次训练中的某一次的用来验证的数据的索引
            #indices[1] 保存的是这个k=5次训练中的某一次的用来测试的数据的索引
            lr.fit(X_train_data.iloc[indices[0],:],
                   y_train_data.iloc[indices[0],:].values.ravel())#.ravel可以将输出降到一维
            #用剩下的一份数据进行测试（即indices[1]中所保存的下标）
            y_undersample_pred = lr.predict(X_train_data.iloc[indices[1],:].values)

            recall = recall_score(y_train_data.iloc[indices[1],:].values,
                                  y_undersample_pred)
            f1_score_value = f1_score(y_train_data.iloc[indices[1],:].values,
                                  y_undersample_pred)
            recall_list.append(recall)
            print('Iteration ',iteration," 召回率为：",recall,"f1_score为",f1_score_value)
        print('')
        print('平均召回率为：', np.mean(recall_list))
        print('')
        result_tables.loc[j,'Mean recall score'] = np.mean(recall_list)
        j = j+1

#     print(result_tables['Mean recall score'])
    result_tables['Mean recall score'] = result_tables['Mean recall score'].astype('float64')
    best_c_param = result_tables.loc[result_tables['Mean recall score'].idxmax(), 'C_parameter']
    print('*********************************************************************************')
    print('最佳模型对应的C参数 = ', best_c_param)
    print('*********************************************************************************')
    return best_c_param



def showData(data):
    print(data.shape)
    print(data.head())

def dataPrepare():
    #读取正常数据，异常数据并合并

    #读取正常数据 使用数据 Danmini_Doorbell 物理设备。
    data = pd.read_csv("benign_traffic.csv")
    data['Class']=0

    #读取异常数据mirai_attacks
    udp = pd.read_csv("./mirai_attacks/udp.csv")
    udp['Class'] = 1

    ack = pd.read_csv("./mirai_attacks/ack.csv")
    ack['Class'] = 1

    scan = pd.read_csv("./mirai_attacks/scan.csv")
    scan['Class'] = 1

    syn = pd.read_csv("./mirai_attacks/syn.csv")
    syn['Class'] = 1

    udpplain = pd.read_csv("./mirai_attacks/udpplain.csv")
    udpplain['Class'] = 1

    #读取异常数据gafgyt_attacks
    g_combo = pd.read_csv("./gafgyt_attacks/combo.csv")
    g_combo['Class'] = 1

    g_junk = pd.read_csv("./gafgyt_attacks/junk.csv")
    g_junk['Class'] = 1

    g_scan = pd.read_csv("./gafgyt_attacks/scan.csv")
    g_scan['Class'] = 1

    g_tcp = pd.read_csv("./gafgyt_attacks/tcp.csv")
    g_tcp['Class'] = 1

    g_udp = pd.read_csv("./gafgyt_attacks/udp.csv")
    g_udp['Class'] = 1


    frames = [data, udp, ack, scan, syn, udpplain
              ,g_combo, g_junk, g_scan, g_tcp, g_udp]

    #结果拼接
    result = pd.concat(frames, ignore_index=True)

    showData(result)
    #result.to_csv('test_res.csv')
    return result
def dataPreprocessing(data):
    #统计正例子和负例数目
    print('------')


    count_class = pd.value_counts(data['Class'],sort=True).sort_index()
    print(count_class)

    print('------')
    # 分离出特征X 和 输出变量y
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']

    #下采样使得分类数据1:1

    # 正样本个数
    positive_sample_count = len(data[data.Class == 1])
    print("正样本个数为：", positive_sample_count)

    # 负样本所对应的索引为
    negative_sample_index = np.array(data[data.Class == 0].index)
    print("负样本在数据集中所对应的索引为（打印前5个）：", negative_sample_index[:5])

    # 正样本所对应的索引
    positive_sample_index = data[data.Class == 1].index

    # numpy.random.choice(a, size=None, replace=True, p=None) 从给定的一维阵列生成一个随机样本
    # replace 样品是否有更换  True 表示每次都随机生成， false表示只随机生成一次
    random_positive_sample_index = np.random.choice(positive_sample_index, int(1*len(data[data.Class == 0])), replace=False)

    #在异常检测时调整样本数目
    #negative_sample_index = np.random.choice(positive_sample_index, len(data[data.Class == 0]), replace=False)


    print("正样本在数据集中所对应的索引为（打印前5个）：", random_positive_sample_index[:5])

    under_sample_index = np.concatenate([random_positive_sample_index, negative_sample_index])
    under_sample_data = data.iloc[under_sample_index, :]
    X_under_sample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
    y_under_sample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
    print('下采样后，新数据集中，正样本所占比例：',
          len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
    print('下采样后，新数据集中，负样本所占比例：',
          len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
    print('下采样后，新数据集的样本个数为：', len(under_sample_data))


    #数据集划分 将数据集进行7训练 3测试的划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    X_train_under_sample, X_test_under_sample, y_train_under_sample, y_test_under_sample = train_test_split(
        X_under_sample,
        y_under_sample,
        test_size=0.3,
        random_state=0)
    print('训练集样本数：', len(X_train_under_sample))
    print('测试集样本数：', len(X_test_under_sample))

    return X_train, X_test, y_train, y_test,X_train_under_sample, X_test_under_sample, y_train_under_sample, y_test_under_sample

def plot_confusion_matrix(confusion_matrix, classes):
#     print(confusion_matrix)
    #plt.imshow 绘制热图
    plt.figure()
    plt.imshow(confusion_matrix, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print('查准率为：',confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1]))
    print('召回率为：',confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0]))
    print('准确率为：',(confusion_matrix[0,0]+confusion_matrix[1,1])/(confusion_matrix[0,0]+confusion_matrix[0,1]+confusion_matrix[1,1]+confusion_matrix[1,0]))
    print('*********************************************************************************')


#数据准备
result=dataPrepare()
#数据预处理 训练集、验证集、测试集
X_train, X_test, y_train, y_test,X_train_under_sample, X_test_under_sample, y_train_under_sample, y_test_under_sample=dataPreprocessing(result)
# #模型
# # best_c_param = Kfold_for_TrainModel(X_train_under_sample, y_train_under_sample)
# best_c_param=10
# #评估
# lr = LogisticRegression(C = best_c_param, penalty = 'l1',max_iter=5)
# lr.fit(X_train_under_sample, y_train_under_sample.values.ravel())
#
# #获得测试集的测试结果
# y_undersample_pred = lr.predict(X_test.values)
# #构建稀疏矩阵
# conf_matrix = confusion_matrix(y_test,y_undersample_pred)
#
# np.set_printoptions(precision=2)
# class_names = [0,1]
#
# plot_confusion_matrix(conf_matrix
#                      , classes=class_names)


#模型
# best_c_param = Kfold_for_TrainModel(X_train_under_sample, y_train_under_sample)
best_c_param=10
#评估
lr = RandomForestClassifier()
lr.fit(X_train_under_sample, y_train_under_sample.values.ravel())

#获得测试集的测试结果
y_undersample_pred = lr.predict(X_test_under_sample.values)
#构建稀疏矩阵
conf_matrix = confusion_matrix(y_test_under_sample,y_undersample_pred)

np.set_printoptions(precision=2)
class_names = [0,1]

plot_confusion_matrix(conf_matrix
                     , classes=class_names)

# #IsolationForest
# rng = np.random.RandomState(42)
# method=''
# clf = IsolationForest(max_samples="auto", random_state=rng)
# clf.fit(X_train_under_sample)
# pred_y = clf.predict(X_train_under_sample)
# #IsolationForest 返回-1是异常值 返回1是正常值 所以画矩阵时将1替换为0，-1替换为1异常
# print(pred_y)
# pred_y = [0 if x==1 else x for x in pred_y]
# pred_y = [1 if x==-1 else x for x in pred_y]
# print(pred_y)
# #构建稀疏矩阵
# conf_matrix = confusion_matrix(y_train_under_sample,pred_y)
#
# np.set_printoptions(precision=2)
# class_names = [0,1]
#
# plot_confusion_matrix(conf_matrix
#                      , classes=class_names)
# #end

# #oneSVM
# rng = np.random.RandomState(42)
# method=''
# clf = svm.OneClassSVM(nu=0.02, kernel="rbf",
#                                       gamma=0.1)
# clf.fit(X_train_under_sample)
# pred_y = clf.predict(X_train_under_sample)
# #IsolationForest 返回-1是异常值 返回1是正常值 所以画矩阵时将1替换为0，-1替换为1异常
# print(pred_y)
# pred_y = [0 if x==1 else x for x in pred_y]
# pred_y = [1 if x==-1 else x for x in pred_y]
# print(pred_y)
# #构建稀疏矩阵
# conf_matrix = confusion_matrix(y_train_under_sample,pred_y)
#
# np.set_printoptions(precision=2)
# class_names = [0,1]
#
# plot_confusion_matrix(conf_matrix
#                      , classes=class_names)
# #end

#测试真实数据

