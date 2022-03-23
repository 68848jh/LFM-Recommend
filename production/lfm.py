# 开发时间: 2022/3/22 17:13
# _*_coding=utf8_*_

"""
author:gjh
date:2022/3/22
lfm model train main function： lfm模型训练函数
"""
import operator

import numpy as np
import sys

sys.path.append("../util/")
import util.read as read


def lfm_train(train_data, F, alpha, beta,
              step):  # traina_dat:训练样本,F:user_vec和item_vec的维度,alpha:正则化参数，beta:模型的学习率，step:模型迭代的次数
    """
    Args:
      train_data: train_data for lfm 
      F: user_vec len,item_vec len
      alpha: regularization factor
      beta: learning rate
      step: iteration num
    Return: 
      dict: key: itemid, value: np.ndarray,
      dict: key: userid, value: np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):  # 在每一次迭代的时候
        for data_instance in train_data:  # 在训练样本中得到每一个训练的实例
            userid, itemid, label = data_instance
            if userid not in user_vec:  # 如果此userid是第一被训练的话  就需要初始化
                user_vec[userid] = init_model(F)  # 定义初始化  将userid对应的user_vec初始化为长度为F的向量 按照标准分布去初始
            if itemid not in item_vec:  # 如果词itemid是第一次被训练的话  需要初始化
                item_vec[itemid] = init_model(F)  # 定义初始化  将itemid对应的item_vec初始化为长度为F的向量 按照标准分布去初始
        # 模型参数迭代部分
        delta = label - model_predict(user_vec[userid], item_vec[itemid])
        for index in range(F):
            user_vec[userid][index] += beta * (
                        delta * item_vec[itemid][index] - alpha * user_vec[userid][index])  # index 公式中的f
            item_vec[itemid][index] += beta * (delta * user_vec[userid][index] - alpha * item_vec[itemid][index])
        beta = beta * 0.9  # 将学习率进行衰减
    return user_vec, item_vec


# 初始化函数
def init_model(vector_len):
    """
    Args:
        vector_len:the len of vector
    Return:
        a ndarray
    """
    return np.random.randn(vector_len)


# 模型对于已经得到的user_vec和item_vec的预测集
def model_predict(user_vecor, item_vector):
    """
    Args:
        user_vec: model produce user vector
        item_vec: model produce item vector
    Return:
        a num
    """
    # cos余弦相似度
    res = np.dot(user_vecor, item_vector) / (np.linalg.norm(user_vecor) * np.linalg.norm(item_vector))
    return res


# 将所有的函数串起来
def model_train_process():
    """
    test lfm model train
    """
    train_data = read.get_train_data("../data/ratings.csv")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 50)
    recom_result = give_rceom_result(user_vec,item_vec,'2')
    ana_recom_result(train_data,"2",recom_result)

def give_rceom_result(user_vec, item_vec, userid):
    """
    use lfm model result give fix userid recom result
    Args:
        user_vec: lfm model result
        item_vec: lfm model result
        userid: userid
    Return:
        a list: [(itemid,score),(itemid1,score1)]
    """

    fix_num = 10
    if userid not in user_vec:  # 如果给出的userid不在user_vec直接返回空
        return {}
    record = {}  # 存储每一个item与user_vec之间的距离
    recom_list = []
    user_vector = user_vec[userid]  # 计算每一个itemid与userid所对应的向量之间的距离

    for itemid in item_vec:
        item_vector = item_vec[itemid]
        res = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))  # 欧式距离计算
        record[itemid] = res  # 将得到的结果存到定义的字典record中
    for zuhe in sorted(record.items(), key=operator.itemgetter(1), reverse=True)[:fix_num]:  # 对定义的数据结果进行排序
        itemid = zuhe[0]
        score = round(zuhe[1],3)
        recom_list.append((itemid,score))
    return recom_list

#评估推荐结果
def ana_recom_result(train_data,userid,recom_list):
    """
    debug recom result for userid
    Args:
        train_data:train data for lfm model
        userid: fix userid
        recom_list:recom result by lfm
    """
    item_info = read.get_item_info("../data/movies.csv")
    for data_instance in train_data:
        tmp_userid,itemid,label = data_instance
        if tmp_userid == userid and label == 1:
            print(item_info[itemid])
    print("recom result")
    for zuhe in recom_list:
        print(item_info[zuhe[0]])


if __name__ == "__main__":
    model_train_process()
