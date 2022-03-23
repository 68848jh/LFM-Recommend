# 开发时间: 2022/3/22 14:37
# _*_coding:utf8_*_

"""
author: gjh
date:2022/3/22
util function：
get_item_info：从movies.csv中抽取item详情
get_ave_score：得到用户对item的平均评分
"""
import os


# 得到item的详情
def get_item_info(input_file):
    """
    get item info:[title,genre]
    Args:    #参数
        input_file:item info file
    Return:  #输出
        a dict: key: itemid,value: [title,genre]
    """
    # 判断文件是否存在
    if not os.path.exists(input_file):
        return {}
    item_info = {}  # 定义一个数据结构
    linenum = 0
    fp = open(input_file,encoding='gb18030', errors='ignore')
    for line in fp:  # 按行读取
        if linenum == 0:  # 如果是第一行
            linenum += 1
            continue
        item = line.strip().split(',')  # 由于数据中用的是，这里用，切分
        if len(item) < 3:  # 如果切分出来的数据小于3列  过滤
            continue
        elif len(item) == 3:  # 如切分出来的数据等于3列 那么就是item里的三列，下标item[0],item[1],item[2]
            itemid, title, genres = item[0], item[1], item[2]
        elif len(item) > 3:  # 如果大于3列  选择的itemid是第一列item[0]  分类genres是最后一列item[-1]
            itemid = item[0]
            genres = item[-1]
            title = ",".join(item[1:-1])  # title是itemid和genres中间的部分用，连接起来
        item_info[itemid] = [title, genres]  # 将title，genres存储到iteminfo中 以itemid对应
    fp.close()  # 关闭文件提取
    return item_info


#得到用户对item的平均评分
def get_ave_score(input_file):
    """
    get item ave rating score
    Args:
        input file: user rating file
    Rreturn
        a dict : key: itemid, value: ave_score
    """
    #判断输入文件是否存在
    if not os.path.exists(input_file):
        return {}       #如果不存在返回空
    linenum = 0
    record_dict = {}    #定义数据结构 存储结果
    score_dict = {}    #定义数据结构  存储中间的一些变量
    fp = open(input_file,encoding='gb18030', errors='ignore')   #编码范围“gb18030” ，忽略非法字符“ignore”
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')  # 由于数据中用的是，这里用，切分
        if len(item) < 4:
            continue
        usedid,itemid,rating = item[0],item[1],item[2]
        if itemid not in record_dict:        #记录itemid被多少用户评过分 总分是多少  记录到record_dict
            record_dict[itemid] = [0,0]
        record_dict[itemid][0] += 1           #第一列记录被多少用户评过分
        record_dict[itemid][1] += float(rating)      #第二列记录总分
    fp.close()
    #将结果存入到输出的数据结构中
    for itemid in record_dict:
        score_dict[itemid] = round(record_dict[itemid][1]/record_dict[itemid][0],3)  #round，3  保留3位有效数字
    return  score_dict


#得到训练样本数据的函数
def get_train_data(input_file):
    """
    get train data for LFM model train
    Args:
      input_file: user_item rating file
    Return:
      a list :[(userid,itemid,label),(userid1,itemid2,label)]
    """

    if not os.path.exists(input_file):
        return []

    score_dict = get_ave_score(input_file)
    neg_dict = {}               #由于正负样本要保持一致  定义两个数据结构分别存储 负样本
    pos_dict = {}               #由于正负样本要保持一致  定义两个数据结构分别存储 正样本
    train_data = []             #定义一个输出的数据结构
    linenum = 0
    score_thr = 4.0
    fp = open(input_file,encoding='gb18030', errors='ignore')
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid,itemid,rating = item[0],item[1],float(item[2])
        if userid not in pos_dict:
            pos_dict[userid] = []
        if userid not in neg_dict:
            neg_dict[userid] = []
        if float(rating) >= score_thr:          #如果评分大于等于了定义的评分 就是正样本
            pos_dict[userid].append((itemid,1))  #1表示label的元组
        else:
            score = score_dict.get(itemid,0)  #存储用户对itemid的打分，如果没有用户打分就默认0分
            neg_dict[userid].append((itemid,score))
    fp.close()

    #进行正负样本的均衡以及负采样
    for userid in pos_dict:
        data_num = min(len(pos_dict[userid]),len(neg_dict.get(userid,[])))            #计算每一个userid的训练样本的数目
        if data_num > 0:        #如果正负样本大于0  []列表推导式
            train_data += [(userid,zuhe[0],zuhe[1]) for zuhe in pos_dict[userid]][:data_num]
        else:
            continue
        #对userid负样本的list进行排序
        sorted_neg_list = sorted(neg_dict[userid],key=lambda element:element[1],reverse=True)[:data_num]
        train_data += [(userid,zuhe[0],0) for zuhe in sorted_neg_list]
    return train_data