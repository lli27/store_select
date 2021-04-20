# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2021/03/10 10:19
@Desc   : 根据历史热销款号计算距离矩阵
之前是将款号0-1化处理，因为特征个数不一致，采用Jaccard距离。改进版，将热销频次也考虑进来，使用TF-IDF将文本数据数值化。这里也用余弦距离。
向量化TF-IDF+标准化Normalizer+余弦距离
"""

from src.db.ODPSdb import ODPSdb
from src.conf.Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import pdist
from src.utils.LogFactory import LogFactory

class DistribStyDistance():

    def __init__(self):
        self.config = Config().config_data
        self.logging = LogFactory()
        self.odps = ODPSdb()

    def preprocessing(self, data):
        """
        数据预处理
        :param data:
        :return:
        """
        index = data['str_code'].values
        # 向量化
        tfidf2 = TfidfVectorizer()
        x_train = tfidf2.fit_transform(data['sty_code_list'])
        # 标准化
        scaler = Normalizer()
        x_train = scaler.fit_transform(x_train)
        return x_train, index


    def distance(self, x_train):
        """
        计算距离矩阵
        :param x_train:
        :return: 返回一维的压缩距离矩阵
        对于一个m维矩阵，(i, j)对应的距离，在位置m * i + j - ((i + 2) * (i + 1)) // 2
        """
        d_condensed = pdist(x_train.toarray(), metric='cosine')
        return d_condensed

    def main(self):
        sql_cmd = self.config.get('GET_STR_STY_LABEL')
        data = self.odps.get_data(sql=sql_cmd)
        d_condensed, index, distrib = [], [], []
        for distrib_code in set(data.distrib_code):
            x_train, idx = self.preprocessing(data=data[data['distrib_code']==distrib_code])
            d_condensed.append(self.distance(x_train))
            index.append(idx)
            distrib.append(distrib_code)
        return d_condensed, index, distrib


if __name__ == '__main__':
    d = DistribStyDistance().main()


