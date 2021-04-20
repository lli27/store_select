# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2021/03/06 10:42
@Desc   : 根据店长评分数据计算距离矩阵
"""
"""
店长评分数据--反映主观的门店客群
行中心化+正则化Normalizer+余弦距离
"""
from src.db.ODPSdb import ODPSdb
from src.utils.LogFactory import LogFactory
from sklearn.preprocessing import Normalizer
from src.conf.Config import Config
from scipy.spatial.distance import pdist
import datetime


class CusLabelDistance:

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
        data.drop(columns=['str_code', 'distrib_code'], axis=1, inplace=True)
        # 空值补零
        data.fillna(0, inplace=True)
        # 行中心化
        m_row = data.mean(axis=1)
        data = data - m_row[:, None]
        # 正则化Normalizer
        scaler = Normalizer()
        x_train = scaler.fit_transform(data)
        return x_train, index

    def distance(self, x_train):
        """
        计算距离矩阵
        :param x_train:
        :return: 返回一维的压缩距离矩阵
        对于一个m维矩阵，(i, j)对应的距离，在位置m * i + j - ((i + 2) * (i + 1)) // 2
        """
        d_condensed = pdist(x_train, metric='cosine')
        return d_condensed

    def main(self):
        """
        按分区
        :return:
        """
        bdp_date = (datetime.date.today() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
        sql_cmd = self.config.get('GET_CUSTOM_LABEL').format(bdp_date)
        data = self.odps.get_data(sql=sql_cmd)
        d_condensed, index, distrib = [], [], []
        for distrib_code in set(data.distrib_code):
            x_train, idx = self.preprocessing(data=data[data['distrib_code']==distrib_code])
            d_condensed.append(self.distance(x_train))
            index.append(idx)
            distrib.append(distrib_code)
        return d_condensed, index, distrib

if __name__ == '__main__':
    d = CusLabelDistance().main()
