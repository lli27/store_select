# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2021/03/06 10:42
@Desc   : 根据店长评分数据生成分组结果
"""
"""
店长评分数据--反映主观的门店客群
行中心化+正则化Normalizer+余弦距离+KMeans聚类
使用轮廓系数进行评价
"""
from src.db.ODPSdb import ODPSdb
from src.utils.LogFactory import LogFactory
from sklearn.preprocessing import Normalizer
from src.conf.Config import Config
from Bio.Cluster import kcluster
import pandas as pd
from sklearn import metrics
import datetime


class CusLabelCluster:

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
        index = data['str_code']
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

    def cluster(self, x_train):
        """
        聚类
        :param x_train:
        :return:
        根据门店数目确定聚类数目的大概范围。1、每组最少30家门店，最多200家门店；2、聚类数目最少为2组，最多为30组。
        """
        silhouette_best = 0
        cluster_label = None
        cluster_num = None
        n = max(len(x_train) // 200, 2)
        m = min(len(x_train) // 30, 30) + 1
        self.logging.info("聚类数目最小值：{0}，最大值：{1}".format(n, m))
        step = 1 if m < 11 else 2
        for i in range(n, m, step):
            clusterid, error, nfound = kcluster(x_train, nclusters=i, dist='u', npass=1000)
            silhouette_score = metrics.silhouette_score(x_train, clusterid, metric='cosine')
            self.logging.info("聚类数目：{0}，聚类得分：{1}".format(i, silhouette_score))
            self.logging.info("找到解的次数：{0}".format(nfound))
            if silhouette_best < silhouette_score:
                silhouette_best = silhouette_score
                cluster_label = clusterid
                cluster_num = i
        self.logging.info("最优聚类数目：{0}，最优轮廓系数得分：{1}".format(cluster_num, silhouette_best))
        return cluster_label

    def main(self):
        bdp_date = (datetime.date.today() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
        sql_cmd = self.config.get('GET_CUSTOM_LABEL').format(bdp_date)
        data = self.odps.get_data(sql=sql_cmd)
        cluster_label, index = [], []
        for distrib_code in set(data.distrib_code):
            x_train, idx = self.preprocessing(data=data[data['distrib_code'] == distrib_code])
            cluster_label.extend(self.cluster(x_train=x_train))
            index.extend(idx.values)
        cluster_df = pd.DataFrame(cluster_label, index=index, columns=['label'], dtype=object)
        return cluster_df


if __name__ == '__main__':
    cluster_df = CusLabelCluster().main()
