# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2021/03/20 10:42
@Desc   : 门店组内日均、与试销热卖门店相似度
"""

# 门店分组
from src.cluster.CusLabelCluster import CusLabelCluster
from src.db.ODPSdb import ODPSdb
from src.conf.Config import Config
from src.utils.LogFactory import LogFactory
import pandas as pd
import datetime


class NewStoreCluster:
    def __init__(self):
        self.cuslabelcluster = CusLabelCluster()
        self.logging = LogFactory()
        self.config = Config().config_data
        self.odps = ODPSdb()
        self.bdp_date = (datetime.date.today() + datetime.timedelta(days=-1)).strftime("%Y%m%d")

    def main(self):
        cluster_df = self.cuslabelcluster.main()
        sql_cmd = self.config.get('GET_GEOGRAPHY_LABEL').format(self.bdp_date)
        full_store = self.odps.get_data(sql=sql_cmd)

        full_store = pd.merge(full_store, cluster_df, how='left', left_on=['str_code'], right_index=True)

        # 不同组中出现最频繁的：商圈、城市、城市等级、省区
        # 商圈+城市
        feature_matrix1 = full_store.groupby(by=['distrib_code', 'label', 'city_name', 'cbd_type_name'],
                                             as_index=False).agg(
            feature1=('str_code', 'count'))
        # 城市
        feature_matrix2 = full_store.groupby(by=['distrib_code', 'label', 'city_name'], as_index=False).agg(
            feature2=('str_code', 'count'))
        # 省区+城市等级
        feature_matrix3 = full_store.groupby(by=['distrib_code', 'label', 'str_org4_name', 'city_level'],
                                             as_index=False).agg(
            feature3=('str_code', 'count'))
        # 省区
        feature_matrix4 = full_store.groupby(by=['distrib_code', 'label', 'str_org4_name'], as_index=False).agg(
            feature4=('str_code', 'count'))
        for i, item in full_store[full_store['label'].isnull()].iterrows():
            cbd_type_name = item['cbd_type_name']
            city_name = item['city_name']
            city_level = item['city_level']
            str_org4_name = item['str_org4_name']
            if cbd_type_name is not None:
                data = feature_matrix1[(feature_matrix1['cbd_type_name'] == cbd_type_name) & (
                        feature_matrix1['city_name'] == city_name)].sort_values(by='feature1')
                if len(data) > 0:
                    full_store.loc[i, 'label'] = data.iloc[-1]['label']
                    continue
            if city_name is not None:
                data = feature_matrix2[feature_matrix2['city_name'] == city_name].sort_values(by='feature2')
                if len(data) > 0:
                    full_store.loc[i, 'label'] = data.iloc[-1]['label']
                    continue
            if city_level is not None:
                data = feature_matrix3[(feature_matrix3['city_level'] == city_level) & (
                        feature_matrix3['str_org4_name'] == str_org4_name)].sort_values(by='feature3')
                if len(data) > 0:
                    full_store.loc[i, 'label'] = data.iloc[-1]['label']
                    continue
            if str_org4_name is not None:
                data = feature_matrix4[feature_matrix4['str_org4_name'] == str_org4_name].sort_values(by='feature4')
                if len(data) > 0:
                    full_store.loc[i, 'label'] = data.iloc[-1]['label']
                    continue
            self.logging.error('{0}未匹配到！'.format(item['str_code']))
        full_store['label'] = full_store['label'].astype('int')
        return full_store


if __name__ == '__main__':
    cluster_df = NewStoreCluster().main()
