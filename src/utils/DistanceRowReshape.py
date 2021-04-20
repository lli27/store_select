# -*- coding: utf-8 -*-

"""
@Author: Lily
@Date: 2021/03/29 10:04
@Desc: 将N*N的距离矩阵，转为一行三列的形式，便于数据库中的表计算。(rowindex, columnindex, value)
"""

from src.distance.DistribStyDistance import DistribStyDistance
from src.distance.CusLabelDistance import CusLabelDistance
from src.db.ODPSdb import ODPSdb
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import squareform
from src.utils.LogFactory import LogFactory
import numpy as np
import pandas as pd


class DistanceRowReshape:
    def __init__(self, topn=-1, method=1):
        """
        参数初始化
        :param topn: 取前topn，如果未输入该参数, 默认值-1，则取数据的前1/4
        :param method: 选择哪种距离。0表示客群标签，1表示历史热销款号
        """
        self.distribstydistance = DistribStyDistance()
        self.cuslabeldistance = CusLabelDistance()
        self.odps = ODPSdb()
        self.logging = LogFactory()
        self.topn = topn
        self.method = method

    def RowReshape(self, C_sparse, index):
        """
        将N*N的距离矩阵，转为一行三列的形式
        :param C_sparse:
        :param index:
        :return:
        """
        df = pd.DataFrame()
        for i in range(C_sparse.shape[0]):
            row = C_sparse.getrow(i)
            sim_str_code = index[row.indices]
            str_code = index[i]
            row_data = np.reshape(row.data, newshape=(-1, 1))
            row_indices = np.reshape(sim_str_code, newshape=(-1, 1))
            row_df = np.hstack([row_indices, row_data])
            row_df = pd.DataFrame(row_df, columns=['sim_str_code', 'distance'])
            row_df['str_code'] = str_code
            df = df.append(row_df)
        return df

    def main(self):
        """
        :return:
        """
        if self.method:
            d = self.distribstydistance.main()
        else:
            d = self.cuslabeldistance.main()
        df_full = pd.DataFrame()
        for i, distrib_code in enumerate(d[2]):
            d_condensed = d[0][i]
            index = d[1][i]
            d_sparse = squareform(d_condensed)
            C_sparse = csr_matrix(d_sparse)
            df_full = df_full.append(self.RowReshape(C_sparse, index))
        return df_full


if __name__ == '__main__':
    result = SortTopNCompute().main()




