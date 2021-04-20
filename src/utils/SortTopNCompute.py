# -*- coding: utf-8 -*-

"""
@Author: Lily
@Date: 2021/03/22 16:56
@Desc: 计算试销款热卖门店的相似店
输出TOPN相似店
"""
from src.distance.DistribStyDistance import DistribStyDistance
from src.distance.CusLabelDistance import CusLabelDistance
from src.db.ODPSdb import ODPSdb
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import squareform
from src.utils.LogFactory import LogFactory
import numpy as np
import pandas as pd


class SortTopNCompute:
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

    def get_csr_topn_idx_data(self, csr_row, index):
        """
        对行元素，输出排序的topn
        :param csr_row:
        :param topn:
        :return: [门店编码， 门店编码， 距离]
        """
        nnz = csr_row.getnnz()
        if nnz == 0:
            return None
        elif nnz <= self.topn:
            result = zip(csr_row.indices, csr_row.data)
        else:
            arg_idx = np.argpartition(csr_row.data, self.topn)[:self.topn]
            result = zip(index[arg_idx], csr_row.data[arg_idx])

        return sorted(result, key=lambda x: x[1])

    def scipy_cossim_top(self, C, index):
        """
        输入距离矩阵，迭代输出topn矩阵
        :param C:
        :param topn:
        :return:
        """
        C_sparse = csr_matrix(C)
        return [self.get_csr_topn_idx_data(row, index) for row in C_sparse]

    def main(self):
        """
        :return:
        """
        if self.method:
            d = self.distribstydistance.main()
        else:
            d = self.cuslabeldistance.main()
        result = []
        for i, distrib_code in enumerate(d[2]):
            d_condensed = d[0][i]
            index = d[1][i]
            self.topn = len(index) // 4 if self.topn == -1 else self.topn
            d_sparse = squareform(d_condensed)
            d_topn = self.scipy_cossim_top(d_sparse, index)
            a = np.repeat(index, self.topn)
            b = np.reshape(d_topn, (-1, 2))
            c = np.hstack((a[:, None], b))
            result.extend(c)
        sort_top_df = pd.DataFrame(result, columns=['str_code', 'sim_str_code', 'distance'])
        sort_top_df['distance'] = sort_top_df['distance'].astype('float')
        return sort_top_df


if __name__ == '__main__':
    sort_top_df = SortTopNCompute().main()
