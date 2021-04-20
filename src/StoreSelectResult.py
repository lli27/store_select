# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2021/02/26 10:42
@Desc   : 铺货选店
将结果数据写入数据库：
门店分组、相似店topn
"""
from src.cluster.NewStoreCluster import NewStoreCluster
from src.utils.SortTopNCompute import SortTopNCompute
from src.utils.LogFactory import LogFactory
from src.db.ODPSdb import ODPSdb
import traceback


class StoreSelectResult:
    def __init__(self):
        self.new_store_cluster = NewStoreCluster()
        self.sort_top_compute = SortTopNCompute()
        self.odps = ODPSdb()
        self.logging = LogFactory()

    def main(self):
        try:
            cluster_df = self.new_store_cluster.main()
            self.logging.info("将门店分组数据写入数据库!")
            self.odps.write_to_db(data=cluster_df, tablename="dws_fd_toc_match_str_group", if_partition=0)
            sort_top_df = self.sort_top_compute.main()
            self.odps.write_to_db(data=sort_top_df, tablename="dws_fd_toc_match_sim_str_topn", if_partition=0)
            self.logging.info("将相似店topn矩阵写入数据库!")
        except:
            self.logging.error(traceback.format_exc())
        return


if __name__ == '__main__':
    StoreSelectResult().main()
