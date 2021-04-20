# -*- coding: utf-8 -*-

"""
@Author : Lily
@Date   : 2020/6/30 13:57
@Desc   : odps
"""

import traceback
import datetime
from odps import ODPS
from odps import DataFrame
from src.utils.LogFactory import LogFactory
from src.conf.Config import Config


class ODPSdb:
    """
    配置数据库
    """

    def __init__(self):
        """
        初始化
        """
        self.logging = LogFactory()
        self.config_data = Config().config_data
        self.conn = self.get_odps_conn()

    def get_odps_conn(self):
        """
        连接ODPS
        :return:
        """
        odps_config = self.config_data.get('ODPS')
        try:
            conn = ODPS(access_id=odps_config['USER'],
                        secret_access_key=odps_config['PASSWD'],
                        project=odps_config['DBNAME'],
                        endpoint=odps_config['URL'])
        except:
            self.logging.error(traceback.format_exc())
            raise
        return conn

    def get_data(self, sql):
        """
        查询数据
        :param sql:
        :return:
        """
        self.logging.info("查询数据：" + sql)
        try:
            with self.conn.execute_sql(sql).open_reader() as reader:
                data = reader.to_pandas()
            if len(data) == 0:
                self.logging.error("数据为空！")
        except:
            self.logging.error(traceback.format_exc())
            return None
        self.logging.info("read_data success!")
        return data

    def write_to_db(self, data, tablename, if_partition=1):
        """
        写入数据库
        :param data:
        :param tablename:
        :param if_partition: 是否分区
        :return:（1-成功；0-失败）
        """
        if data is None or data.empty:
            self.logging.error("{0} 写入数据库失败！数据为空！".format(tablename))
            return 0
        else:
            try:
                data['dw_date'] = datetime.datetime.now()
                if if_partition:
                    bdp_date = (datetime.date.today() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
                    DataFrame(data).persist(name=tablename, overwrite=True, partition="ds='{}'".format(bdp_date),
                                            create_partition=True, odps=self.conn, cast=True)
                else:
                    DataFrame(data).persist(name=tablename, overwrite=True, odps=self.conn, cast=True)
            except:
                self.logging.error(traceback.format_exc())
                return 0
        self.logging.info("{0} 成功写入数据库！".format(tablename))
        return 1
