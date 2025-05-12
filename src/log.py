import os
import logging
from logging.handlers import RotatingFileHandler
import datetime


class Logger:
    def __init__(self, log_dir="../logs",
                 log_file=f"app_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
                 max_bytes=5 * 1024 * 1024):
        """
        初始化日志模块。

        :param log_dir: 日志存储目录，默认为 "logs"
        :param log_file: 日志文件名，默认为 "app.log"
        :param max_bytes: 单个日志文件的最大大小（字节），默认 5MB
        """
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 完整的日志文件路径
        log_path = os.path.join(log_dir, log_file)

        # 配置日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s '
                '(%(filename)s:%(lineno)d)',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 配置日志处理器
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 配置 Logger
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)  # 设置全局日志级别为 DEBUG
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        """返回配置好的 Logger 对象"""
        return self.logger


log = Logger().get_logger()
