from texttable import Texttable
from mmcv.runner.hooks.hook import Hook

from nncf.nncf_logger import logger

class CompressionHook(Hook):
    def __init__(self, compression_ctrl=None):
        self.compression_ctrl = compression_ctrl

    def after_train_iter(self, runner):
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        self.compression_ctrl.scheduler.epoch_step()

    def before_run(self, runner):
        runner.logger.info(print_statistics(self.compression_ctrl.statistics()))

    def after_run(self, runner):
        self.compression_ctrl.export_model("compressed_model.onnx")


def print_statistics(stats):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            logger.info(key)
            logger.info(val.draw())
        else:
            logger.info('{}: {}'.format(key, val))