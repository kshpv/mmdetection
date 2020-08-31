from texttable import Texttable
from mmcv.runner.hooks.hook import Hook

import torch

from nncf import NNCFConfig
from nncf.nncf_logger import logger

class CompressionHook(Hook):
    def __init__(self, compression_ctrl=None, cfg=None):
        self.compression_ctrl = compression_ctrl
        self.nncf_config = NNCFConfig(cfg.nncf_config)

    def after_train_iter(self, runner):
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        self.compression_ctrl.scheduler.epoch_step()

    def before_run(self, runner):
        runner.logger.info(print_statistics(self.compression_ctrl.statistics()))

    def after_run(self, runner):
        input_size = self.nncf_config.get("input_info").get('sample_size')
        # TODO: add device description
        device = "cpu"
        input_args = ([torch.randn(input_size).to(device), ],)
        input_kwargs = dict(return_loss=False, dummy_forward=True)

        logger.info("Exporting the model to ONXX format")
        self.compression_ctrl.export_model("compressed_model.onnx", *input_args, **input_kwargs)


def print_statistics(stats):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            logger.info(key)
            logger.info(val.draw())
        else:
            logger.info('{}: {}'.format(key, val))