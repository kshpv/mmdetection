from texttable import Texttable
from mmcv.runner.hooks.hook import Hook

import torch
import time

class CompressionHook(Hook):
    def __init__(self, compression_ctrl=None, cfg=None):
        from nncf import NNCFConfig

        self.compression_ctrl = compression_ctrl
        self.nncf_config = NNCFConfig(cfg.nncf_config)
        self.work_dir = cfg.work_dir

    def after_train_iter(self, runner):
        self.compression_ctrl.scheduler.step()

    def after_train_epoch(self, runner):
        self.compression_ctrl.scheduler.epoch_step()

        # TODO: add exporting support
        # if runner.rank == 0:
        #     self.export_model_to_onnx(runner)

    def before_run(self, runner):
        print_statistics(self.compression_ctrl.statistics(), runner.logger)

    # TODO: add export model to ONNX
    # def export_model_to_onnx(self):
    #     model = self.compression_ctrl._model
    #
    #     input_size = self.nncf_config.get("input_info").get('sample_size')
    #     device = "cpu"
    #
    #     input_kwargs = dict(return_loss=False, dummy_forward=True)
    #
    #     runner.logger.info("Exporting the model to ONXX format")
    #     print(f"self.compression_ctrl = {self.compression_ctrl}")
    #     # TODO: args and kwargs
    #     f_name = str(self.work_dir) + "/compressed_model_" + str(self.compression_ctrl.scheduler.last_epoch) + ".onnx"
    #
    #     data = tools.export.get_fake_input(self.cfg, input_size)
    #
    #     tools.export.export_to_onnx(model, data, input_kwargs)


    # def export_model_to_onnx(self, runner):
    #     input_size = self.nncf_config.get("input_info").get('sample_size')
    #     device = "cpu"
    #     input_args = ([torch.randn(input_size).to(device), ],)
    #     input_kwargs = dict(return_loss=False, dummy_forward=True)
    #
    #     runner.logger.info("Exporting the model to ONXX format")
    #     print(f"self.compression_ctrl = {self.compression_ctrl}")
    #     # TODO: args and kwargs
    #     f_name = str(self.work_dir) + "/compressed_model_" + str(self.compression_ctrl.scheduler.last_epoch) + ".onnx"
    #     self.compression_ctrl.export_model(f_name, *input_args, **input_kwargs)
    #     print("FINISH EXPORTING")


def print_statistics(stats, logger):
    for key, val in stats.items():
        if isinstance(val, Texttable):
            logger.info(key)
            logger.info(val.draw())
        else:
            logger.info('{}: {}'.format(key, val))
