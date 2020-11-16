import pathlib
from .utils import is_nncf_enabled, check_nncf_is_enabled, load_checkpoint

from mmdet.utils import get_root_logger

if is_nncf_enabled():
    try:
        from nncf.initialization import InitializingDataLoader
        from nncf.structures import QuantizationRangeInitArgs
        from nncf.dynamic_graph.patch_pytorch import nncf_model_input

        from nncf import NNCFConfig
        from nncf import load_state
        from nncf import create_compressed_model, register_default_init_args
        from nncf.utils import get_all_modules
        from nncf.nncf_network import NNCFNetwork

        class_InitializingDataLoader = InitializingDataLoader
    except:
        raise RuntimeError("Incompatible version of NNCF")
else:
    class DummyInitializingDataLoader:
        pass


    class_InitializingDataLoader = DummyInitializingDataLoader


class MMInitializeDataLoader(class_InitializingDataLoader):
    def get_inputs(self, dataloader_output):
        # redefined InitializingDataLoader because
        # of DataContainer format in mmdet
        kwargs = {k: v.data[0] for k, v in dataloader_output.items()}
        return (), kwargs

    # TODO: not tested; need to test
    def get_target(self, dataloader_output):
        return dataloader_output["gt_bboxes"], dataloader_output["gt_labels"]


def wrap_nncf_model(model, cfg, data_loader_for_init=None, get_fake_input_func=None,
                    should_use_dummy_forward_with_export_part=True):
    """
    The function wraps mmdet model by NNCF
    Note that the parameter `get_fake_input_func` should be the function `get_fake_input`
    -- cannot import this function here explicitly
    """
    check_nncf_is_enabled()
    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    nncf_config = NNCFConfig(cfg.nncf_config)
    logger = get_root_logger(cfg.log_level)

    if data_loader_for_init:
        wrapped_loader = MMInitializeDataLoader(data_loader_for_init)
        nncf_config = register_default_init_args(nncf_config, None, wrapped_loader)
    elif not cfg.nncf_load_from:
        raise RuntimeError("Tried to load NNCF checkpoint, but there is no path")

    if cfg.nncf_load_from:
        resuming_state_dict = load_checkpoint(model, cfg.nncf_load_from)
        logger.info(f"Loaded NNCF checkpoint from {cfg.nncf_load_from}")
    else:
        resuming_state_dict = None

    def _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        assert get_fake_input_func is not None

        input_size = nncf_config.get("input_info").get('sample_size')
        assert len(input_size) == 4 and input_size[0] == 1

        H, W = input_size[-2:]
        C = input_size[1]
        orig_img_shape = tuple([H, W, C])  # HWC order here for np.zeros to emulate cv2.imread

        device = next(model.parameters()).device

        # NB: the full cfg is required here!
        fake_data = get_fake_input_func(cfg, orig_img_shape=orig_img_shape, device=device)
        return fake_data

    def dummy_forward_without_export_part(model):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        fake_data = _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func)
        img, img_metas = fake_data["img"], fake_data["img_metas"]
        img = nncf_model_input(img)
        with model.forward_dummy_context(img_metas):
            model(img)

    def dummy_forward_with_export_part(model):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        fake_data = _get_fake_data_for_forward(cfg, nncf_config, get_fake_input_func)
        img, img_metas = fake_data["img"], fake_data["img_metas"]
        img = nncf_model_input(img)
        with model.forward_export_context(img_metas):
            model(img)

    if "nncf_should_use_dummy_forward_with_export_part" in cfg:
        # TODO: this parameter is for debugging, remove it later
        should_use_dummy_forward_with_export_part = cfg.get("nncf_should_use_dummy_forward_with_export_part")
        logger.debug(f"set should_use_dummy_forward_with_export_part={should_use_dummy_forward_with_export_part}")

    if should_use_dummy_forward_with_export_part:
        logger.debug(f"dummy_forward = dummy_forward_with_export_part")
        dummy_forward = dummy_forward_with_export_part
    else:
        logger.debug(f"dummy_forward = dummy_forward_without_export_part")
        dummy_forward = dummy_forward_without_export_part

    model.dummy_forward_fn = dummy_forward

    compression_ctrl, model = create_compressed_model(model, nncf_config, dummy_forward_fn=dummy_forward,
                                                      resuming_state_dict=resuming_state_dict)
    import torch
    from nncf.utils import no_jit_trace
    from nncf.quantization.layers import QuantizerExportMode, get_scale_zp_from_input_low_input_high, \
        ExportQuantizeToONNXQuantDequant, ExportQuantizeToFakeQuantize
    from functools import partial
    def run_hacked_export_quantization(self, x: torch.Tensor):
        with no_jit_trace():
            input_range = abs(self.scale) + self.eps
            # todo: take bias into account during input_low/input_high calculation
            input_low = input_range * self.level_low / self.level_high
            input_high = input_range

            if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
                y_scale, y_zero_point = get_scale_zp_from_input_low_input_high(self.level_low,
                                                                               self.level_high,
                                                                               input_low,
                                                                               input_high)

        if self._export_mode == QuantizerExportMode.ONNX_QUANTIZE_DEQUANTIZE_PAIRS:
            return ExportQuantizeToONNXQuantDequant.apply(x, y_scale, y_zero_point)
        if self._export_mode == QuantizerExportMode.FAKE_QUANTIZE:
            x = x / 2.0
            return ExportQuantizeToFakeQuantize.apply(x, self.levels, input_low, input_high, input_low * 2,
                                                      input_high * 2)
        raise RuntimeError

    model.nncf_module.backbone.features.init_block.conv.pre_ops._modules[
        '0'].op.run_export_quantization = partial(run_hacked_export_quantization,
                                                  model.nncf_module.backbone.features.init_block.conv.pre_ops._modules[
                                                      '0'].op)

    return compression_ctrl, model


def unwrap_nncf_model(module):
    if not is_nncf_enabled():
        return module
    if isinstance(module, NNCFNetwork):
        return module.get_nncf_wrapped_model()
    return module
