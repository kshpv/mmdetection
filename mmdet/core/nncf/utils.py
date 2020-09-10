import pathlib
from collections import OrderedDict
from contextlib import contextmanager

import torch

try:
    import nncf

    _is_nncf_enabled = True
except:
    _is_nncf_enabled = False


def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError("Tried to use NNCF, but NNCF is not installed")


if is_nncf_enabled():
    try:
        from nncf.initialization import InitializingDataLoader
        from nncf.structures import QuantizationRangeInitArgs

        from nncf import NNCFConfig
        from nncf import load_state
        from nncf import create_compressed_model, register_default_init_args
        from nncf.utils import get_all_modules
        from nncf.dynamic_graph.context import no_nncf_trace as original_no_nncf_trace
    except:
        raise RuntimeError("Incompatible version of NNCF")


def wrap_nncf_model(model, cfg, data_loader_for_init=None):
    check_nncf_is_enabled()
    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    nncf_config = NNCFConfig(cfg.nncf_config)

    if data_loader_for_init:
        wrapped_loader = MMInitializeDataLoader(data_loader_for_init)
        # TODO: [NNCF] need check the arguments in register_default_init_args()
        # TODO: add loss factory that reads config file, creates them and passes to register_default_init_args()
        nncf_config.register_extra_structs([QuantizationRangeInitArgs(wrapped_loader)])
    elif not cfg.nncf_load_from:
        raise RuntimeError("Tried to load NNCF checkpoint, but there is no path")

    if cfg.nncf_load_from:
        resuming_state_dict = load_checkpoint(model, cfg.load_from)
    else:
        resuming_state_dict = None

    def dummy_forward(model):
        input_size = nncf_config.get("input_info").get('sample_size')
        device = next(model.parameters()).device
        input_args = ([torch.randn(input_size).to(device), ],)
        input_kwargs = dict(return_loss=False, dummy_forward=True)
        model(*input_args, **input_kwargs)

    model.dummy_forward_fn = dummy_forward

    compression_ctrl, model = create_compressed_model(model, nncf_config, dummy_forward_fn=dummy_forward,
                                                      resuming_state_dict=resuming_state_dict)
    print(*get_all_modules(model).keys(), sep="\n")
    return compression_ctrl, model


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint


def export_model_to_onnx(compression_ctrl, nncf_config, f_name):
    check_nncf_is_enabled()
    input_size = nncf_config.get("input_info").get('sample_size')
    device = "cpu"
    input_args = ([torch.randn(input_size).to(device), ],)
    input_kwargs = dict(return_loss=False, dummy_forward=True)
    compression_ctrl.export_model(f_name, *input_args, **input_kwargs)


class MMInitializeDataLoader(InitializingDataLoader):
    def get_inputs(self, dataloader_output):
        # redefined InitializingDataLoader because
        # of DataContainer format in mmdet
        kwargs = {k: v.data[0] for k, v in dataloader_output.items()}
        return (), kwargs

    # TODO: not tested; need to test
    def get_target(self, dataloader_output):
        return dataloader_output["gt_bboxes"], dataloader_output["gt_labels"]


@contextmanager
def nullcontext():
    """
    Context which does nothing; is needed to support python > python3.7
    """
    yield


def no_nncf_trace():
    """
    Wrapper for original NNCF no_nncf_trace() context
    """
    if is_nncf_enabled():
        return original_no_nncf_trace()
    return nullcontext()
