import os
import importlib
from collections import OrderedDict
from contextlib import contextmanager

import torch

from mmdet.utils import get_root_logger

_is_nncf_enabled = importlib.util.find_spec('nncf') is not None


def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError('Tried to use NNCF, but NNCF is not installed')


def get_nncf_version():
    if not is_nncf_enabled():
        return None
    import nncf
    return nncf.__version__


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
    from nncf.torch import load_state

    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint


@contextmanager
def nullcontext():
    """
    Context which does nothing
    """
    yield


def no_nncf_trace():
    """
    Wrapper for original NNCF no_nncf_trace() context
    """

    if is_nncf_enabled():
        from nncf.torch.dynamic_graph.context import no_nncf_trace as original_no_nncf_trace
        return original_no_nncf_trace()
    return nullcontext()


def is_in_nncf_tracing():
    if not is_nncf_enabled():
        return False

    from nncf.torch.dynamic_graph.context import get_current_context

    ctx = get_current_context()

    if ctx is None:
        return False
    return ctx.is_tracing


def is_accuracy_aware_training_set(nncf_config):
    if not is_nncf_enabled():
        return False
    from nncf.config.utils import is_accuracy_aware_training
    is_acc_aware_training_set = is_accuracy_aware_training(nncf_config)
    if is_acc_aware_training_set:
        logger = get_root_logger()
        if 'target_metric_name' not in nncf_config:
            logger.warning('The "target_metric_name" parameter not '
                           'found in the NNCF config - proceeding with the default "bbox_mAP"')
            nncf_config.target_metric_name = 'bbox_mAP'
    return is_acc_aware_training_set


def is_lazy_initialization_quantization(cfg):
    if not is_nncf_enabled():
        return False
    return cfg.get('lazy_initialization_quantization', False)


def add_checkpoint_reference_to_config(cfg, work_dir):
    def get_latest_checkpoint_path():
        checkpoint_path = os.path.join(work_dir, 'latest.pth')

    latest_checkpoint_path = get_latest_checkpoint_path()
    if os.path.exists(latest_checkpoint_path):
        cfg['load_from'] = latest_checkpoint_path
    else:
        raise RuntimeError('Can not find the latest checkpoint')


def remove_quantization_from_config(nncf_config):
    from nncf.config.extractors import extract_algorithm_names
    from nncf.config.extractors import extract_algo_specific_config

    algos = extract_algorithm_names(nncf_config)
    quantization_name = 'quantization'
    if quantization_name in algos:
        saved_quantization_config = extract_algo_specific_config(nncf_config, quantization_name)
    else:
        raise RuntimeError(f'There is no {quantization_name} config')
    for i in range(len(nncf_config['compression'])):
        if nncf_config['compression'][i]['algorithm'] == quantization_name:
            nncf_config['compression'].pop(i)
    return nncf_config, saved_quantization_config


def restore_quantization_to_config(nncf_config, quantization_config):
    assert isinstance(nncf_config['compression'], list)
    nncf_config['compression'].append(quantization_config)
    return nncf_config
