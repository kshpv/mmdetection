import json
import os
import os.path as osp
from pathlib import Path
import sys
import unittest
from shutil import copy2 as copy
from subprocess import run, CalledProcessError, PIPE

from mmcv import Config

from common import replace_text_in_file, collect_ap


class PublicModelsTestCase(unittest.TestCase):
    root_dir = Path('/tmp')
    coco_dir = root_dir.joinpath('data/coco')
    widerface_dir = root_dir.joinpath('data/widerface')
    snapshots_dir = root_dir.joinpath('snapshots')

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted([item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [item for item in content['images'] if
                                 item['id'] in selected_indexes]
            content['annotations'] = [item for item in content['annotations'] if
                                      item['image_id'] in selected_indexes]
            content['licenses'] = [item for item in content['licenses'] if
                                   item['id'] in selected_indexes]

        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    # доп проверка на валидационном датасете не ухдшилось ли?
    @classmethod
    def setUpClass(cls):
        cls.test_on_full, cls.train_on_full = False, False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}',
                check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(f'unzip {osp.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}', check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, "annotations_trainval2017.zip")):
            run(f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip '
                f'-P {cls.coco_dir}',
                check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}',
                check=True, shell=True)

        if not osp.exists(osp.join(cls.coco_dir, 'train2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/train2017.zip -P {cls.coco_dir}',
                check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'train2017')):
            run(f'unzip {osp.join(cls.coco_dir, "train2017.zip")} -d {cls.coco_dir}', check=True, shell=True)

        if not osp.exists(osp.join(cls.coco_dir, 'annotations/instances_train2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_train2017.zip")} -d {cls.coco_dir}',
                check=True, shell=True)

        if cls.train_on_full:
            cls.train_shorten_to = 581929
        else:
            cls.train_shorten_to = 200
        cls.annotation_train_file = osp.join(cls.coco_dir,
                                             f'annotations/instances_train2017_short_{cls.train_shorten_to}.json')
        cls.shorten_annotation(osp.join(cls.coco_dir, 'annotations/instances_train2017.json'),
                               cls.annotation_train_file, cls.train_shorten_to)

        if cls.test_on_full:
            cls.test_shorten_to = 5000
        else:
            cls.test_shorten_to = 200
        cls.annotation_test_file = osp.join(cls.coco_dir,
                                            f'annotations/instances_val2017_short_{cls.test_shorten_to}.json')
        cls.shorten_annotation(osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               cls.annotation_test_file, cls.test_shorten_to)

    def prerun(self, config_path, test_dir):
        log_file = osp.join(test_dir, 'test.log')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = osp.join(test_dir, 'config.py')
        cfg = Config.fromfile(config_path)
        update_args = {
            'data_root': f'{self.coco_dir}/',
            'data.train.dataset.ann_file': self.annotation_train_file,
            'data.train.dataset.img_prefix': osp.join(self.coco_dir, 'train2017/'),
            'data.train.ann_file': self.annotation_train_file,
            'data.train.img_prefix': osp.join(self.coco_dir, 'train2017/'),
            'data.val.ann_file': self.annotation_train_file,
            'data.val.img_prefix': osp.join(self.coco_dir, 'train2017/'),
            'data.test.ann_file': self.annotation_train_file,
            'data.test.img_prefix': osp.join(self.coco_dir, 'train2017/')
            # 'data.val.ann_file': self.annotation_test_file,
            # 'data.val.img_prefix': osp.join(self.coco_dir, 'val2017/'),
            # 'data.test.ann_file': self.annotation_test_file,
            # 'data.test.img_prefix': osp.join(self.coco_dir, 'val2017/')
        }
        if 'retinanet' in config_path:
            del update_args['data.train.dataset.ann_file']
            del update_args['data.train.dataset.img_prefix']
        cfg.merge_from_dict(update_args)
        with open(target_config_path, 'wt') as config_file:
            config_file.write(cfg.pretty_text)
        if not self.test_on_full:
            replace_text_in_file(target_config_path, 'keep_ratio=True', 'keep_ratio=False')
        return log_file, target_config_path

    def postrun(self, log_file, expected_output_file, metrics, thr):
        print('expected outputs', expected_output_file)
        ap = collect_ap(log_file)
        with open(expected_output_file) as read_file:
            content = json.load(read_file)
        reference_ap = content['map']
        print(f'expected {reference_ap} vs actual {ap}')
        for expected, actual, m in zip(reference_ap, ap, metrics):
            if expected - thr > actual:
                raise AssertionError(f'{m}: {expected} (expected) - {thr} (threshold) > {actual}')

    def download_if_not_yet(self, url):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = osp.join(self.snapshots_dir, osp.basename(url))
        if not osp.exists(path):
            run(f'wget {url} -P {self.snapshots_dir}', check=True, shell=True)
        return path

    def run_train_nncf(self, config_path, checkpoint):
        name = config_path.replace('configs/', '')[:-3]
        test_dir = osp.join(self.root_dir, name, 'pytorch')
        test_log_file, target_config_path = self.prerun(config_path, test_dir)

        work_dir = osp.join(test_dir, 'output')
        seed = '0'

        cfg = Config.fromfile(config_path)
        update_arg = {'load_from': checkpoint}
        cfg.merge_from_dict(update_arg)

        train_log_file = osp.join(test_dir, 'train.log')
        with open(train_log_file, 'w') as log_f:
            error = None
            try:
                run(['python',
                     'tools/train.py',
                     target_config_path,
                     '--work-dir=' + work_dir,
                     '--no-validate',
                     '--seed=' + seed
                     ], stdout=log_f, stderr=PIPE, check=True)
            except CalledProcessError as ex:
                error = 'Train script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

        return osp.join(work_dir, 'latest.pth')

    def run_nncf_pytorch_train_and_test(self, config_path, checkpoint, metrics=('bbox',), thr=0.01):
        print('\n\ntesting ' + config_path, file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        test_dir = osp.join(self.root_dir, name, 'pytorch')
        test_log_file, target_config_path = self.prerun(config_path, test_dir)

        snapshot = self.run_train_nncf(config_path, checkpoint)
        metrics_str = ' '.join(metrics)

        with open(test_log_file, 'w') as log_f:
            error = None
            try:
                run(f'python tools/test.py '
                    f'{target_config_path} '
                    f'{snapshot} '
                    f'--out {test_dir}/res.pkl --eval {metrics_str}',
                    stdout=log_f, stderr=PIPE, check=True, shell=True)
            except CalledProcessError as ex:
                error = 'Test script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

        expected_output_file = f'tests/expected_outputs/public/{name}-{self.train_shorten_to}-{self.test_shorten_to}.json'
        self.postrun(test_log_file, expected_output_file, metrics, thr)

    def run_nncf_openvino_export_test(self, config_path, snapshot, metrics=('bbox',), thr=0.02, alt_ssd_export=False):
        print('\n\ntesting OpenVINO export ' + '(--alt_ssd_export)' if alt_ssd_export else '' + config_path,
              file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        test_dir = osp.join(self.root_dir, name, 'openvino_alt_ssd_export' if alt_ssd_export else 'openvino_export')
        log_file, target_config_path = self.prerun(config_path, test_dir)

        metrics_str = ' '.join(metrics)

        snapshot_dir = osp.join(self.root_dir, name, 'pytorch/output/latest.pth')
        if not osp.exists(snapshot_dir):
            snapshot = self.run_train_nncf(self, config_path, snapshot)
        else:
            snapshot = snapshot_dir

        with open(log_file, 'w') as log_f:
            error = None
            try:
                run(f'/opt/intel/openvino/bin/setupvars.sh && '
                    f'python3 tools/export.py '
                    f'{target_config_path} '
                    f'{snapshot} '
                    f'{test_dir} '
                    f'openvino {"--alt_ssd_export" if alt_ssd_export else ""}',
                    stdout=log_f, stderr=PIPE, check=True, shell=True)
            except CalledProcessError as ex:
                error = 'Export script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

            try:
                run(f'/opt/intel/openvino/bin/setupvars.sh && '
                    f'python tools/test_exported.py '
                    f'{target_config_path} '
                    f'{osp.join(test_dir, "config.xml")} '
                    f'--out res.pkl --eval {metrics_str} 2>&1 | tee {log_file}',
                    stdout=log_f, stderr=PIPE, check=True, shell=True)
            except CalledProcessError as ex:
                error = 'Test script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

        expected_output_file = f'tests/expected_outputs/public/{name}-{self.train_shorten_to}-{self.test_shorten_to}.json'
        self.postrun(log_file, expected_output_file, metrics, thr)

    # def test_openvino_ssd300_coco_int8(self):
    #     origin_config = 'configs/nncf_compression/ssd/ssd300_coco_int8.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
    #           'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
    #     self.run_nncf_openvino_export_test(origin_config, self.download_if_not_yet(url))

    # def test_pytorch_ssd300_coco_int8(self):
    #     origin_config = 'configs/nncf_compression/ssd/ssd300_coco_int8.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
    #           'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
    #     self.run_nncf_pytorch_train_and_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_retinanet_r50_fpn_1x_coco_int8(self):
        origin_config = 'configs/nncf_compression/retinanet/retinanet_r50_fpn_1x_coco_int8.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_nncf_pytorch_train_and_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_retinanet_r50_fpn_1x_coco_int8(self):
        origin_config = 'configs/nncf_compression/retinanet/retinanet_r50_fpn_1x_coco_int8.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_nncf_openvino_export_test(origin_config, self.download_if_not_yet(url))


if __name__ == '__main__':
    unittest.main()
