import unittest
import torch
import json
import os
import numpy as np
from mmcv.ops import three_nn
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestThreeNNMLU(TestCase):
    def _read_io_from_txt(self, address=""):
        with open(address) as f:
            data = f.read()
        parsed_data = json.loads(data)
        input_sample1 = parsed_data['input_1']
        input_sample2 = parsed_data['input_2']
        dist_sample = parsed_data['dist']
        idx_sample = parsed_data['idx']
        return input_sample1, input_sample2, dist_sample, idx_sample

    def _test_three_nn(self, device='mlu', address=""):
        dtype_list = [torch.float]
        input_sample1, input_sample2, dist_baseline, idx_baseline = self._read_io_from_txt(address)
        for dtype in dtype_list:
            input1 = torch.tensor(input_sample1).to(device).type(dtype)
            input2 = torch.tensor(input_sample2).to(device).type(dtype)
            dist, idx = three_nn(input1, input2)
            assert np.allclose(dist.cpu().numpy(), dist_baseline, atol=1e-4)
            assert np.allclose(idx.cpu(), idx_baseline, 0)

    def test_three_nn(self, device="mlu"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._test_three_nn(address=dir_path + "/testcase_samples/three_nn/three_nn_samples.txt")

if __name__ == '__main__':
    run_tests()