import unittest
import numpy as np
import pytest
import torch
import os
from utils import make_tensor
from mmcv.ops import RoIAlignRotated, roi_align_rotated

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)

class TestRoiAlignMLU(TestCase):
    def _sample_inputs_rotated_feature_align(self, device, dtype):
        test_cases = (
            ((1, 4, 7, 1), (1, 6)), 
            ((9, 27, 24, 1), (1, 6)),
        )
        samples_input = []
        samples_rois = []
        for feature_shape, bbox_shape in test_cases:
            a = make_tensor(feature_shape, device=device, dtype=dtype, requires_grad=False, low=-100, high=100, seed=90)
            b = make_tensor(bbox_shape, device=device, dtype=dtype, requires_grad=False, low=0, high=10, seed=90)
            samples_input.append(a)
            samples_rois.append(b)
        return samples_input, samples_rois
    
    def test_roi_align(self, device='mlu'):
        dtype_list = [torch.float]
        for dtype in dtype_list:
            inputs, roises= self._sample_inputs_rotated_feature_align(device='cpu', dtype=dtype)
            for input, rois in zip(inputs, roises):
                rois[:, 3:5] += rois[:, 1:3]
                input_mlu = input.to(device)
                rois_mlu = rois.to(device)

                pool_h = np.random.randint(1, input.shape[2] + 1)
                pool_w = np.random.randint(1, input.shape[3] + 1)
                spatial_scale = np.random.choice([0.5, 1.0, 2.0])
                sampling_ratio = np.random.choice([0, 2, 4])

                input.requires_grad = True
                output = roi_align_rotated(input, rois, (pool_h, pool_w), spatial_scale,
                                sampling_ratio, True)
                output.backward(torch.ones_like(output))

                input_mlu.requires_grad = True
                output_mlu = roi_align_rotated(input_mlu, rois_mlu, (pool_h, pool_w), spatial_scale,
                                sampling_ratio, True)
                output_mlu.backward(torch.ones_like(output_mlu))

                assert torch.allclose(
                    output_mlu.cpu(), output, atol=1e-3)
                                
                assert torch.allclose(
                    input_mlu.grad.cpu(), input.grad, atol=1e-3)
    

if __name__ == '__main__':
    run_tests()