# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from __future__ import absolute_import
from __future__ import division
import itertools
import os
import unittest
import time
import numpy as _np
import mxnet as mx
from mxnet import np, npx, autograd
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray, retry, use_np
from common import with_seed, TemporaryDirectory
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf, assert_exception, is_op_runnable, collapse_sum_like
from mxnet.ndarray.ndarray import py_slice
from mxnet.base import integer_types
import scipy.stats as ss

def label_mapping(label, local_index, unique_sorted_global_label, rank=0, num_sample=1, num_local=1):
    _mapping_dict = {}
    local_sampled_class = local_index + rank * num_local
    global_label_set = set(unique_sorted_global_label)
    for idx, absolute_label in enumerate(local_sampled_class):
        if absolute_label in global_label_set:
            _mapping_dict[absolute_label] = idx + rank * num_sample

    label_list = list(label)
    mapping_label = []
    for i in range(len(label_list)):
        absolute_label = label_list[i]
        if absolute_label in _mapping_dict.keys():
            mapping_label.append(_mapping_dict[absolute_label])
        else:
            mapping_label.append(-1)
    return mapping_label


@with_seed()
def test_np_mapping():
    # (input dtype, expected output dtype)
    t0 = time.time()
    num_sample = 100000
    num_local = 500000
    label = _np.random.choice(a=1000000, size=10000, replace=True)
    postive = label[label < num_local]
    num_postive = len(postive)
    num_negtive = num_sample - num_postive
    negtive = _np.random.choice(a=1000000, size=5000, replace=False)
    local_index = _np.concatenate((postive, negtive))
    local_index = _np.unique(local_index)
    unique_sorted_gloabl_label = _np.unique(label)

    # label = _np.array([0, 101, 234, 356, 417, 506, 623, 790, 845, 901], dtype='int32')
    # local_index = _np.array([0, 87, 101, 201, 234, 289, 356, 390, 417, 487], dtype='int32')
    mapping_label = label_mapping(label, local_index, unique_sorted_gloabl_label, rank=0, num_sample=len(local_index), num_local=num_local)
    print(mapping_label)
    label_nd = mx.nd.array(label)
    local_index_nd = mx.nd.array(local_index)
    unique_sorted_global_label_nd = mx.nd.array(label)
    mapping_label_nd = mx.nd.contrib.label_mapping(label=label_nd, local_index=local_index_nd, unique_sorted_global_label=unique_sorted_global_label_nd, rank=0, num_sample=len(local_index), num_local=num_local)
    t1 = time.time()
    print(mapping_label_nd)
    t2 = time.time()
    print("t mapping: {}".format(t1 - t0))
    print("t print: {}".format(t2 - t1))
    mapping_label_np = mapping_label_nd.asnumpy().astype('int32')
    diff = _np.sum(mapping_label_np - mapping_label)
    print('diff: {}'.format(diff))



if __name__ == '__main__':
    # import nose
    # nose.runmodule()
    test_np_mapping()
    
