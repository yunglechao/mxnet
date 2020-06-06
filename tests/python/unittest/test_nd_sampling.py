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

def uniqe(ar):
    ar.sort()
    aux = ar
    mask = mx.nd.empty(aux.shape, dtype='uint8', ctx=aux.context)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    return aux[mask]

def setdiff1d(ar1, ar2):
    t0 = time.time()
    ar = mx.nd.concatenate([ar1, ar2])
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort()

    t1 = time.time()
    print("t sort: {}".format(t1 - t0))

    sar = ar[order]
    bool_ar = mx.nd.empty(ar.shape, dtype='uint8', ctx=sar.context)
    bool_ar[:-1] = (sar[1:] != sar[:-1]).astype('uint8')
    bool_ar[-1] = 1
    t11 = time.time()
    print("t comp: {}".format(t11 - t1))
    # flag = mx.nd.concatenate([bool_ar, mx.nd.array([1], dtype='uint8')])
    t2 = time.time()
    print("t concat: {}".format(t2 - t11))
    ret = mx.nd.empty(ar.shape, dtype='uint8', ctx=bool_ar.context)
    ret[order] = bool_ar # flag
    t3 = time.time()
    print("t index: {}".format(t3 - t2))
    diff = ar1[ret[:len(ar1)]]
    t4 = time.time()
    print("t mask index: {}".format(t4 - t3))
    return diff

class CenterNegetiveClassSample(object):
    """ Sample negative class center
    """

    def __init__(self, num_sample, num_local, rank, ctx=mx.cpu()):
        self.num_sample = num_sample
        self.num_local = num_local
        self.rank = rank
        self.negative_class_pool = mx.nd.arange(num_local, ctx=ctx)
        self.ctx = ctx

    def __call__(self, positive_center_index):
        """
        Return:
        -------
        negative_center_index: list of int
        """
        t1 = time.time()
        positive_center_index = uniqe(positive_center_index)
        t2 = time.time()
        print("t unique new: {}".format(t2 - t1))
        # t0 = time.time()
        # positive_center_index = mx.nd.unique(positive_center_index)
        # t1 = time.time()
        # print("t uniqe: {}".format(t1 - t0))
        negative_class_pool = setdiff1d(self.negative_class_pool, positive_center_index)
        negative_sample_size = self.num_sample - len(positive_center_index)
        t3 = time.time()
        negative_center_index = mx.nd.choice(negative_class_pool, a=None, size=negative_sample_size, replace=False)
        t4 = time.time()
        print("t choice: {}".format(t4 - t3))
        return negative_center_index

@with_seed()
def test_np_sampling():
    # (input dtype, expected output dtype)
    t0 = time.time()
    negative = CenterNegetiveClassSample(400000, 2000000, 0, ctx=mx.gpu(0))
    positive_center_index =mx.nd.shuffle(mx.nd.arange(200000), ctx=mx.gpu(0))
    # t0 = time.time()
    # positive_center_index = mx.nd.unique(positive_center_index)
    t1 = time.time()
    # print("t uniqe: {}".format(t1 - t0))
    sampled_negtive = negative(positive_center_index)
    t2 = time.time()
    print("t sampling: {}".format(t2 - t1))
    print(sampled_negtive[:10])
    t3 = time.time()
    print("t print: {}".format(t3 - t2))
    print("t total: {}".format(t3 - t0))



if __name__ == '__main__':
    # import nose
    # nose.runmodule()
    test_np_sampling()
    
