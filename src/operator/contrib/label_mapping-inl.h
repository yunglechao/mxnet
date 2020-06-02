/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2020 by Contributors
 * \file label_mapping-inl.h
 * \brief mapping sampled local label
 * \author YongleZhao, XiangAn
 */

#ifndef MXNET_OPERATOR_CONTRIB_LABEL_MAPPING_INL_H_
#define MXNET_OPERATOR_CONTRIB_LABEL_MAPPING_INL_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../tensor/init_op.h"
// #include "../operator_common.h"
// #include "../mshadow_op.h"
namespace std
{
    template<> struct hash<mshadow::half::half_t>
    {
        std::size_t operator()(mshadow::half::half_t const& k) const noexcept
        {
            std::size_t h = std::hash<float>{}(static_cast<float>(k));
            return h;
        }
    };
}

namespace mxnet {
namespace op {

struct LabelMappingParam : public dmlc::Parameter<LabelMappingParam> {
  int rank;
  int num_sample;
  int num_local;
  DMLC_DECLARE_PARAMETER(LabelMappingParam) {
    DMLC_DECLARE_FIELD(rank).set_default(0)
    .describe("rank of local device.");
    DMLC_DECLARE_FIELD(num_sample).set_default(1)
    .describe("number of local sampled ids.");
    DMLC_DECLARE_FIELD(num_local).set_default(1)
    .describe("number of local ids.");
  }
};

inline bool LabelMappingShape(const NodeAttrs& attrs,
                            std::vector<mxnet::TShape>* in_attrs,
                            std::vector<mxnet::TShape>* out_attrs) {
  // inputs[0]: label tensor
  // inputs[1]: local_index vector
  // inputs[2]: unique_sorted_global_label tensor
  CHECK_EQ(in_attrs->size(), 3U);
  // outputs[0]: a new tensor
  CHECK_EQ(out_attrs->size(), 1U);
  // inputs[1] must be a vector
  CHECK_EQ(in_attrs->at(0).ndim(), 1);
  CHECK_EQ(in_attrs->at(1).ndim(), 1);
  CHECK_EQ(in_attrs->at(2).ndim(), 1);
  // The the length of the first dim of copied tensor
  // must equal to the size of index vector
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  return !mxnet::op::shape_is_none(out_attrs->at(0));
}

inline bool LabelMappingType(const NodeAttrs& attrs,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

// template<typename DType>
// inline void GetMappingLabel(const std::vector<DType> &label,
//                      const std::vector<DType> &local_index,
//                      const std::vector<DType> & unique_sorted_global_label,
//                      std::vector<DType> &label_mapping,
//                      int rank, int num_sample, int num_local) {
//     std::unordered_map<int, int> _mapping_dict;
//     std::unordered_set<int> global_label_set(unique_sorted_global_label.begin(), unique_sorted_global_label.end());
//     int absolute_label;
//     for(int i = 0; i < local_index.size(); i++) {
//         absolute_label = local_index[i] + rank * num_local;
//         if(global_label_set.count(absolute_label)) {
//             _mapping_dict[absolute_label] = i + rank * num_sample;
//         }
//     }
//     for(int& elem: label){
//         if(_mapping_dict.count(elem)){
//             elem = _mapping_dict[elem];
//         }else elem = -1;
//     }

// }

// template<typename DType>
// inline void GetMappingLabel(const std::vector<DType> &label,
//                      const std::vector<DType> &local_index,
//                      const std::vector<DType> & unique_sorted_global_label,
//                      std::vector<DType> &label_mapping,
//                      int rank, int num_sample, int num_local) {
//     std::unordered_map<int, int> _mapping_dict;
//     std::unordered_set<int> global_label_set(unique_sorted_global_label.begin(), unique_sorted_global_label.end());
//     int absolute_label;
//     for(int i = 0; i < local_index.size(); i++) {
//         absolute_label = local_index[i] + rank * num_local;
//         if(global_label_set.count(absolute_label)) {
//             _mapping_dict[absolute_label] = i + rank * num_sample;
//         }
//     }
//     for(int& elem: label){
//         if(_mapping_dict.count(elem)){
//             elem = _mapping_dict[elem];
//         }else elem = -1;
//     }
// }

template<typename DType, typename xpu>
inline void GetMappingLabel(const mshadow::Tensor<xpu, 1, DType> &label,
                     const mshadow::Tensor<xpu, 1, DType> &local_index,
                     const mshadow::Tensor<xpu, 1, DType> & unique_sorted_global_label,
                     mshadow::Tensor<xpu, 1, DType> &label_mapping,
                     int rank, int num_sample, int num_local) {
    std::vector<DType> label_vector (label.dptr_, label.dptr_ + label.size(0));
    std::vector<DType> local_index_vector (local_index.dptr_, local_index.dptr_ + local_index.size(0));
    std::vector<DType> unique_sorted_global_label_vector (unique_sorted_global_label.dptr_,
                                                           unique_sorted_global_label.dptr_ + unique_sorted_global_label.size(0));
    std::unordered_map<DType, DType> _mapping_dict;
    std::unordered_set<DType> global_label_set(unique_sorted_global_label_vector.begin(), unique_sorted_global_label_vector.end());
    for(int i = 0; i < local_index_vector.size(); i++) {
        DType absolute_label = local_index_vector[i] + rank * num_local;
        if(global_label_set.count(absolute_label)) {
            _mapping_dict[absolute_label] = i + rank * num_sample;
        }
    }
#pragma omp parallel for
    for(int i = 0; i < label_vector.size(); i++) {
        DType elem = label_vector[i];
        if(_mapping_dict.count(elem)){
            label_mapping[i] = _mapping_dict[elem];
        }
        else label_mapping[i] = -1;
    }
}

template<typename xpu>
void LabelMapping(const nnvm::NodeAttrs& attrs,
                 const OpContext &ctx,
                 const std::vector<TBlob> &inputs,
                 const std::vector<OpReqType> &req,
                 const std::vector<TBlob> &outputs) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const auto& param = nnvm::get<LabelMappingParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    mshadow::Tensor<xpu, 1, DType> label_mapping = outputs[0].FlatTo1D<xpu, DType>(s);
    mshadow::Tensor<xpu, 1, DType> label = inputs[0].FlatTo1D<xpu, DType>(s);
    mshadow::Tensor<xpu, 1, DType> local_index = inputs[1].FlatTo1D<xpu, DType>(s);
    mshadow::Tensor<xpu, 1, DType> unique_sorted_global_label = inputs[2].FlatTo1D<xpu, DType>(s);
    GetMappingLabel<DType, xpu>(label, local_index, unique_sorted_global_label, label_mapping,
                    param.rank, param.num_sample, param.num_local);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_LABEL_MAPPING_INL_H_
