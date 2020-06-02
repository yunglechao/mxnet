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
 *  Copyright (c) 2019 by Contributors
 * \file label_mapping.cc
 * \brief mapping sampled local label
 * \author YongleZhao, XiangAn
 */

#include "./label_mapping-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LabelMappingParam);

NNVM_REGISTER_OP(_contrib_label_mapping)
.describe(R"code(Set difference of two 1D arrays
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LabelMappingParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LabelMappingShape)
.set_attr<nnvm::FInferType>("FInferType", LabelMappingType)
.set_attr<FCompute>("FCompute<cpu>", LabelMapping<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"label", "local_index", "unique_sorted_global_label"};
  })
.add_argument("label", "NDArray-or-Symbol", "Batch label")
.add_argument("local_index", "NDArray-or-Symbol", "Local index this rank")
.add_argument("unique_sorted_global_label", "NDArray-or-Symbol", "Unique sorted global label")
.add_arguments(LabelMappingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
