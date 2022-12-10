/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef INTERFACE_TYPE_CONSTRAINT_HPP
#define INTERFACE_TYPE_CONSTRAINT_HPP

#include "interface/op.hpp"

namespace dnnl {
namespace graph {
namespace impl {

bool check_bn_fwd_data_type(const op_t *n);

bool check_bn_bwd_data_type(const op_t *n);

bool check_ln_data_type(const op_t *n);

bool check_typecast_data_type(const op_t *n);

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
