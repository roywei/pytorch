/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_LOOP_TRANSFORM_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_LOOP_TRANSFORM_HPP
#include <vector>
#include <compiler/ir/sc_stmt.hpp>

namespace sc {
/**
 * Removes redundant loops with parallel attribute. Will reserve the outmost
 * one loop with parallel.
 * @param body the stmts for parallel remove
 * */
void remove_parallel(stmt body);

void remove_parallel(func_t body);

/**
 * Collect loops inside this body. Won't recurisvely look into loop body.
 * For example.
 *  for()     # loop 1
 *    for()   # loop 2
 *  for()     # loop 3
 *    for()   # loop 4
 * Only loop 1 and 3 are returned.
 * @param body the stmts for collection
 * */
std::vector<for_loop> collect_loops(stmt body);

/**
 * Collect nested loops inside this body.
 * For example.
 *  for()     # loop 1
 *    for()   # loop 2
 *      for() # loop 3
 *      for() # loop 4
 *
 * Only loop 1 and 2 are returned because loop 3 and 4 are not nested loop.
 *
 * @param body the stmts for collection
 * */
std::vector<for_loop> collect_nested_loops(stmt body);

// get inner for_loop
for_loop get_inner_for_loop(const for_loop_node_t *f);

} // namespace sc

#endif
