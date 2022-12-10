/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_SCHEDULE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_SCHEDULE_HPP

#include <utility>
#include "../function_pass.hpp"
#include <compiler/config/context.hpp>

namespace sc {

namespace attr_keys {
// the buffer scheduler type: 0 - no buffer schedule, 1 - whole buffer reuse, 2
// - static memory planner (minimize size), 3 - static memory planner (hot
// memory first)
constexpr const char *buf_sched_type = "pass.buf_sched_type";
// hint tick info for tensors in loops from graph or fusion mgr: int64_t. It
// guides the buffer scheduler to correctly compute the tensor life time. Buffer
// scheduler will it add to current tick to calculate final tick
constexpr const char *hint_first_access_tick = "pass.hint_first_access_tick";
constexpr const char *hint_last_access_tick = "pass.hint_last_access_tick";
constexpr const char *tsr_dont_buf_sched = "pass.tsr_dont_buf_sched";
constexpr int BUF_SCHED_NONE = 0;
constexpr int BUF_SCHED_WHOLE = 1;
constexpr int BUF_SCHED_SIZE = 2;
constexpr int BUF_SCHED_HOT = 3;
} // namespace attr_keys

/**
 * Schedule tensor buffers to reuse them if they are no longer needed.
 * This pass should only work on 1D tensors. It should be placed after
 * index_flatten
 *
 * 1) We sort all the expressions by execution order and all exprs are assigned
 * a tick. A greater tick means that the expr will be executed later than other
 * expr with less tick.
 *
 * 2) First collect the last-read-tick (LRT), all write ticks (in writes_ set)
 * and first-access-tick (FAT), creation tick, deletion tick for each tensor. We
 * collect these ticks on indexing_nodes, and functions calls. To distinguish
 * writes from reads, we also process assign_nodes (lvalues are written). The
 * function arguments can be annotated with "read_buffer" and "write_buffer" in
 * the function declaration. If no annotation is applied on an argument, the
 * tensor is considered read-written. Special case for "for_loop":
 * the tensors in a for-loop will be accessed mutiple times in "body_" and
 * "iter_end_". We manually set ticks of all tensors accessed in a for-loop to
 * the tick at the end of the loop.
 *
 * 3) Optionally (if eliminate_dead_writes_=true), remove all writes to local
 * tensors which is no longer read, where tick > tensor.LRT
 *
 * 4) Schedule the tensors. For each defined local tensors (in tensor creation
 * order), say, "cur", find another local defined/ function arg tensor, say
 * "candidate", where:
 *  1. cur.FAT > candidate.LRT && cur.FAT >= candidate.creation_tick &&
 * cur.deletion_tick <= candidate.deletion_tick.
 *  2. in the tick set candidate.writes, there are no writes to the candidates
 * that happens between [cur.FAT, cur.LRT].
 *  3. If the candidate is an function argument, make sure that cur writes will
 * not overwrite the candidate's final values: cur.last_write < candidate.FAT
 *
 * If such candidate is found, replace cur with the candidate
 *
 * 5) if "cur" is larger than "candidate" in size, extend candidate
 * */
class buffer_scheduler_t : public function_pass_t {
public:
    context_ptr ctx_;
    bool eliminate_dead_writes_;
    buffer_scheduler_t(context_ptr ctx, bool eliminate_dead_writes)
        : ctx_(std::move(ctx)), eliminate_dead_writes_(eliminate_dead_writes) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f) const;
};
// todo: if the buffer ("candidate") is larger than the "cur" tensor, we can
// split "candidate" tensor into two and reuse the remaining of it for other
// tensors

} // namespace sc

#endif
