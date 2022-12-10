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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_FORMAT_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SC_DATA_FORMAT_HPP

#include <array>
#include <ostream>
#include <vector>
#include <common/dimensions.hpp>
#include <unordered_map>
#include <util/def.hpp>
#include <util/hash_utils.hpp>

namespace sc {

/// Memory format kind
enum class sc_format_category {
    // any means: support block and plain
    any,
    // blocked means: only support block
    blocked,
    // vnni blocked means: specific block format used by vnni instructions
    vnni_blocked,
    // non_blocking means: plain or permuted
    non_blocking
};

/**
 * The encoded data format kind. It stores the mapping of each axis in the real
 * shape with the axis in the original shape. We interpret the 64-bit storage as
 * 16x 4-bit ints. We use the last 4-bit int (15-th) as the control block
 * [slot0],[slot1],[slot2],...[slot15]
 * The slots 0~14 (15 slots) store the original axis index of the corresponding
 * dimension. For a N-dimension format, any slots with index >=N should contain
 * a value of (0xF). For example, NCHWc =>[0,1,2,3,1,-1,-1,...]
 * The slot15 is a control block, which indicates whether the format is in batch
 * mode. The batch mode means that we only care about the last a few dimensions
 * in the real/original shape. The original axis with index 0 in batch mode is
 * the leftmost axis in the original shape we are interested.
 */
struct SC_API sc_data_format_kind_t {
    static constexpr int NUM_SLOTS = 16;
    static constexpr int MAX_DIMS = NUM_SLOTS - 1; // 15
    static constexpr int BITS_PER_SLOT = sizeof(uint64_t) * 8 / NUM_SLOTS; // 4
    static constexpr int UNDEF_DIM = (1 << BITS_PER_SLOT) - 1; // 0xf
    uint64_t storage_;
    // gets the original axis of the idx-th dimemsion of the format
    constexpr int get(int idx) const {
        return 0xf & (storage_ >> (idx * BITS_PER_SLOT));
    }
    constexpr int get_control_block() const { return get(MAX_DIMS); }
    constexpr bool is_batch_format() const { return get_control_block() == 1; }
    void set(int idx, int data) { storage_ = set_ith_int(storage_, idx, data); }
    constexpr sc_data_format_kind_t(uint64_t storage) : storage_(storage) {}
    constexpr sc_data_format_kind_t() : storage_(0xffffffffffffffff) {}

private:
    static constexpr uint64_t set_ith_int(uint64_t oldv, int idx, int data) {
        return (oldv & ~(uint64_t(UNDEF_DIM) << (idx * BITS_PER_SLOT)))
                | (uint64_t(data) << (idx * BITS_PER_SLOT));
    }
    template <int start, typename... Args>
    static constexpr uint64_t make_storage(uint64_t oldv, int v, Args... args) {
        return make_storage<start + 1>(set_ith_int(oldv, start, v), args...);
    }

    template <int start>
    static constexpr uint64_t make_storage(uint64_t oldv, int v) {
        return set_ith_int(oldv, start, v);
    }

public:
    template <typename... Args>
    constexpr sc_data_format_kind_t(bool is_batch, Args... args)
        : storage_(make_storage<0>(
                set_ith_int(0xffffffffffffffff, MAX_DIMS, (int)is_batch),
                args...)) {
        static_assert(sizeof...(args) <= MAX_DIMS,
                "At most 15 dimensions are supported");
    }

    sc_data_format_kind_t(bool is_batch, const std::vector<int> &storage_args);

    constexpr sc_data_format_kind_t(const sc_data_format_kind_t &) = default;
    sc_data_format_kind_t &operator=(const sc_data_format_kind_t &) = default;
    constexpr bool operator==(const sc_data_format_kind_t &other) const {
        return storage_ == other.storage_;
    }

    constexpr bool operator!=(const sc_data_format_kind_t &other) const {
        return storage_ != other.storage_;
    }

    constexpr operator uint64_t() const { return storage_; }

    // gets the number of dimensions. For any, returns -1. For batch format,
    // returns the number of dims specified by the format. e.g. X_YZyz => 4
    int ndims() const;

    // gets the number of original dimensions. For any, returns -1. For batch
    // format, returns the number of dims specified by the format. e.g. X_YZyz
    // => 2
    int norig_dims() const;

    // checks if the format is valid. If not, throws an runtime_exception
    void check() const;

    bool is_plain() const;
    bool is_blocking() const;

    // collects the number of axies in the format. For original axis `i`,
    // `out[i]` will be the number of occurence of the axis in this format.
    // e.g. NCHWc => out=[1,2,1,1], the axis C occurs twice
    void collect_dim_count(int out[MAX_DIMS]) const;

    // collects the index of blocking with given `axis`. e.g. NCHWc and given
    // `axis=1` we get the blocking index vector {0}
    std::vector<int> collect_blocking_index(int axis) const;

    // collects the mapping from plain axis to blocking axis for each dimension.
    // e.g. NCHWc will return {{0},{1,4},{2},{3}}, MKmk will return
    // {{0,2},{1,3}}
    std::vector<std::vector<int>> collect_p2b_mapping() const;

    sc_data_format_kind_t to_plain() const;

    // makes an N-D plain format.
    static sc_data_format_kind_t get_plain_by_dims(size_t ndims);
    // makes a format that 2d blocking are at the lowest 2 dimensions. e.g. if
    // ndims=4, is_vnni_format=false, format is ABCDcd. if ndims=5,
    // is_vnni_format=false, then the format is ABCDEde.
    static sc_data_format_kind_t get_2dblocking_by_dims(
            size_t ndims, bool is_vnni_format = false);
};

namespace format_kinds {
#define SC_DEF_FMT(name, batch, ...) \
    constexpr sc_data_format_kind_t name {batch, __VA_ARGS__};
/* this format means continous memory format, which can be converted to
         any format*/
SC_DEF_FMT(any, 0xffffffffffffffff)
SC_DEF_FMT(A, false, 0)
SC_DEF_FMT(AB, false, 0, 1)
SC_DEF_FMT(BA, false, 1, 0)
SC_DEF_FMT(ABC, false, 0, 1, 2)
SC_DEF_FMT(ABCD, false, 0, 1, 2, 3)
SC_DEF_FMT(ABCDE, false, 0, 1, 2, 3, 4)

// special formats: X means any number of axises for batch, Y and Z
// means the last two axises in plain dims
SC_DEF_FMT(X_YZ, true, 0, 1)
SC_DEF_FMT(X_ZY, true, 1, 0)

// blocked format start
SC_DEF_FMT(Aa, false, 0, 0)
SC_DEF_FMT(ABab, false, 0, 1, 0, 1)
SC_DEF_FMT(ABba, false, 0, 1, 1, 0)
SC_DEF_FMT(BAab, false, 1, 0, 0, 1)
SC_DEF_FMT(ABCDb, false, 0, 1, 2, 3, 1)
SC_DEF_FMT(ABCDba, false, 0, 1, 2, 3, 1, 0)
// for bert
SC_DEF_FMT(ABDCcd, false, 0, 1, 3, 2, 2, 3)
SC_DEF_FMT(ABDCcdc, false, 0, 1, 3, 2, 2, 3, 2)
SC_DEF_FMT(ABCDdcd, false, 0, 1, 2, 3, 3, 2, 3)
SC_DEF_FMT(ABCDEb, false, 0, 1, 2, 3, 4, 1)
SC_DEF_FMT(ABCDEba, false, 0, 1, 2, 3, 4, 1, 0)

// special formats: see X_ZY
SC_DEF_FMT(X_YZyz, true, 0, 1, 0, 1)
SC_DEF_FMT(X_ZYyz, true, 1, 0, 0, 1)

// vnni format
SC_DEF_FMT(KCRSckc, false, 0, 1, 2, 3, 1, 0, 1)
SC_DEF_FMT(KCDRSckc, false, 0, 1, 2, 3, 4, 1, 0, 1)
SC_DEF_FMT(NKknk, false, 1, 0, 0, 1, 0)
SC_DEF_FMT(BNKknk, true, 1, 0, 0, 1, 0)

// used for bertBMM
SC_DEF_FMT(ACBD, false, 0, 2, 1, 3)
SC_DEF_FMT(ABCDdc, false, 0, 1, 2, 3, 3, 2)
SC_DEF_FMT(ABCDcd, false, 0, 1, 2, 3, 2, 3)
SC_DEF_FMT(ACBDdc, false, 0, 2, 1, 3, 3, 2)
SC_DEF_FMT(ACBDcd, false, 0, 2, 1, 3, 2, 3)
SC_DEF_FMT(ACBDcdc, false, 0, 2, 1, 3, 2, 3, 2)

constexpr auto NCHW = ABCD, KCRS = ABCD, NKHW = ABCD, MK = AB, KN = AB, NK = BA,
               MN = AB, BMK = X_YZ, BKN = X_YZ, NCHWc = ABCDb, NKHWk = ABCDb,
               KCRSck = ABCDba, MKmk = ABab, NKkn = BAab, MNmn = ABab,
               BMKmk = X_YZyz, BNKkn = X_ZYyz, BMNmn = X_YZyz, NCDHW = ABCDE,
               KCDRS = ABCDE, NCDHWc = ABCDEb, KCDRSck = ABCDEba;
#undef SC_DEF_FMT
}; // namespace format_kinds

struct SC_API sc_data_format_t {
    using blocking_t = std::array<int, 4>;
    sc_data_format_t() : format_code_(format_kinds::any), blocks_ {0} {}
    constexpr sc_data_format_t(
            sc_data_format_kind_t format_code, const blocking_t &blocks = {0})
        : format_code_(format_code), blocks_(blocks) {}

    sc_data_format_t(bool is_batch, const std::vector<int> &storage_args,
            const blocking_t &blocks = {0})
        : format_code_(is_batch, storage_args), blocks_(blocks) {}

    bool operator==(const sc_data_format_t &other) const {
        return format_code_ == other.format_code_ && blocks_ == other.blocks_;
    }
    bool operator!=(const sc_data_format_t &other) const {
        return !(*this == other);
    }
    bool is_convertible(const sc_data_format_t &other) const;

    bool is_blocking() const;

    bool is_plain() const;

    bool is_any() const;

    sc_data_format_t to_plain() const;

    sc_format_category get_format_category() const;

    constexpr static inline sc_data_format_t NCHW() {
        return sc_data_format_t(format_kinds::NCHW);
    }
    constexpr static inline sc_data_format_t NCHWc(int c) {
        return sc_data_format_t(format_kinds::NCHWc, {c});
    }
    constexpr static inline sc_data_format_t KCRS() {
        return sc_data_format_t(format_kinds::KCRS);
    }
    constexpr static inline sc_data_format_t KCRSck(int c, int k) {
        return sc_data_format_t(format_kinds::KCRSck, {c, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KCRSck2c(int c, int k) {
        return sc_data_format_t(format_kinds::KCRSckc, {c, k, 2});
    }
    constexpr static inline sc_data_format_t KCRSck4c(int c, int k) {
        return sc_data_format_t(format_kinds::KCRSckc, {c, k, 4});
    }
    constexpr static inline sc_data_format_t MK() {
        return sc_data_format_t(format_kinds::MK);
    }
    constexpr static inline sc_data_format_t BMK() {
        return sc_data_format_t(format_kinds::BMK);
    }
    constexpr static inline sc_data_format_t MKmk(int m, int k) {
        return sc_data_format_t(format_kinds::MKmk, {m, k, 0, 0});
    }
    constexpr static inline sc_data_format_t BMKmk(int m, int k) {
        return sc_data_format_t(format_kinds::BMKmk, {m, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KN() {
        return sc_data_format_t(format_kinds::KN);
    }
    constexpr static inline sc_data_format_t NK() {
        return sc_data_format_t(format_kinds::NK);
    }
    constexpr static inline sc_data_format_t BKN() {
        return sc_data_format_t(format_kinds::BKN);
    }
    constexpr static inline sc_data_format_t NKkn(int k, int n) {
        return sc_data_format_t(format_kinds::NKkn, {k, n, 0, 0});
    }
    constexpr static inline sc_data_format_t NKkn2k(int k, int n) {
        return sc_data_format_t(format_kinds::NKknk, {k, n, 2});
    }
    constexpr static inline sc_data_format_t NKkn4k(int k, int n) {
        return sc_data_format_t(format_kinds::NKknk, {k, n, 4});
    }
    constexpr static inline sc_data_format_t BNKkn(int k, int n) {
        return sc_data_format_t(format_kinds::BNKkn, {k, n, 0, 0});
    }
    constexpr static inline sc_data_format_t BNKkn2k(int k, int n) {
        return sc_data_format_t(format_kinds::BNKknk, {k, n, 2});
    }
    constexpr static inline sc_data_format_t BNKkn4k(int k, int n) {
        return sc_data_format_t(format_kinds::BNKknk, {k, n, 4});
    }
    constexpr static inline sc_data_format_t NCDHW() {
        return sc_data_format_t(format_kinds::NCDHW);
    }
    constexpr static inline sc_data_format_t NCDHWc(int c) {
        return sc_data_format_t(format_kinds::NCDHWc, {c});
    }
    constexpr static inline sc_data_format_t KCDRS() {
        return sc_data_format_t(format_kinds::KCDRS);
    }
    constexpr static inline sc_data_format_t KCDRSck(int c, int k) {
        return sc_data_format_t(format_kinds::KCDRSck, {c, k, 0, 0});
    }
    constexpr static inline sc_data_format_t KCDRSck2c(int c, int k) {
        return sc_data_format_t(format_kinds::KCDRSckc, {c, k, 2, 0});
    }
    constexpr static inline sc_data_format_t KCDRSck4c(int c, int k) {
        return sc_data_format_t(format_kinds::KCDRSckc, {c, k, 4, 0});
    }

    sc_data_format_kind_t format_code_;
    // The blocking numbers. It stores the blocking of the blocking axes in
    // the format_code_ from left to right. At most 4 blocking numbers can be
    // stored. Unused slots should be 0. For example, for format NK16k8n4k, the
    // blocks_ should be {16,8,4,0}. std::vector is unnecessary for block info.
    // And in g++, sizeof(vector<int>)==24, while static array only takes 16
    // bytes
    blocking_t blocks_;
    int get_blocks_size() const;
    bool is_same_format_kind(const sc_data_format_t &input_format) const;

    // {plain_axis, block}
    std::unordered_map<int, std::vector<int>> get_blocked_axis() const;

    static sc_dims get_reordered_shapes(const sc_dims &input_shapes,
            const sc_data_format_t &input_format,
            const sc_data_format_t &output_format);
    // given plain shapes and the data format, gets the real blocking shapes
    static sc_dims get_blocking_shapes(
            const sc_dims &plain_shapes, const sc_data_format_t &format);
    // given real blocking shapes and the data format, infers plain shapes. Note
    // that if there is padding when converting plain shapes and format to
    // blocking shapes, we cannot infer the original plain shapes from the
    // padded blocking shapes and the format
    static sc_dims get_padded_plain_shapes(
            const sc_dims &real_shapes, const sc_data_format_t &format);

    // gets an N-D plain format
    static sc_data_format_t get_plain_by_dims(size_t shape_size) {
        return sc_data_format_t(
                sc_data_format_kind_t::get_plain_by_dims(shape_size));
    }

    void to_string(std::ostream &os) const;
};

SC_INTERNAL_API std::ostream &operator<<(
        std::ostream &os, const sc_data_format_t &in);

} // namespace sc

namespace std {
template <>
struct hash<sc::sc_data_format_t> {
    std::size_t operator()(const sc::sc_data_format_t &k) const;
};
} // namespace std

#endif
