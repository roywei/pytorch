/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef BINARY_HPP
#define BINARY_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/perf_report.hpp"

namespace binary {

using alg_t = attr_t::post_ops_t::kind_t;

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    prb_vdims_t prb_vdims;

    std::vector<std::vector<dnnl_data_type_t>> sdt {{dnnl_f32, dnnl_f32}};
    std::vector<dnnl_data_type_t> ddt {dnnl_f32};
    std::vector<std::vector<std::string>> stag {{tag::abx, tag::abx}};
    std::vector<std::string> dtag {tag::any};
    std::vector<alg_t> alg {alg_t::ADD};
    std::vector<bool> inplace {false};
    std::vector<attr_t::arg_scales_t> scales {attr_t::arg_scales_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};
    attr_t attr = {};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%sdt%,%ddt%,%stag%,%dtag%,%alg%,%attr%,"
              "%DESC%,%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t : public prb_vdims_t {
    prb_t(const prb_vdims_t &prb_vdims,
            const std::vector<dnnl_data_type_t> &sdt, dnnl_data_type_t ddt,
            const std::vector<std::string> &stag, std::string dtag, alg_t alg,
            bool inplace, const attr_t &attr)
        : prb_vdims_t(prb_vdims)
        , sdt(sdt)
        , ddt(ddt)
        , stag(stag)
        , dtag(dtag)
        , alg(alg)
        , inplace(inplace)
        , attr(attr) {}
    ~prb_t() {}

    std::vector<dnnl_data_type_t> sdt;
    dnnl_data_type_t ddt;
    std::vector<std::string> stag;
    std::string dtag;
    alg_t alg;
    bool inplace;
    attr_t attr;
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , stag_({})
        , dtag_(normalize_tag(p_->dtag, p_->ndims)) {
        for (size_t d = 0; d < p_->stag.size(); d++)
            stag_.push_back(normalize_tag(p_->stag[d], p_->ndims));
    }

    void dump_alg(std::ostream &s) const override { s << p_->alg; }

    void dump_desc(std::ostream &s) const override {
        s << static_cast<const prb_vdims_t &>(*p_);
    }

    void dump_desc_csv(std::ostream &s) const override { dump_desc(s); }

    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &p_->sdt;
    }
    const attr_t *attr() const override { return &p_->attr; }
    const dnnl_data_type_t *ddt() const override { return &p_->ddt; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

int setup_binary_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &mem_dt, std::vector<dnn_mem_t> &mem_fp,
        bool only_positive_values = false, bool only_integer_values = false);

void compute_ref(const prb_t *prb, const dnn_mem_t &src0, const dnn_mem_t &src1,
        const std::vector<dnn_mem_t> &binary_po, dnn_mem_t &dst);

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace binary

#endif
