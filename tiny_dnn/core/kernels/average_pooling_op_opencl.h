/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {
// forward_propagation
inline void tiny_average_pooling_kernel_opencl(
  bool parallelize,
  const std::vector<tensor_t *> &in_data,
  std::vector<tensor_t *> &out_data,
  const shape3d &out_dim,
  float_t scale_factor,
  std::vector<typename partial_connected_layer::wi_connections> &out2wi) {
  for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
    const vec_t &in = (*in_data[0])[sample];
    const vec_t &W  = (*in_data[1])[0];
    const vec_t &b  = (*in_data[2])[0];
    vec_t &out      = (*out_data[0])[sample];

    auto oarea = out_dim.area();
    size_t idx = 0;
    for (size_t d = 0; d < out_dim.depth_; ++d) {
      float_t weight = W[d] * scale_factor;
      float_t bias   = b[d];
      for (size_t i = 0; i < oarea; ++i, ++idx) {
        const auto &connections = out2wi[idx];
        float_t value{0};
        for (auto connection : connections) value += in[connection.second];
        value *= weight;
        value += bias;
        out[idx] = value;
      }
    }

    assert(out.size() == out2wi.size());
  });
}

// back_propagation
inline void tiny_average_pooling_back_kernel_opencl(
  bool parallelize,
  const std::vector<tensor_t *> &in_data,
  const std::vector<tensor_t *> &out_data,
  std::vector<tensor_t *> &out_grad,
  std::vector<tensor_t *> &in_grad,
  const shape3d &in_dim,
  float_t scale_factor,
  std::vector<typename partial_connected_layer::io_connections> &weight2io,
  std::vector<typename partial_connected_layer::wo_connections> &in2wo,
  std::vector<std::vector<size_t>> &bias2out) {
  CNN_UNREFERENCED_PARAMETER(out_data);
  for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
    const vec_t &prev_out = (*in_data[0])[sample];
    const vec_t &W        = (*in_data[1])[0];
    vec_t &dW             = (*in_grad[1])[sample];
    vec_t &db             = (*in_grad[2])[sample];
    vec_t &prev_delta     = (*in_grad[0])[sample];
    vec_t &curr_delta     = (*out_grad[0])[sample];

    auto inarea = in_dim.area();
    size_t idx  = 0;
    for (size_t i = 0; i < in_dim.depth_; ++i) {
      float_t weight = W[i] * scale_factor;
      for (size_t j = 0; j < inarea; ++j, ++idx) {
        prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
      }
    }

    for (size_t i = 0; i < weight2io.size(); ++i) {
      const auto &connections = weight2io[i];
      float_t diff{0};

      for (auto connection : connections)
        diff += prev_out[connection.first] * curr_delta[connection.second];

      dW[i] += diff * scale_factor;
    }

    for (size_t i = 0; i < bias2out.size(); i++) {
      const std::vector<size_t> &outs = bias2out[i];
      float_t diff{0};

      for (auto o : outs) diff += curr_delta[o];

      db[i] += diff;
    }
  });
}


}  // namespace kernels
}  // namespace tiny_dnn
