/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"
#include "tiny_dnn/core/framework/op_kernel.h"

namespace tiny_dnn {
namespace kernels {
// forward_propagation
inline void tiny_average_pooling_kernel_opencl(
  core::OpKernelContext &context,
  bool parallelize,
  const std::vector<tensor_t *> &in_data,
  std::vector<tensor_t *> &out_data,
  const shape3d &out_dim,
  float_t scale_factor,
  std::vector<typename partial_connected_layer::wi_connections> &out2wi) {


#if defined(USE_OPENCL) || defined(USE_CUDA)
  // retrieve program from register
  // mgu: kernel_string may be a big string, and may cause performance problem
  CLCudaAPI::Program program = ProgramManager::getInstance().program(
    Program(context.device(), context.Layer()->layer_type(), std::move(context.Layer()->kernel_string())));
  nn_warn("Got Program");

  // Creates the kernel from the compiled program and sets the arguments.
  printCLPrograms(program);
  auto kernel = CLCudaAPI::Kernel(program, "AveragePooling");
  nn_warn("Got Kernel");

  tiny_dnn::Device *device = context.device();
  CLCudaAPI::Context ctx = context.device()->context();
  CLCudaAPI::Queue queue = context.device()->queue();

  const vec_t &W = (*in_data[1])[0];
  auto dev_W = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, out_dim.depth_);
  dev_W.WriteAsync(queue, out_dim.depth_, &W[0], 0);
  const vec_t &b = (*in_data[2])[0];
  auto dev_bias = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, out_dim.depth_);
  dev_bias.WriteAsync(queue, out_dim.depth_, &b[0], 0);

  const size_t sample_num = (*in_data[0]).size();
  assert((*in_data[0]).size() == (*out_data[0]).size());

  const size_t input_size = (*in_data[0])[0].size();
  const size_t output_size = (*out_data[0])[0].size();

  auto dev_in = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, sample_num * input_size);
  auto dev_out = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kWriteOnly, sample_num * output_size);

  for (size_t i = 0; i < sample_num; ++i) {
    dev_in.WriteAsync(queue, input_size, &((*in_data[0])[i][0]), input_size*i);
  }

  kernel.SetArgument(0, dev_in);    // in_data
  kernel.SetArgument(1, dev_W);     // kernel_data
  kernel.SetArgument(2, dev_bias);  // bias
  kernel.SetArgument(3, scale_factor);  // 
  kernel.SetArgument(4, dev_out);   // out_data

  auto local = std::vector<size_t>{ out_dim.width_, out_dim.height_, 1 };
  assert(local[0] * local[1] * local[2] <= device->device().MaxWorkGroupSize());

  const int sample_sqrt = sqrtf(sample_num);
  assert(sample_sqrt*sample_sqrt == sample_num);
  auto global = std::vector<size_t>{ local[0] * sample_sqrt, local[1] * sample_sqrt, out_dim.depth_ };

  // Creates a new CLCudaAPI event to be able to time kernels
  auto event = CLCudaAPI::Event();
  nn_info("## Running the kernel ...");

  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);

  nn_info(" > Took " + to_string(event.GetElapsedTime()) + " ms");

  // Upload data GPU -> CPU
  for (size_t i = 0; i < (*out_data[0]).size(); ++i) {
    dev_out.ReadAsync(queue, output_size, &((*out_data[0])[i][0]), output_size*i);
  }

  // FOR DEBUG ONLY
  if (0)
  {
    nn_warn("output kernel:\n");
    std::cout << "Average pooling: GPU output: " << std::endl;
    for (size_t s = 0; s < std::min<size_t>(2, out_data.size()); ++s) {
      for (size_t j = 0; j < output_size; ++j) {
        std::cout << (*out_data[0])[s][j] << " ";
        if ((j+1)% out_dim.width_ ==0)
        {
          std::cout << std::endl;
        }
        if ((j+1) % (out_dim.width_ * out_dim.width_) == 0)
        {
          std::cout << std::endl;
        }
      }
      std::cout << std::endl;
    }
  }
#endif

}

// back_propagation
inline void tiny_average_pooling_back_kernel_opencl(
  core::OpKernelContext &context,
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
