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

inline void fully_connected_op_opencl(core::OpKernelContext &context,
                                        const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
  // retrieve program from register
  // mgu: kernel_string may be a big string, and may cause performance problem
  CLCudaAPI::Program program = ProgramManager::getInstance().program(
    Program(context.device(), context.Layer()->layer_type(), std::move(context.Layer()->kernel_string())));
  nn_warn("Got Program");

  // Creates the kernel from the compiled program and sets the arguments.
  printCLPrograms(program);
  auto kernel = CLCudaAPI::Kernel(program, "FullyConnected");
  nn_warn("Got Kernel");

  tiny_dnn::Device *device = context.device();
  CLCudaAPI::Context ctx = context.device()->context();
  CLCudaAPI::Queue queue = context.device()->queue();

  auto dev_W = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, W.size());
  dev_W.WriteAsync(queue, W.size(), &W[0], 0);
  auto dev_bias = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, bias.size());
  dev_bias.WriteAsync(queue, bias.size(), &bias[0], 0);

  auto dev_in = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, in_data.size()*in_data[0].size());
  auto dev_out = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kWriteOnly, out_data.size()*out_data[0].size());

  for (size_t i = 0; i < in_data.size(); ++i) {
    dev_in.WriteAsync(queue, in_data[0].size(), &in_data[i][0], in_data[0].size()*i);
  }

  kernel.SetArgument(0, dev_in);    // in_data
  kernel.SetArgument(1, 0);         // IN_OFFSET
  kernel.SetArgument(2, dev_W);     // kernel_data
  kernel.SetArgument(3, 0);         // KERNEL_OFFSET
  kernel.SetArgument(4, dev_bias);  // bias
  kernel.SetArgument(5, 0);         // BIAS_OFFSET
  kernel.SetArgument(6, dev_out);   // out_data
  kernel.SetArgument(7, 0);         // OUT_OFFSET

  auto local = std::vector<size_t>{ params.out_size_, 1 };
  assert(local[0] * local[1] <= device->device().MaxWorkGroupSize());

  assert(in_data.size() == out_data.size());

  // Creates a new CLCudaAPI event to be able to time kernels
  auto event = CLCudaAPI::Event();
  nn_info("## Running the kernel ...");

  auto global = std::vector<size_t>{ params.out_size_, in_data.size() };

  // Enqueues the kernel and waits for the result.
  // Note that launching the kernel is always a-synchronous and thus
  // requires finishing the queue in order to complete the operation.
  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);

  nn_info(" > Took " + to_string(event.GetElapsedTime()) + " ms");

  // Upload data GPU -> CPU
  for (size_t i = 0; i < in_data.size(); ++i) {
    dev_out.ReadAsync(queue, out_data[0].size(), &out_data[i][0], out_data[0].size()*i);
  }

  // FOR DEBUG ONLY
  if (0)
  {
    nn_warn("output kernel:\n");
    std::cout << "Fully connected: CPU output: " << std::endl;
    for (size_t i = 0; i < std::min<size_t>(2, out_data.size()); ++i) {
      for (size_t j = 0; j < out_data[i].size(); ++j) {
        std::cout << out_data[i][j] << " ";
        if ((j + 1) % 16 == 0)
        {
          std::cout << std::endl;
        }
      }
      std::cout << std::endl;
    }
  }


#endif
}

inline void fully_connected_op_opencl(core::OpKernelContext &context,
                                        const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const core::fully_params &params,
                                        const bool layer_parallelize) {
#if defined(USE_OPENCL) || defined(USE_CUDA)
  //// retrieve program from register
  //// mgu: kernel_string may be a big string, and may cause performance problem
  //CLCudaAPI::Program program = ProgramManager::getInstance().program(
  //  Program(context.device(), context.Layer()->layer_type(), std::move(context.Layer()->kernel_string())));
  //nn_warn("Got Program");

  //// Creates the kernel from the compiled program and sets the arguments.
  //printCLPrograms(program);
  //auto kernel = CLCudaAPI::Kernel(program, "FullyConnected_bprop");
  //nn_warn("Got Kernel");

  //tiny_dnn::Device *device = context.device();
  //CLCudaAPI::Context ctx = context.device()->context();
  //CLCudaAPI::Queue queue = context.device()->queue();

  //auto dev_W = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, W.size());
  //dev_W.WriteAsync(queue, W.size(), &W[0], 0);
  //auto dev_prev_out = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, prev_out.size()*prev_out[0].size());
  //for (size_t i = 0; i < prev_out.size(); ++i) {
  //  dev_prev_out.WriteAsync(queue, prev_out[0].size(), &prev_out[i][0], prev_out[0].size()*i);
  //}
  //auto dev_curr_delta = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, curr_delta.size()*curr_delta[0].size());
  //for (size_t i = 0; i < curr_delta.size(); ++i) {
  //  dev_prev_out.WriteAsync(queue, curr_delta[0].size(), &curr_delta[i][0], curr_delta[0].size()*i);
  //}

  //auto dev_prev_delta = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kWriteOnly, prev_delta.size()*prev_delta[0].size());
  //auto dev_db = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, db.size());
  //auto dev_dW = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, dW.size());


  //kernel.SetArgument(0, dev_prev_out);    // 
  //kernel.SetArgument(1, dev_W);           // 
  //kernel.SetArgument(2, dev_curr_delta);  // 
  //kernel.SetArgument(3, dev_prev_delta);  // 
  //kernel.SetArgument(4, dev_dW);          // 
  //kernel.SetArgument(5, dev_db);          // 

  //auto local = std::vector<size_t>{ params.in_size_, 1 };
  //assert(local[0] * local[1] <= device->device().MaxWorkGroupSize());
  //auto global = std::vector<size_t>{ params.in_size_, prev_delta.size() };

  //// Creates a new CLCudaAPI event to be able to time kernels
  //auto event = CLCudaAPI::Event();
  //nn_info("## Running the kernel ...");


  //// Enqueues the kernel and waits for the result.
  //// Note that launching the kernel is always a-synchronous and thus
  //// requires finishing the queue in order to complete the operation.
  //kernel.Launch(queue, global, local, event.pointer());
  //queue.Finish(event);

  //nn_info(" > Took " + to_string(event.GetElapsedTime()) + " ms");

  //// Upload data GPU -> CPU
  //for (size_t i = 0; i < prev_delta.size(); ++i) {
  //  dev_prev_delta.ReadAsync(queue, prev_delta[0].size(), &prev_delta[i][0], prev_delta[0].size()*i);
  //}

  //// FOR DEBUG ONLY
  //if (0)
  //{
  //  nn_warn("output kernel:\n");
  //  for (size_t i = 0; i < 16/*out_data.size()*/; ++i) {
  //    for (size_t j = 0; j < prev_delta[i].size(); ++j) {
  //      std::cout << prev_delta[i][j] << " ";
  //    }
  //    std::cout << std::endl;
  //  }
  //}

  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                          r.end() - r.begin(),
                          &dW[sample][c * params.out_size_ + r.begin()]);
      }

      if (params.has_bias_) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] += curr_delta[sample][i];
        }
      }
    });
  }
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
