/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <vector>

#include "tiny_dnn/core/framework/op_kernel.h"

namespace tiny_dnn {

class Conv2dOpenCLForwardOp : public core::OpKernel {
 public:
  explicit Conv2dOpenCLForwardOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) 
  {
	  // mgu
	  if (OpKernel::device_ != nullptr) {
		  auto params = OpKernel::params_->conv();
		  init_opencl(OpKernel::device_, params);
	  }
  }

  // mgu
  void init_opencl(const Device *device, const core::conv_params &params) {
  }

  void compute(core::OpKernelContext &context) override {
#if defined(USE_OPENCL) || defined(USE_CUDA)
    auto params = OpKernel::params_->conv();

    // incoming/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t &bias    = context.input(2);
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    context.Layer()->layer_type();
    // retrieve program from register
    CLCudaAPI::Program program = ProgramManager::getInstance().program(
      Program(context.device(), context.Layer()->layer_type()));
    nn_warn("Got Program");

    // Creates the kernel from the compiled program and sets the three
    // arguments.
    // Note that the indices of the arguments have to be set according to
    // their
    // order in the kernel.
    printCLPrograms(program);
    auto kernel = CLCudaAPI::Kernel(program, "CFMulti");
    nn_warn("Got Kernel");

    tiny_dnn::Device *device = context.device();
    CLCudaAPI::Context ctx   = context.device()->context();
    CLCudaAPI::Queue queue   = context.device()->queue();

    // pass connect table to kernel - mgu
    std::vector<cl_uchar> connect_table;
    if (params.tbl.is_empty())
    {
      connect_table.assign(params.in.depth_*params.out.depth_, 1);
    }
    else
    {
      connect_table.resize(params.tbl.cols_ * params.tbl.rows_);
      std::transform(connect_table.begin(), connect_table.end(), params.tbl.connected_.begin(), 
        [](auto a) {return a ? 1 : 0; });
    }
    auto connect_table_buf = CLCudaAPI::Buffer<cl_uchar>(ctx, queue, connect_table.begin(),
      connect_table.end());


    auto dev_W =
      CLCudaAPI::Buffer<float_t>(ctx, queue, W[0].begin(), W[0].end());
    auto dev_bias =
      CLCudaAPI::Buffer<float_t>(ctx, queue, bias[0].begin(), bias[0].end());

    auto dev_in = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kReadOnly, in_data.size()*in_data[0].size());
    auto dev_out = CLCudaAPI::Buffer<float_t>(ctx, CLCudaAPI::BufferAccess::kWriteOnly, out_data.size()*out_data[0].size());

    for (size_t i = 0; i < in_data.size(); ++i) {
      dev_in.WriteAsync(queue, in_data[0].size(), &in_data[i][0], in_data[0].size()*i);
    }

    kernel.SetArgument(0, dev_in);    // image_data
    kernel.SetArgument(1, 0);         // image_offset
    kernel.SetArgument(2, dev_W);     // kernel_data
    kernel.SetArgument(3, 0);         // kernel_offset
    kernel.SetArgument(4, dev_bias);  // bias
    kernel.SetArgument(5, 0);         // bias_offset
    kernel.SetArgument(6, dev_out);   // convolved_image
    kernel.SetArgument(7, 0);         // convolved_image_offset

    kernel.SetArgument(8, static_cast<cl_ushort>(params.in.width_));  // WIDTH
    kernel.SetArgument(9,
      static_cast<cl_ushort>(params.in.height_));  // HEIGHT
    kernel.SetArgument(
      10,
      static_cast<cl_ushort>(params.out.width_));  // OUTPUT_W
    kernel.SetArgument(
      11, static_cast<cl_ushort>(params.out.height_));  // OUTPUT_H


    kernel.SetArgument(12, connect_table_buf);  // connect table
    kernel.SetArgument(
      13, static_cast<cl_ushort>(params.in.depth_));  // DEPTH

    // We make sure that work group size is multiple of 16
    // auto global = std::vector<size_t>{params.in.width_, params.in.height_, params.in.depth_};
    auto local = std::vector<size_t>{ params.out.width_, params.out.height_, 1 };
    auto global = std::vector<size_t>{ params.out.width_* in_data.size(), params.out.height_, params.out.depth_ };

    assert(local[0] * local[1] * local[2] <= device->device().MaxWorkGroupSize());

    // Creates a new CLCudaAPI event to be able to time kernels
    auto event = CLCudaAPI::Event();

    // Enqueues the kernel and waits for the result.
    // Note that launching the kernel is always a-synchronous and thus
    // requires finishing the queue in order to complete the operation.
    nn_info("## Running the kernel ...");

    kernel.Launch(queue, global, local, event.pointer());
    queue.Finish(event);

    nn_info(" > Took " + to_string(event.GetElapsedTime()) + " ms");

    // Upload data GPU -> CPU
    for (size_t i = 0; i < in_data.size(); ++i) {
      dev_out.ReadAsync(queue, out_data[0].size(), &out_data[i][0], out_data[0].size()*i);
    }

    //// FOR DEBUG ONLY
    //nn_warn("output kernel");
    //for (size_t j = 0; j < out.size(); ++j) {
    //  std::cout << out[j] << " ";
    //}
    //std::cout << std::endl;

    // FOR DEBUG ONLY
    if (0)
    {
      nn_warn("output kernel:\n");
      for (size_t i = 0; i < 2/*out_data.size()*/; ++i) {
        for (size_t j = 0; j < out_data[i].size(); ++j) {
          std::cout << out_data[i][j] << " ";
          if ((j + 1) % 28 == 0)
          {
            std::cout << std::endl;
            if ((j + 1) % (28 * 28) == 0)
            {
              std::cout << std::endl;
            }
          }
        }
        std::cout << std::endl;
      }
    }

    // copy back
    //std::copy(std::begin(out), std::end(out),out_data[i].begin());

    int ii = 10;

#else
    CNN_UNREFERENCED_PARAMETER(context);
    throw nn_error("Not compiled with OpenCL");
#endif
  }
};

class Conv2dOpenCLBackwardOp : public core::OpKernel {
 public:
  explicit Conv2dOpenCLBackwardOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    CNN_UNREFERENCED_PARAMETER(context);
    nn_error("Not implemented yet.");
  }
};

}  // namespace tiny_dnn
