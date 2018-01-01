    // ss << "#define IN_WIDTH " << in_.width_ << "\n";
    // ss << "#define IN_HEIGHT " << in_.height_ << "\n";
    // ss << "#define CHANNELS " << in_.depth_ << "\n";
    // ss << "#define OUT_WIDTH " << out_.width_ << "\n";
    // ss << "#define OUT_HEIGHT " << out_.height_ << "\n";
    // ss << "#define STRIDE_X " << stride_x_ << "\n";
    // ss << "#define STRIDE_Y " << stride_y_ << "\n";
    // ss << "#define KERNEL_X " << pool_size_x_ << "\n";
    // ss << "#define KERNEL_Y " << pool_size_y_ << "\n";


__kernel void AveragePooling(__global Dtype* in_data, 
  __constant Dtype* kernel_data,
  __constant Dtype* bias, 
  const Dtype scale_factor, 
  __global Dtype* out_data
) {
  const int_tp group_width = get_num_groups(0);
  const int_tp sample_idx = get_group_id(0);
  const int_tp sample_idy = get_group_id(1);
  
  const int_tp sample_id = group_width * sample_idy + sample_idx;
  
  const int_tp outputX = get_local_id(0);
  const int_tp outputY = get_local_id(1);
  const int_tp outputZ = get_group_id(2);
  
  const int_tp input_offset = IN_WIDTH * IN_HEIGHT * CHANNELS * sample_id + IN_WIDTH * IN_HEIGHT * outputZ + IN_WIDTH * STRIDE_Y * outputY + STRIDE_X * outputX;
  const int_tp output_offset = OUT_WIDTH * OUT_HEIGHT * CHANNELS * sample_id + OUT_WIDTH * OUT_HEIGHT * outputZ + OUT_WIDTH * outputY + outputX;
  
  const Dtype W = kernel_data[outputZ] * scale_factor;
  
  Dtype sum = 0;
  
#pragma unroll
  for(int_tp y = 0; y < KERNEL_Y; ++y)
  {
    for(int_tp x = 0; x < KERNEL_X; ++x)
	{
	  sum += in_data[input_offset + IN_WIDTH * y + x];
	}
  }
  
  sum *= W;
  sum += bias[outputZ];
  
  out_data[output_offset] = sum ;
}

__kernel void AveragePooling_bprop(__global Dtype* prev_out, __global Dtype* W,
  __global Dtype* curr_delta, __global Dtype* out_prev_delta,
  __global Dtype* out_dW, __global Dtype* out_db
) {

}
