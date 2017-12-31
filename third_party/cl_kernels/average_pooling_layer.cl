
// ss << "#define OUTPUT_SIZE " << params_.out_size_ << "\n";
// ss << "#define INPUT_SIZE " << params_.in_size_ << "\n";
// ss << "#define HAS_BIAS " << params_.has_bias_ << "\n";

__kernel void FullyConnected(__global Dtype* in_data, int_tp IN_OFFSET,
  __constant Dtype* kernel_data, int_tp KERNEL_OFFSET,
  __constant Dtype* bias, const int_tp BIAS_OFFSET,
  __global Dtype* out_data, const int_tp OUT_OFFSET
) {
}

__kernel void FullyConnected_bprop(__global Dtype* prev_out, __global Dtype* W,
  __global Dtype* curr_delta, __global Dtype* out_prev_delta,
  __global Dtype* out_dW, __global Dtype* out_db
) {

}
