
// ss << "#define OUTPUT_SIZE " << params_.out_size_ << "\n";
// ss << "#define INPUT_SIZE " << params_.in_size_ << "\n";
// ss << "#define HAS_BIAS " << params_.has_bias_ << "\n";

__kernel void FullyConnected(__global Dtype* in_data, int_tp IN_OFFSET,
  __constant Dtype* kernel_data, int_tp KERNEL_OFFSET,
  __constant Dtype* bias, const int_tp BIAS_OFFSET,
  __global Dtype* out_data, const int_tp OUT_OFFSET
) {
  const int_tp sample_id = get_group_id(1);

  const int_tp out_id = get_local_id(0);

  Dtype totalSum = 0.0f;
  for (int_tp i = 0; i < INPUT_SIZE; i++)
  {
    totalSum += kernel_data[KERNEL_OFFSET + OUTPUT_SIZE * i + out_id] * in_data[IN_OFFSET + INPUT_SIZE * sample_id + i];
  }

  if (HAS_BIAS)
  {
    totalSum += bias[BIAS_OFFSET + OUTPUT_SIZE * sample_id + out_id];
  }

  out_data[OUT_OFFSET + OUTPUT_SIZE * sample_id + out_id] = totalSum;
}