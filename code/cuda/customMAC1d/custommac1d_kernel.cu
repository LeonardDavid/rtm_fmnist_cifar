#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cstdint>

#include <curand.h>
#include <curand_kernel.h>

#define DEBUG_1D 0
#define DEBUG_THREAD_INFO_FLOAT32 0
#define DEBUG_THREAD_INFO_INT32 0
#define DEBUG_BITS 0
#define DEBUG_SEEDS 0

template <typename scalar_t>
__global__ void custommac1d_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output
  )
{

  // handle access indices
  const int c = blockIdx.x * blockDim.x + threadIdx.x; // y
  const int d = blockIdx.y * blockDim.y + threadIdx.y; // x

  // make sure we don't modify memory regions outside of output
  if ((d < output.size(0)) && (c < output.size(1)))
  {
    // curandState state;
    // curand_init(clock() + c, c, 0, &state);
    // random_numbers[c] = curand_uniform(&state[c]);
    // if(c==0 && d==0){
    //   printf("%f ", curand_uniform(&state));
    // }

    // this is (c,d,e), we have as many threads as we have pixels in output out
    // each thread of out calculates a MAC (row of filter times column of input)

    // every thread is responsible for one sum, there are as many threads as mac sums in output
    output[d][c] = 0;
    float mult_result = 0;
    for(int i = 0; i < weight.size(1); i++)
    {
      //printf("Thread: (%d,%d,%d)\nWeight: %.4f, Input: %.4f\n", c, d, e, weight[c][i], input[d][i][e]);
      
      // float w_prev=0, w_next=0, x_prev=0, x_next=0;
      // float w_used = weight[c][i];
      // float x_used = input[d][i];

      // if(i > 0){
      //   w_prev = weight[c][i-1];
      //   // x_prev = input[d][i-1];
      // }
      // else{
      //   w_prev = 0; // rand()
      // }

      // if(i < weight.size(1)-1){
      //   w_next = weight[c][i+1];
      // }
      // else{
      //   w_next = 0; // rand()
      // }

      // if(curand_uniform(&state) < 0.5){ // 10% prob for shift
      //   // printf("shift");
      //   if(c % 2 == 0){ // 50% chance of left or right shift
      //     // printf(" right\n");
      //     w_used = w_prev;
      //   }else{
      //     // printf(" left\n");
      //     w_used = w_next;
      //   }
      // }

      // mult_result = w_used * x_used;
      // output[d][c] += mult_result;


      mult_result = weight[c][i] * input[d][i];
      output[d][c] += mult_result;

      // if (d == 0 && c == 0)
      // {
      //   printf("f01: %.2f, f10: %.2f, seed0: %d, cantor_val: %d\n", f01, f10, seed0, cantor_val);
      //   //printf("CUDA shape of weight [%d]", weight.size(0));
      //   //printf("CUDA shape of input [%d,%d]",  input.size(0), input.size(1));
      //   //printf("CUDA shape of output [%d,%d]\n\n", output.size(0), output.size(1));
      // }
    }
  }
}

torch::Tensor custommac1d_cuda(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor output
) {
  // The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.
  // dim3 threadsPerBlock(8,8,8)
  // <<<number of blocks per grid, number of threads ber block>>>
  // grid is created with enough blocks to have one thread per matrix element

  // https://devtalk.nvidia.com/default/topic/1028226/how-many-concurrent-threads-are-running-on-my-geforce-gtx-1080-ti-/
  const int output_size_x = output.size(1);
  const int output_size_y = output.size(0);
  int threads_x = 16; // per block, 16
  int threads_y = 16; // per block, 16

  #if DEBUG_1D
    threads_x = 1;
    threads_y = 1;
  #endif

  const dim3 threads(threads_x,threads_y);
  const dim3 blocks((output_size_x + threads_x - 1) / threads_x,
                    (output_size_y + threads_y - 1) / threads_y);

  AT_DISPATCH_ALL_TYPES(input.type(), "custommac1d_cuda", ([&] {
    custommac1d_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        weight.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()
    );
  }));

  return output;
}
