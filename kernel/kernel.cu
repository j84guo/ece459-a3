// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void conv2d_relu(// 100x100
                                       const double *image,
                                       // 10x5x5
                                       const double *filters,
                                       // 10x20x20
                                       double *output) {
    // Index into flattened output array
    for (int output_i = blockIdx.x * blockDim.x + threadIdx.x;
         output_i < 4000;
         output_i += gridDim.x * blockDim.x) {

        // 5x5 filter
        const int filter_num = output_i / 400;
        const double *filter = filters + filter_num * 5 * 5;

        // 20x20 filter output
        const int filter_output_i = output_i % 400;
        const int filter_output_r = filter_output_i / 20;
        const int filter_output_c = filter_output_i % 20;

        // Top-left row, col in 5x5 image tile corresponding to filter output
        const int image_r = filter_output_r * 5;
        const int image_c = filter_output_c * 5;

        // Accumulate sum
        double sum = 0;
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                const int i = (image_r + r) * 100 + (image_c + c);
                const int j = r * 5 + c;
                sum += image[i] * filter[j];
            }
        }
        if (sum < 0) {
            sum = 0;
        }
        output[output_i] = sum;
    }
}

extern "C" __global__ void flattened_to_dense(// 10x20x20
                                              const double *conv2d_output,
                                              // 10x4000
                                              const double *weights,
                                              // 10xblockDim.x
                                              double *block_outputs) {
    // Output neuron
    const int output_neuron = blockIdx.y;
    if (output_neuron >= 10) {
        return;
    }

    const int num_blocks_per_output_neuron = gridDim.x;

    /*
    if (threadIdx.x != 0) {
        return;
    }
    if (blockIdx.x != 0){
        return;
    } 
    double sum = 0;
    for (int i = 0; i < 4000; i++) {
        sum += weights[output_neuron * 4000 + i] * conv2d_output[i];
    }
    block_outputs[output_neuron * num_blocks_per_output_neuron + 0] = sum;
    */

    // Cache, it is assumed that 4000 >= blockDim.x
    __shared__ double x[4000];

    // For each segment of the row that this block is responsible for
    for (int segment_start = blockIdx.x * blockDim.x; segment_start < 4000; segment_start += gridDim.x * blockDim.x) {
        const int segment_len = min(blockDim.x, 4000 - segment_start);

        // Populate cache with this segment
        if (threadIdx.x < segment_len) {
            x[threadIdx.x] = weights[output_neuron * 4000 + segment_start + threadIdx.x] * conv2d_output[segment_start + threadIdx.x];
        }
        __syncthreads();

        // Segment in cache
        double *segment = x;
        int right = segment_len - 1;
        while (0 < right) {
            const int mid = right / 2;
            if (((right + 1) % 2) == 0) {
                if (threadIdx.x <= mid) {
                    segment[threadIdx.x] += segment[right - threadIdx.x];
                }
            } else {
                if (threadIdx.x < mid) {
                    segment[threadIdx.x] += segment[right - threadIdx.x];
                }
            }
            right = mid;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int block_outputs_i = output_neuron * num_blocks_per_output_neuron + blockIdx.x;
            block_outputs[block_outputs_i] += segment[0];
        }
        __syncthreads();
    }
}
