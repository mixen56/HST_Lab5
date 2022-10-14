/*
 * File:   Main.cpp
 * Author: maslov_a
 *
 * Created on 17 сентября 2022 г., 13:56
 */

// nvcc -c ClientLab5.cu -o ClientLab5CUDA.o

#include <iostream>
#include <vector>
#include <chrono>

#include <boost/numeric/ublas/matrix.hpp>   // Matrix
#include <boost/array.hpp>                  // Matrix
#include <boost/program_options.hpp>        // program args

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>


using namespace std;
namespace buplas = boost::numeric::ublas;
namespace po     = boost::program_options;


__global__
void cuda_kernel(double *full_data, double *cuda_answer, int vector_size, int row_size) {
    //             matrix_id    row_size      thread        row_size    -> jump to every line only
    int line_pos = blockIdx.x * blockDim.x * blockDim.x + threadIdx.x * blockDim.x;

    // get max
    //double max = full_data[line_pos];
    double max = 0;
    for (int i = 0; i < row_size; i++) {
        if (full_data[line_pos + i] > max)
            max = full_data[line_pos + i];
    }

    cuda_answer[blockIdx.x * blockDim.x + threadIdx.x] = max*max;
}


// cuda cacls
vector<vector<double>> get_max_cuda(vector<buplas::matrix<double>> &matrix_vector,
                  chrono::high_resolution_clock::duration &duration) {
    int vector_size = matrix_vector.size();
    int row_size    = matrix_vector[0].size1();
    int matrix_size = row_size * row_size;

    // start measure
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

    // allocate low-level 3d array in CUDA (GTX 1080 8Gb memory - our limit: ~ 4Gb input data)
    double *full_data;
    double *cuda_answer;
    cudaMallocManaged(&full_data, vector_size * matrix_size * sizeof(double));
    cudaMallocManaged(&cuda_answer, vector_size * row_size * sizeof(double));

    // fill CUDA array
    int tmp_index;
    for (int i = 0; i < vector_size; i++) {
        for (int j = 0; j < row_size; j++) {
            for (int k = 0; k < row_size; k++) {
                tmp_index = i * matrix_size + j * row_size + k;
                full_data[tmp_index] = matrix_vector[i](j, k);
            }
        }
    }

    // start inner measure
    chrono::high_resolution_clock::time_point start_inner = chrono::high_resolution_clock::now();

    // start calcs
    dim3 grid(vector_size, 1, 1);
    dim3 block(row_size, 1, 1);
    cuda_kernel<<<grid, block>>>(full_data, cuda_answer, vector_size, row_size);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cuda_kernel!\n", cudaStatus);
    }

    // stop inner measure
    chrono::high_resolution_clock::time_point end_inner = std::chrono::high_resolution_clock::now();
    chrono::high_resolution_clock::duration duration_inner = std::chrono::duration_cast<std::chrono::nanoseconds>(end_inner - start_inner);
    long double calcs_time_s_inner = duration_inner.count()*1e-9;
    cout << "GPU Calcs time: " << calcs_time_s_inner << " s" << endl;

    // create and fill vector of matrices
    vector<vector<double>> answer(vector_size);
    for (int i = 0; i < vector_size; i++) {
        vector<double> line(row_size);
        for (int h = 0; h < row_size; h++) {
            tmp_index = row_size * i + h;
            line[h] = cuda_answer[tmp_index];
        }
        answer[i] = line;
    }

    // print info about memory
    size_t free_gpu_memory, total_gpu_memory;
    cudaMemGetInfo ( &free_gpu_memory, &total_gpu_memory );
    double free_gpu_mem     = (double)free_gpu_memory / 1024 / 1024;
    double total_gpu_mem    = (double)total_gpu_memory / 1024 / 1024;
    double used_gpu_mem     = total_gpu_mem - free_gpu_mem;
    printf("%25s%25s%25s\n", "Total GPU Memory, Mb", "Free GPU Memory, Mb", "Used GPU Memory, Mb");
    printf("%25f%25f%25f\n", total_gpu_mem, free_gpu_mem, used_gpu_mem);

    // unallocate 3d array
    cudaFree(full_data);
    cudaFree(cuda_answer);

    // stop measure
    chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << endl;

    return answer;
}
