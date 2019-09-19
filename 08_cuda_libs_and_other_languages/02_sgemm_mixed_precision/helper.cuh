#ifndef _UTIL_H_
#define _UTIL_H_

#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>
#include <curand.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

namespace fp16
{
    __global__ void float2half(half *out, float *in)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        out[idx] = __float2half(in[idx]);
    }

    __global__ void half2float(float *out, half *in)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        out[idx] = __half2float(in[idx]);
    }
}

namespace helper
{

template <typename T>
class CBuffer
{
    int size_;

  public:
    CBuffer()
    {
        h_ptr_ = NULL;
        d_ptr_ = NULL;
        size_ = 0;
    }
    ~CBuffer()
    {
        if (h_ptr_ != NULL)
            delete[] h_ptr_;
        if (d_ptr_ != NULL)
            cudaFree(d_ptr_);
    }

    T *h_ptr_;
    T *d_ptr_;

    void init(int size, bool do_fill = true)
    {
        if (h_ptr_ != NULL)
            return;

        size_ = size * sizeof(T);
        h_ptr_ = (T *)new T[size_];
        d_ptr_ = NULL;

        if (do_fill == true)
        {
            for (int i = 0; i < size_; i++)
                if (sizeof(T) >= 2)    
                    h_ptr_[i] = .1f * rand() / (float)RAND_MAX;
                else
                    h_ptr_[i] = rand() % INT8_MAX;

        }
    }

    int cuda(bool do_copy = true)
    {
        if (d_ptr_ != NULL)
            return 1;

        cudaMalloc((void**)&d_ptr_, size_ * sizeof(T));

        if (do_copy == true)
            copyToDevice();

        return 0;
    }

    void copyToDevice()
    {
        cudaMemcpy(d_ptr_, h_ptr_, size_, cudaMemcpyHostToDevice);
    }

    void copyToHost()
    {
        cudaMemcpy(h_ptr_, d_ptr_, size_, cudaMemcpyDeviceToHost);
    }

    int diff_count()
    {
        int diff_count = 0;

        if (h_ptr_ == NULL || d_ptr_ == NULL)
            return -1;

        T *temp_ptr = (T *)new T[size_];
        cudaMemcpy(temp_ptr, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        
        #pragma omp parallel
        {
        #pragma omp for
            for (int i = 0; i < size_; i++) {
                if (fabs(temp_ptr[i] - h_ptr_[i]) > 0.0001) 
                    diff_count++;
            }
        }
        delete [] temp_ptr;

        return diff_count;
    }
};

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n) {
    std::cout << "[" << __FUNCTION__ << "]:: Not supported type request" << std::endl;
}
void printMatrix(const float *matrix, const int ldm, const int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < ldm; i++) {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        }
        std::cout << std::endl;
    }
}
void printMatrix(const int *matrix, const int ldm, const int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < ldm; i++) {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        }
        std::cout << std::endl;
    }
}
void printMatrix(const half *matrix, const int ldm, const int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < ldm; i++) {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << __half2float(matrix[IDX2C(i, j, ldm)]);
        }
        std::cout << std::endl;
    }
}
void printMatrix(const int8_t *matrix, const int ldm, const int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < ldm; i++) {
            std::cout << std::fixed << std::setw(4) << static_cast<int16_t>(matrix[IDX2C(i, j, ldm)]);
        }
        std::cout << std::endl;
    }
}

    
}

#endif // _UTIL_H_