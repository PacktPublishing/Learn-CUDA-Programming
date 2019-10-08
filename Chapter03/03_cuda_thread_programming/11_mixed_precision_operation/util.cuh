#ifndef _UTIL_H_
#define _UTIL_H_

#include <cstdio>
#include <cfloat>
#include <iostream>

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

        size_ = size;
        h_ptr_ = (T *)new T[size_];
        d_ptr_ = NULL;

        if (do_fill == true)
        {
            for (int i = 0; i < size_; i++)
                if (sizeof(T) >= 2)
                    h_ptr_[i] = (rand() & 0xFF) / (float)RAND_MAX;
                else
                    h_ptr_[i] = i & 0xF;
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
        cudaMemcpy(d_ptr_, h_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyToHost()
    {
        cudaMemcpy(h_ptr_, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
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

#endif // _UTIL_H_