#ifndef _UTIL_H_
#define _UTIL_H_

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

    void init(int size, bool do_fill = false)
    {
        if (h_ptr_ != NULL)
            return;

        size_ = size;
        h_ptr_ = (T *)new T[size_];
        d_ptr_ = NULL;

        if (do_fill == true)
        {
            for (int i = 0; i < size_; i++)
                h_ptr_[i] = (rand() & 0xFF) / (float)RAND_MAX;
        }
    }

    int cuda(bool do_copy = false)
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
};

#endif // _UTIL_H_