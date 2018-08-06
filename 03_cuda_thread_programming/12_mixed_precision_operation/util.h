#ifndef _UTIL_H_
#define _UTIL_H_

template <typename T>
class CBuffer
{
    int size_;
    T *h_ptr_;
    T *d_ptr_;

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

    void init(int size, is_fill = false)
    {
        if (ptr_ != NULL)
            return;

        size_ = size;
        ptr_ = (T *)new T[size_];

        if (is_fill == true) {
            for (int i = 0; i < size_; i++)
                ptr_[i] = (rand() & 0xFF) / (float)RAND_MAX;
        }
    }

    void init_gpu(is_copy = false)
    {
        
    }
};

#endif // _UTIL_H_