#ifndef _BLOB_H_
#define _BLOB_H_

#include <array>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <cudnn.h>

namespace cudl
{
typedef enum {
    host,
    cuda
} DeviceType;

template <typename ftype>
class Blob
{
    public:
    Blob(int n = 1, int c = 1, int h = 1, int w = 1): n_(n), c_(c), h_(h), w_(w) 
    {
        h_ptr_ = new float[n_ * c_ * h_ * w_];
    }
    Blob(std::array<int, 4> size): n_(size[0]), c_(size[1]), h_(size[2]), w_(size[3]) 
    {
        h_ptr_ = new float[n_ * c_ * h_ * w_];
    }
    
    ~Blob() 
    { 
        if (h_ptr_ != nullptr) 
            delete [] h_ptr_; 
		if (d_ptr_ != nullptr)
			cudaFree(d_ptr_);
        if (is_tensor_)
            cudnnDestroyTensorDescriptor(tensor_desc_);
    }

    // reset the current blob with the new size information
    void reset(int n = 1, int c = 1, int h = 1, int w = 1)
    {
        // update size information
        n_ = n;
        c_ = c;
        h_ = h;
        w_ = w;

        // terminate current buffers
        if (h_ptr_ != nullptr)
        {
            delete [] h_ptr_;
            h_ptr_ = nullptr;
        }
        if (d_ptr_ != nullptr)
        {
            cudaFree(d_ptr_);
            d_ptr_ = nullptr;
        }

        // create new buffer
        h_ptr_ = new float[n_ * c_ * h_ * w_];
        cuda();

        // reset tensor descriptor if it was tensor
        if (is_tensor_)
        {
            cudnnDestroyTensorDescriptor(tensor_desc_);
            is_tensor_ = false;
        }
    }

    void reset(std::array<int, 4> size)
    {
        reset(size[0], size[1], size[2], size[3]);
    }

    // returns array of tensor shape
    std::array<int, 4> shape() { return std::array<int, 4>({n_, c_, h_, w_}); }
    
    // returns number of elements for 1 batch
    int size() { return c_ * h_ * w_; }

    // returns number of total elements in blob including batch
    int len() { return n_ * c_ * h_ * w_; }
    
    // returns size of allocated memory
    int buf_size() { return sizeof(ftype) * len(); }

    int n() const { return n_; }
    int c() const { return c_; }
    int h() const { return h_; }
    int w() const { return w_; }

    /* Tensor Control */
    bool is_tensor_ = false;
    cudnnTensorDescriptor_t tensor_desc_;
    cudnnTensorDescriptor_t tensor()
    {
        if (is_tensor_)
            return tensor_desc_;
        
        cudnnCreateTensorDescriptor(&tensor_desc_);
        cudnnSetTensor4dDescriptor(tensor_desc_, 
                                    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                    n_, c_, h_, w_);
		is_tensor_ = true;

        return tensor_desc_;
    }

    /* Memory Control */
    // get specified memory pointer
    ftype *ptr() { return h_ptr_; }

    // get cuda memory
    ftype *cuda() 
    { 
        if (d_ptr_ == nullptr)  
            cudaMalloc((void**)&d_ptr_, sizeof(ftype) * len());

        return d_ptr_;
    }

    // transfer data between memory
    ftype *to(DeviceType target) { 
        ftype *ptr = nullptr;
        if (target == host)
        {
            cudaMemcpy(h_ptr_, cuda(), sizeof(ftype) * len(), cudaMemcpyDeviceToHost);
            ptr = h_ptr_;
        }
        else // DeviceType::cuda
        {
            cudaMemcpy(cuda(), h_ptr_, sizeof(ftype) * len(), cudaMemcpyHostToDevice);
            ptr = d_ptr_;
        }
        return ptr;
    }

    void print(std::string name, bool view_param = false, int num_batch = 1, int width = 16)
    {
        to(host);
        std::cout << "**" << name << "\t: (" << size() << ")\t";
        std::cout << ".n: " << n_ << ", .c: " << c_ << ", .h: " << h_ << ", .w: " << w_;
        std::cout << std::hex << "\t(h:" << h_ptr_ << ", d:" << d_ptr_ << ")" << std::dec << std::endl;

        if (view_param)
        {
            std::cout << std::fixed;
            std::cout.precision(6);
            
            int max_print_line = 4;
            if (width == 28) {
                std::cout.precision(3);
                max_print_line = 28;
            }
            int offset = 0;

            for (int n = 0; n < num_batch; n++) {
                if (num_batch > 1)
                    std::cout << "<--- batch[" << n << "] --->" << std::endl;
                int count = 0;
                int print_line_count = 0;
                while (count < size() && print_line_count < max_print_line)
                {
                    std::cout << "\t";
                    for (int s = 0; s < width && count < size(); s++)
                    {
                        std::cout << h_ptr_[size()*n + count + offset] << "\t";
                        count++;
                    }
                    std::cout << std::endl;
                    print_line_count++;
                }
            }
            std::cout.unsetf(std::ios::fixed);
        }
    }

    /* pretrained parameter load and save */
    int file_read(std::string filename)
    {
        std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            std::cout << "fail to access " << filename << std::endl;
            return -1;
        }

        file.read((char*)h_ptr_, sizeof(float) * this->len());
        this->to(DeviceType::cuda);
        file.close();

        return 0;
    }

    int file_write(std::string filename)
    {
        std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary );
        if (!file.is_open())
        {
            std::cout << "fail to write " << filename << std::endl;
            return -1;
        }
        file.write((char*)this->to(host), sizeof(float) * this->len());
        file.close();

        return 0;
    }

    private:

    ftype *h_ptr_ = nullptr;
    ftype *d_ptr_ = nullptr;

    int n_ = 1;
    int c_ = 1;
    int h_ = 1;
    int w_ = 1;
};

} // namespace cudl

#endif // _BLOB_H_