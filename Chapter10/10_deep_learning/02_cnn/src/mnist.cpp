#include "mnist.h"
#include <cstring>

using namespace cudl;

MNIST::~MNIST()
{
    delete data_;
    delete target_;
}

void MNIST::create_shared_space()
{
    // create blobs with batch size and sample size
    data_ = new Blob<float>(batch_size_, channels_, height_, width_);
    data_->tensor();
    target_ = new Blob<float>(batch_size_, num_classes_);

}

void MNIST::load_data(std::string &image_file_path)
{
    uint8_t ptr[4];
    std::string file_path_ = dataset_dir_ + "/" + image_file_path;

    std::cout << "loading " << file_path_ << std::endl;
    std::ifstream file(file_path_.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "Download dataset first!!" << std::endl;
        std::cout << "You can get the MNIST dataset from 'http://yann.lecun.com/exdb/mnist/' or just use 'download_mnist.sh' file." << std::endl;
        exit(-1);
    }

    file.read((char*)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x803);

    int num_data;
    file.read((char*)ptr, 4);
    num_data = to_int(ptr);
    file.read((char*)ptr, 4);
    height_ = to_int(ptr);
    file.read((char*)ptr, 4);
    width_ = to_int(ptr);

    uint8_t* q = new uint8_t[channels_ * height_ * width_];
    for (int i = 0; i < num_data; i++)
    {
        std::vector<float> image = std::vector<float>(channels_ * height_ * width_);
        float *image_ptr = image.data();

        file.read((char*)q, channels_ * height_ * width_);
        for (int j = 0; j < channels_ * height_ * width_; j++)
        {
            image_ptr[j] = (float)q[j] / 255.f;
        }

        data_pool_.push_back(image);
    }

    delete [] q;

    num_steps_ = num_data / batch_size_;

    std::cout << "loaded " << data_pool_.size() << " items.." << std::endl;

    file.close();
}

void MNIST::load_target(std::string &label_file_path)
{
    uint8_t ptr[4];
    std::string file_path_ = dataset_dir_ + "/" + label_file_path;

    std::ifstream file(file_path_.c_str(), std::ios::in | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "Check dataset existance!!" << std::endl;
        exit(-1);
    }

    file.read((char*)ptr, 4);
    int magic_number = to_int(ptr);
    assert((magic_number & 0xFFF) == 0x801);

    file.read((char*)ptr, 4);
    int num_target = to_int(ptr);

    // prepare input buffer for label
    // read all labels and converts to one-hot encoding
    for (int i = 0; i < num_target; i++)
    {
        std::array<float, MNIST_CLASS> target_batch;
        std::fill(target_batch.begin(), target_batch.end(), 0.f);

        file.read((char*)ptr, 1);
        target_batch[static_cast<int>(ptr[0])] = 1.f;
        target_pool_.push_back(target_batch);
    }

    file.close();
}

void MNIST::shuffle_dataset()
{
    std::random_device rd;
    std::mt19937 g_data(rd());
    auto g_target = g_data;

    std::shuffle(std::begin(data_pool_), std::end(data_pool_), g_data);
    std::shuffle(std::begin(target_pool_), std::end(target_pool_), g_target);
}

int MNIST::to_int(uint8_t *ptr)
{
    return ((ptr[0] & 0xFF) << 24 | (ptr[1] & 0xFF) << 16 |
            (ptr[2] & 0xFF) << 8 | (ptr[3] & 0xFF) << 0);
}

void MNIST::train(int batch_size, bool shuffle)
{
    if (batch_size < 1)
    {
        std::cout << "batch size should be greater than 1." << std::endl;
        return;
    }

    batch_size_ = batch_size;
    shuffle_ = shuffle;

    load_data(train_dataset_file_);
    load_target(train_label_file_);

    if (shuffle_)
        shuffle_dataset();
    create_shared_space();

    step_ = 0;
}

void MNIST::test(int batch_size) {
    if (batch_size < 1)
    {
        std::cout << "batch size should be greater than or equal to 1." << std::endl;
        return;
    }

    batch_size_ = batch_size;
    
    load_data(test_dataset_file_);
    load_target(test_label_file_);

    create_shared_space();

    step_ = 0;
}

void MNIST::get_batch()
{
    if (step_ < 0)
    {
        std::cout << "You must initialize dataset first.." << std::endl;
        exit (-1);
    }

    // index cliping
    int data_idx = (step_ * batch_size_) % num_steps_;

    // prepare data blob
    int data_size = channels_ * width_ * height_;

    // copy data
    for (int i = 0; i < batch_size_; i++)
        std::copy(data_pool_[data_idx+i].data(),
            &data_pool_[data_idx+i].data()[data_size],
            &data_->ptr()[data_size*i]);

    // copy target with one-hot encoded
    for (int i = 0; i < batch_size_; i++)
        std::copy(target_pool_[data_idx+i].data(),
            &target_pool_[data_idx+i].data()[MNIST_CLASS],
            &target_->ptr()[MNIST_CLASS*i]);
}

int MNIST::next()
{
    step_++;

    get_batch();

    return step_;
}