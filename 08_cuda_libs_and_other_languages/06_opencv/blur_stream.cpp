#include <iostream>
#include <string>
#include <cuda_profiler_api.h>
#include "opencv2/opencv.hpp"

using namespace cv;

void BlurHost(std::string filename, float scale)
{
    Mat src = imread(filename, 1);
    Mat dst;

    TickMeter tm;

    tm.reset();
    tm.start();
    bilateralFilter(src, dst, 21, 150, 150);
    tm.stop();

    std::cout << "CPU Time: " << tm.getTimeMilli() << " ms." << std::endl;

    imwrite("result_host.jpg", dst);
}

void BlurCuda_Stream(std::string filename, float scale)
{
    // set the OpenCV to allocate the host memory as pinned memory
    Mat::setDefaultAllocator(cuda::HostMem::getAllocator(cuda::HostMem::PAGE_LOCKED));

    // initialize streams
    const int num_stream = 4;
    cuda::Stream stream[num_stream];

    // loading source image and GPU memories
    Mat src = imread(filename, 1);
    Mat dst;
    cuda::GpuMat src_cuda[num_stream], dst_cuda[num_stream];
    for (int i = 0; i < num_stream; i++) {
        src_cuda[i] = cuda::GpuMat(src);
    }

    // openCV timer
    TickMeter tm;

    // warm-up
    cuda::bilateralFilter(src_cuda[0], dst_cuda[0], 21, 50, 50);

    // cudaProfilerStart();
    tm.reset();
    tm.start();
    for (int i = 0; i < num_stream; i++) {
        src_cuda[i].upload(src, stream[i]);
        cuda::bilateralFilter(src_cuda[i], dst_cuda[i], 21, 150.f, 150.f, BORDER_DEFAULT, stream[i]);
        dst_cuda[i].download(dst, stream[i]);
    }
    for (int i = 0; i < num_stream; i++)
        stream[i].waitForCompletion();
    tm.stop();
    // cudaProfilerStop();

    std::cout << "GPU Time: " << tm.getTimeMilli() / num_stream << " ms." << std::endl;
    imwrite("result_cuda.jpg", dst);
}

int main(int argc, char *argv[])
{
    std::string filename("flower.jpg");
    float scale = 1.5f;

    BlurHost(filename, scale);
    BlurCuda_Stream(filename, scale);

    return 0;
}