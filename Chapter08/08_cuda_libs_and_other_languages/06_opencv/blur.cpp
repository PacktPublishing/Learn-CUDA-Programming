#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace cv;

void BlurHost(std::string filename)
{
    Mat src = imread(filename, 1);
    Mat dst;

    TickMeter tm;
    
    tm.start();
    bilateralFilter(src, dst, 21, 150, 150);
    tm.stop();
    std::cout << "CPU Time: " << tm.getTimeMilli() << " ms." << std::endl;

    imwrite("result_host.jpg", dst);
}

void BlurCuda(std::string filename)
{
    TickMeter tm;

    Mat src = imread(filename, 1);
    Mat dst;
    cuda::GpuMat src_cuda(src);
    cuda::GpuMat dst_cuda;

    // warm-up
    cuda::bilateralFilter(src_cuda, dst_cuda, 21, 150.f, 150.f);

    tm.start();
    src_cuda.upload(src);
    cuda::bilateralFilter(src_cuda, dst_cuda, 21, 150.f, 150.f);
    dst_cuda.download(dst);
    tm.stop();
    std::cout << "GPU Time: " << tm.getTimeMilli() << " ms." << std::endl;

    imwrite("result_cuda.jpg", dst);
}

int main(int argc, char *argv[])
{
    cuda::printCudaDeviceInfo(0);
    cuda::printShortCudaDeviceInfo(0);
    std::cout << "Device: " << cuda::getCudaEnabledDeviceCount() << std::endl;

    std::string filename("flower.jpg");

    
    BlurHost(filename);
    BlurCuda(filename);

    return 0;
}