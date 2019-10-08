#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/cudacodec.hpp"

void BlurHost(std::string filename)
{
    cv::Mat src = cv::imread(filename, 1);
    cv::Mat dst;

    cv::TickMeter tm;

    tm.reset();
    tm.start();
    cv::bilateralFilter(src, dst, 10, 50, 50);
    tm.stop();

    std::cout << "CPU Time: " << tm.getTimeMilli() << " ms." << std::endl;

    cv::imwrite("result_host.jpg", dst);
}

void BlurCuda(std::string filename)
{
    // load image to the host memory
    cv::Mat src = cv::imread(filename, 1);
    cv::Mat dst;
    cv::cuda::GpuMat src_cuda = cv::cuda::GpuMat(src.rows, src.cols, CV_8UC1);
    cv::cuda::GpuMat dst_cuda;

    cv::TickMeter tm;

    // warm-up
    cv::cuda::bilateralFilter(src_cuda, dst_cuda, 10, 50, 50);

    tm.reset();
    tm.start();
    src_cuda.upload(src);
    cv::cuda::bilateralFilter(src_cuda, dst_cuda, 10, 50, 50);
    dst_cuda.download(dst);
    tm.stop();

    std::cout << "GPU Time: " << tm.getTimeMilli() << " ms." << std::endl;
    cv::imwrite("result_cuda.jpg", dst);
}

void BlurCuda_cuvid(std::string filename)
{
    // load image to the host memory
    cv::Mat src = cv::imread(filename, 1);
    cv::Mat dst;
    cv::cuda::GpuMat src_cuda = cv::cuda::GpuMat(src.rows, src.cols, CV_8UC1);
    cv::cuda::GpuMat dst_cuda = cv::cuda::GpuMat(src.rows, src.cols, CV_8UC1);

    printf("%s[%d]\n", __FILE__, __LINE__);
    cv::Ptr<cv::cudacodec::VideoReader> videoReader = cv::cudacodec::createVideoReader(filename);
    printf("format: %d\n", videoReader->format());
    cv::cuda::GpuMat frame;

    videoReader->nextFrame(frame);
    frame.empty();

    cv::TickMeter tm;

    // warps-up
    cv::cuda::bilateralFilter(src_cuda, dst_cuda, 10, 50, 50);

    tm.reset();
    tm.start();
    src_cuda.upload(src);
    cv::cuda::bilateralFilter(src_cuda, dst_cuda, 10, 50, 50);
    dst_cuda.download(dst);
    tm.stop();

    std::cout << "GPU Time: " << tm.getTimeMilli() << " ms." << std::endl;
    cv::imwrite("result_cuda.jpg", dst);
}

int main(int argc, char *argv[])
{
    std::string filename("pexel_video.mp4");

    //BlurHost(filename);
    //BlurCuda(filename);
    BlurCuda_cuvid(filename);

    return 0;
}