#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
 
using namespace cv;
 
int main( int argc, char* argv[] )
{
    const int64 start = getTickCount();
 
    cv::Mat src = cv::imread( "flower.jpg", 0 );
 
    if( !src.data ) exit( 1 );
 
    cv::cuda::GpuMat d_src( src );
    cv::cuda::GpuMat d_dst;
 
    cv::cuda::bilateralFilter( d_src, d_dst, -1, 50, 7 );
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector( 35.0, 200.0 );
    canny->detect( d_src, d_dst );
 
    cv::Mat dst( d_dst );
 
    cv::imwrite( "cuda_canny.png", dst );
 
    const double timeSec = (getTickCount() - start) / getTickFrequency();
    std::cout << "Time : " << timeSec << " sec" << std::endl;
    
    return 0;
}
