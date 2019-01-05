#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <npp.h>
#include <FreeImage.h>
#include <helper_timer.h>

struct ImageInfo_t
{
    // Image information
    int nHeight;
    int nWidth;
    int nPitch;
    int nBPP;
    int nChannel;
    BYTE* pData;

    // CUDA device 
    Npp8u *pDataCUDA;
    int nPitchCUDA;
};

FIBITMAP* LoadImage(const char *szInputFile, ImageInfo_t &srcImage) {
    FIBITMAP *pSrcImageBitmap = FreeImage_Load(FIF_JPEG, szInputFile, JPEG_DEFAULT);
    if (!pSrcImageBitmap) {
        std::cout << "Couldn't load " << szInputFile << std::endl;
        FreeImage_DeInitialise();
        return nullptr;
    }

    srcImage.nWidth = FreeImage_GetWidth(pSrcImageBitmap);
    srcImage.nHeight = FreeImage_GetHeight(pSrcImageBitmap);
    srcImage.nPitch = FreeImage_GetPitch(pSrcImageBitmap);
    srcImage.nBPP = FreeImage_GetBPP(pSrcImageBitmap);
    srcImage.pData = FreeImage_GetBits(pSrcImageBitmap);
    assert(srcImage.nBPP == (unsigned int)24); // BGR color image
    srcImage.nChannel = 3;

    return pSrcImageBitmap;
}

NppiSize GetImageSize(ImageInfo_t imageInfo)
{
    NppiSize imageSize;

    imageSize.width = imageInfo.nWidth;
    imageSize.height = imageInfo.nHeight;

    return imageSize;
}

NppiRect GetROI(ImageInfo_t imageInfo)
{
    NppiRect imageROI;

    imageROI.x = 0;
    imageROI.y = 0;
    imageROI.width = imageInfo.nWidth;
    imageROI.height = imageInfo.nHeight;

    return imageROI;
}

void RunCpuResize(const char* szInputFile, ImageInfo_t &dstImage)
{
    StopWatchInterface *timer;
    ImageInfo_t tmpImage;
    FIBITMAP* dib = LoadImage(szInputFile, tmpImage);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    FreeImage_Rescale(dib, dstImage.nWidth, dstImage.nHeight, FILTER_LANCZOS3);

    sdkStopTimer(&timer);
    std::cout << "CPU Rescale elapsed time: " << sdkGetTimerValue(&timer) << std::endl;

    sdkDeleteTimer(&timer);    
}

int RunNppResize(ImageInfo_t &dstImage, ImageInfo_t &srcImage, 
                 NppiSize &dstSize, NppiRect &dstROI, 
                 NppiSize &srcSize, NppiRect &srcROI,
                 float scale)
{
    // update output image size
    dstSize.width = dstROI.width = dstImage.nWidth;
    dstSize.height = dstROI.height = dstImage.nHeight;

    nppiResize_8u_C3R(srcImage.pDataCUDA, srcImage.nPitchCUDA, srcSize, srcROI, 
                                        dstImage.pDataCUDA, dstImage.nPitchCUDA, dstSize, dstROI, 
                                        NPPI_INTER_LANCZOS);
    return 0;
}

int main()
{
    int opt = 0;
    float scaleRatio = 0.5f;

    const char* szInputFile = "flower.jpg";
    const char* szOutputFile = "output.jpg";
    ImageInfo_t srcImage;
    ImageInfo_t dstImage;

    std::cout << "Rescale " << szInputFile << " in " << scaleRatio << " ratio." << std::endl;

    FreeImage_Initialise();

    // Load input image
    FIBITMAP* pSrcImageBitmap = LoadImage(szInputFile, srcImage);
    if (pSrcImageBitmap == nullptr)
        return EXIT_FAILURE;

    // copy loaded image to the device memory
    srcImage.pDataCUDA = nppiMalloc_8u_C3(srcImage.nWidth, srcImage.nHeight, &srcImage.nPitchCUDA);
    cudaMemcpy2D(srcImage.pDataCUDA, srcImage.nPitchCUDA, 
                                 srcImage.pData, srcImage.nPitch, 
                                 srcImage.nWidth * srcImage.nChannel * sizeof(Npp8u), srcImage.nHeight,
                                 cudaMemcpyHostToDevice);

    // setting output image information
    std::memcpy(&dstImage, &srcImage, sizeof(ImageInfo_t));
    dstImage.nWidth *= scaleRatio;
    dstImage.nHeight *= scaleRatio;
    dstImage.pDataCUDA = nppiMalloc_8u_C3(dstImage.nWidth, dstImage.nHeight, &dstImage.nPitchCUDA);

    // obtain image size and ROI
    NppiSize srcImageSize = GetImageSize(srcImage);
    NppiSize dstImageSize = GetImageSize(dstImage);
    NppiRect srcROI = GetROI(srcImage);
    NppiRect dstROI = GetROI(dstImage);

    // warms-up
    RunNppResize(dstImage, srcImage, dstImageSize, dstROI, srcImageSize, srcROI, scaleRatio);

    // create CUDA event to measure GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // do resize operation
    RunNppResize(dstImage, srcImage, dstImageSize, dstROI, srcImageSize, srcROI, scaleRatio);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // do resize operation using FreeImage library (host)
    RunCpuResize(szInputFile, dstImage);

    // print CUDA operation time
    float elapsedTime = 0.f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "GPU: " << elapsedTime << " ms" << std::endl;    

    // Save resized image as file from the device
    FIBITMAP *pDstImageBitmap = FreeImage_Allocate(dstImage.nWidth, dstImage.nHeight, dstImage.nBPP);

    dstImage.nPitch = FreeImage_GetPitch(pDstImageBitmap);
    dstImage.pData = FreeImage_GetBits(pDstImageBitmap);

    cudaMemcpy2D(dstImage.pData, dstImage.nPitch, 
                                 dstImage.pDataCUDA, dstImage.nPitchCUDA, 
                                 dstImage.nWidth * dstImage.nChannel * sizeof(Npp8u), dstImage.nHeight,
                                 cudaMemcpyDeviceToHost);

    FreeImage_Save(FIF_JPEG, pDstImageBitmap, szOutputFile, JPEG_DEFAULT);

    std::cout << "Done (generated " << szOutputFile << ")" << std::endl;

    nppiFree(srcImage.pDataCUDA);
    nppiFree(dstImage.pDataCUDA);

    FreeImage_DeInitialise();
    
    return 0;
}

