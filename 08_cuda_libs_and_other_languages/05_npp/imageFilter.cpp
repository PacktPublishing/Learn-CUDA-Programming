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
    /* image information */
    FIBITMAP* dib;  // FreeImage bitmap
    int nHeight;    // image height size
    int nWidth;     // image width size
    int nPitch;     // image pitch size
    int nBPP;       // Bit Per Pixel (i.e. 24 for BGR color)
    int nChannel;   // number of channels 
    BYTE* pData;    // bytes from freeimage library
   
    /* CUDA */
    int nPitchCUDA;      // image pitch size on CUDA device
    Npp8u *pDataCUDA;    // CUDA global memory for nppi processing
};

void LoadImage(const char *szInputFile, ImageInfo_t &srcImage) {
    FIBITMAP *pSrcImageBitmap = FreeImage_Load(FIF_JPEG, szInputFile, JPEG_DEFAULT);
    if (!pSrcImageBitmap) {
        std::cout << "Couldn't load " << szInputFile << std::endl;
        FreeImage_DeInitialise();
        exit(1);
    }

    srcImage.dib = pSrcImageBitmap;
    srcImage.nWidth = FreeImage_GetWidth(pSrcImageBitmap);
    srcImage.nHeight = FreeImage_GetHeight(pSrcImageBitmap);
    srcImage.nPitch = FreeImage_GetPitch(pSrcImageBitmap);
    srcImage.nBPP = FreeImage_GetBPP(pSrcImageBitmap);
    srcImage.pData = FreeImage_GetBits(pSrcImageBitmap);
    assert(srcImage.nBPP == (unsigned int)24); // BGR color image
    srcImage.nChannel = 3;
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

void ResizeCPU(ImageInfo_t &dstImage, ImageInfo_t &srcImage)
{
    FreeImage_Rescale(srcImage.dib, dstImage.nWidth, dstImage.nHeight, FILTER_LANCZOS3);
}

void ResizeGPU(ImageInfo_t &dstImage, ImageInfo_t &srcImage)
{
    // obtain image size and ROI
    NppiSize srcSize = GetImageSize(srcImage);
    NppiSize dstSize = GetImageSize(dstImage);
    NppiRect srcROI = GetROI(srcImage);
    NppiRect dstROI = GetROI(dstImage);

    nppiResize_8u_C3R(srcImage.pDataCUDA, srcImage.nPitchCUDA, srcSize, srcROI, 
                    dstImage.pDataCUDA, dstImage.nPitchCUDA, dstSize, dstROI, 
                    NPPI_INTER_LANCZOS);
}

int main()
{
    int opt = 0;
    float scaleRatio = 0.5f;

    const char* szInputFile = "flower.jpg";
    const char* szOutputFile = "output.jpg";

    bool profile = true;

    ImageInfo_t srcImage, dstImage;
    float gpu_time_ms = 0.f;

    std::cout << "Rescale " << szInputFile << " in " << scaleRatio << " ratio." << std::endl;

    FreeImage_Initialise();

    // Load input image
    LoadImage(szInputFile, srcImage);

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

    /* do resize operaiton using GPU */
    // warm-up
    ResizeGPU(dstImage, srcImage);

    cudaEvent_t start, stop;
    if (profile)
    {
        // create CUDA event to measure GPU execution time
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // do resize operation
        cudaEventRecord(start);
    }
    ResizeGPU(dstImage, srcImage);
    if (profile)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time_ms, start, stop);
        std::cout << "GPU: " << gpu_time_ms << " ms" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    /* do resize operation using FreeImage library (host) */
	StopWatchInterface *timer;
    if (profile)
    {
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
    }
    ResizeCPU(dstImage, srcImage);
    
    if (profile)
    {
        sdkStopTimer(&timer);
        std::cout << "CPU Rescale elapsed time: " << sdkGetTimerValue(&timer) << std::endl;
        sdkDeleteTimer(&timer);
    }

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

