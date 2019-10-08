# Learn CUDA Programming 

<a href="https://www.packtpub.com/application-development/cuda-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788996242"><img src="https://www.packtpub.com/media/catalog/product/cache/e4d64343b1bc593f1c5348fe05efa4a6/9/7/9781788996242-original.jpeg" alt="Learn CUDA Programming " height="256px" align="right"></a>

This is the code repository for [Learn CUDA Programming ](https://www.packtpub.com/application-development/cuda-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788996242), published by Packt.

**A beginner's guide to GPU programming and parallel computing with CUDA 10.x and C/C++**

## What is this book about?
Compute Unified Device Architecture (CUDA) is NVIDIA's GPU computing platform and application programming interface. It's designed to work with programming languages such as C, C++, and Python. With CUDA, you can leverage a GPU's parallel computing power for a range of high-performance computing applications in the fields of science, healthcare, and deep learning.


This book covers the following exciting features:
* Understand general GPU operations and programming patterns in CUDA 
* Uncover the difference between GPU programming and CPU programming 
* Analyze GPU application performance and implement optimization strategies 
* Explore GPU programming, profiling, and debugging tools 
* Grasp parallel programming algorithms and how to implement them 
Scale GPU-accelerated applications with multi-GPU and multi-nodes 
Delve into GPU programming platforms with accelerated libraries, Python, and OpenACC 
Gain insights into deep learning accelerators in CNNs and RNNs using GPUs

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1788996240) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] \
        From device\n", threadIdx.x,blockIdx.x);
}
```

**Following is what you need for this book:**
This beginner-level book is for programmers who want to delve into parallel computing, become part of the high-performance computing community and build modern applications. Basic C and C++ programming experience is assumed. For deep learning enthusiasts, this book covers Python InterOps, DL libraries, and practical examples on performance estimation.

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| All | CUDA Toolkit 9.x/10.x | Linux |
| 8 | Matlab (later than 2010a) | Linux |
| 9 | PGI Compilers 18.x/19.x | Linux |
| 10 | NGC | Linux |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781788996242_ColorImages.pdf).

### Related product
Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA  [[Packt]](https://www.packtpub.com/application-development/hands-gpu-accelerated-computer-vision-opencv-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781789348293) [[Amazon]](https://www.amazon.com/dp/1789348293)

## Get to Know the Authors
**Jaegeun Han**
is currently working as a solutions architect at NVIDIA, Korea. He has around 9 years' experience and he supports consumer internet companies in deep learning. Before NVIDIA, he worked in system software and parallel computing developments, and application development in medical and surgical robotics fields. He obtained a master's degree in CSE from Seoul National University.

**Bharatkumar Sharma**
obtained a master's degree in information technology from the Indian Institute of Information Technology, Bangalore. He has around 10 years of development and research experience in the domains of software architecture and distributed and parallel computing. He is currently working with NVIDIA as a senior solutions architect, South Asia.

### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
