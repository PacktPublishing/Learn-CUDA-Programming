#pragma once

template <class T>
class CudaContainer
{
public:
	int size;
	T* data;
	CudaContainer(int size);
	~CudaContainer();
};