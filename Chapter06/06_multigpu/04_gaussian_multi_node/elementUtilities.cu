#include "gaussian_multi_gpu_rdma.h"

void resetElement(elemtype* e)
{
    #if (PACK_SIZE == 32)
    {
        *e = 0;
    }
    #elif (PACK_SIZE == 128)
    {
        e->x = 0;
        e->y = 0;
        e->z = 0;
        e->w = 0;
    }
    #else
    {
        #error "Unsupported PACK_SIZE detected"
    }
    #endif
}

void setElementBit(elemtype* e, int bitIndex)
{
    #if (PACK_SIZE == 32)
    {
        (*e) |= (1 << bitIndex);
    }
    #elif (PACK_SIZE == 128)
    {
        if(bitIndex < 32)
        {
            e->x |= (1 << bitIndex);
        }
        else if(bitIndex < 64)
        {
            bitIndex -= 32;
            e->y |= (1 << bitIndex);
        }
        else if(bitIndex < 96)
        {
            bitIndex -= 64;
            e->z |= (1 << bitIndex);
        }
        else
        {
            bitIndex -= 96;
            e->w |= (1 << bitIndex);
        }
    }
    #else
    {
        #error "Unsupported PACK_SIZE detected"
    }
    #endif
}

unsigned char getElementBit(elemtype* e, int bitIndex)
{
    unsigned char bitValue;

    #if (PACK_SIZE == 32)
    {
        bitValue = ((*e) >> bitIndex ) & 1;
    }
    #elif (PACK_SIZE == 128)
    {
        if(bitIndex < 32)
        {
            bitValue = (e->x >> bitIndex) & 1;
        }
        else if(bitIndex < 64)
        {
            bitIndex -= 32;
            bitValue = (e->y >> bitIndex) & 1;
        }
        else if(bitIndex < 96)
        {
            bitIndex -= 64;
            bitValue = (e->z >> bitIndex) & 1;
        }
        else
        {
            bitIndex -= 96;
            bitValue = (e->w >> bitIndex) & 1;
        }
    }
    #else
    {
        #error "Unsupported PACK_SIZE detected"
    }
    #endif
    return bitValue;
}


__device__ unsigned int bfe(unsigned int x, unsigned int bit, unsigned int numBits) 
{
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    return ret;
}


__device__ unsigned int bfew(elemtype* e, int bitIndex)
{
    #if (PACK_SIZE == 32)
    {
         return bfe(*e, bitIndex, 1);
    }
    #elif (PACK_SIZE == 128)
    {
        if(bitIndex < 32)
        {
            return bfe(e->x, bitIndex, 1);
        }
        else if(bitIndex < 64)
        {
            bitIndex -= 32;
            return bfe(e->y, bitIndex, 1);
        }
        else if(bitIndex < 96)
        {
            bitIndex -= 64;
            return bfe(e->z, bitIndex, 1);
        }
        else
        {
            bitIndex -= 96;
            return bfe(e->w, bitIndex, 1);
        }
    }
    #else
    {
        #error "Unsupported PACK_SIZE detected"
    }
    #endif
}
