#ifndef _SCAN_H_
#define _SCAN_H_

#define BLOCK_DIM 512

#define DEBUG_INDEX         0
#define DEBUG_OUTPUT_NUM    16

void scan_v1(float *d_output, float *d_input, int length);
void scan_v2(float *d_output, float *d_input, int length);

#endif // _SCAN_H_