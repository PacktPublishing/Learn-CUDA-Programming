
#define BLOCK_SIZE 128
#define SOFTENING 1e-9f


typedef struct {
        float4 *pos, *vel;
} NBodySystem;

void generateRandomizeBodies(float *data, int n);
__global__ void calculateBodyForce(float4 *p, float4 *v, float dt, int n);
