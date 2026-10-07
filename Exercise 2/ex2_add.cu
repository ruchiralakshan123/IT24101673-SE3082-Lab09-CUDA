#include <cstdio>

// Kernel: this code runs on the device (the NVIDIA GPU)
__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

// host = CPU, device = GPU
int main(void) {
    int a, b, c;
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = 2;
    b = 7;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    add<<<1, 1>>>(d_a, d_b, d_c);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result is %d\n", c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
