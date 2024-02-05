__global__ void test_int_add(int *a, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int worker = blockDim.x * gridDim.x;
    int size_per_worker = n / worker;
    int start = gid * size_per_worker;
    int end = start = size_per_worker;

    for (int i = start; i<end;i++) {
        a[i] += 1;
    }
}
