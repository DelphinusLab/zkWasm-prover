#include "cuda_runtime.h"

#include <assert.h>
#include <cub/device/device_radix_sort.cuh>

#include "zprize_ff_wrapper.cuh"
#include "zprize_ec_wrapper.cuh"

__global__ void count_nonzero_buckets(
    unsigned *non_zero_count,
    unsigned *bucket_indices,
    unsigned *point_indices,
    Bn254FrField *scalars,
    unsigned windows,
    unsigned window_bits,
    unsigned n)
{
    unsigned tasks = n;

    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned workers = gridDim.x * blockDim.x;
    unsigned tpw = (tasks + workers - 1) / workers;
    unsigned start = tpw * gid > tasks ? tasks : tpw * gid;
    unsigned end = start + tpw > tasks ? tasks : start + tpw;

    unsigned local_nonzero_index = 0;

    for (unsigned i = start; i < end; i++)
    {
        Bn254FrField s = scalars[i];
        s.unmont_assign();

        for (unsigned widx = 0; widx < windows; widx++)
        {
            unsigned start_bit = widx * window_bits;
            unsigned remain_bits = 254 - start_bit;
            unsigned bits = remain_bits < window_bits ? remain_bits : window_bits;

            unsigned idx = s.get_nbits(start_bit, bits);
            if (idx > 0)
            {
                local_nonzero_index++;
            }
        }
    }

    non_zero_count[gid] = local_nonzero_index;
}

__global__ void acc_nonzero_buckets(
    unsigned *non_zero_count,
    unsigned n)
{
    unsigned tasks = n;
    unsigned gid = threadIdx.x;
    unsigned workers = blockDim.x;
    unsigned tpw = (tasks + workers - 1) / workers;
    unsigned start = tpw * gid > tasks ? tasks : tpw * gid;
    unsigned end = start + tpw > tasks ? tasks : start + tpw;

    extern __shared__ unsigned acc[];

    for (unsigned i = start + 1; i < end; i++)
    {
        non_zero_count[i] += non_zero_count[i - 1];
    }

    if (start < n)
    {
        acc[gid] = non_zero_count[end - 1];
    }

    __syncthreads();

    if (gid == 0)
    {
        for (int i = 1; i < blockDim.x; i++)
        {
            acc[i] += acc[i - 1];
        }
    }

    __syncthreads();

    if (gid != 0)
    {
        for (unsigned i = start; i < end; i++)
        {
            non_zero_count[i] += acc[gid - 1];
        }
    }
}

__global__ void fill_buckte_indices(
    unsigned *non_zero_count,
    unsigned *total_bucket_indices,
    unsigned *total_point_indices,
    Bn254FrField *scalars,
    unsigned windows,
    unsigned window_bits,
    unsigned n)
{
    unsigned tasks = n;

    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned workers = gridDim.x * blockDim.x;
    unsigned tpw = (tasks + workers - 1) / workers;
    unsigned start = tpw * gid > tasks ? tasks : tpw * gid;
    unsigned end = start + tpw > tasks ? tasks : start + tpw;

    unsigned offset = non_zero_count ? (gid == 0 ? 0 : non_zero_count[gid - 1]) : start * windows;
    unsigned *point_indices = &total_point_indices[offset];
    unsigned *bucket_indices = &total_bucket_indices[offset];
    unsigned indices_idx = 0;

    for (unsigned i = start; i < end; i++)
    {
        Bn254FrField s = scalars[i];
        s.unmont_assign();

        for (unsigned widx = 0; widx < windows; widx++)
        {
            unsigned start_bit = widx * window_bits;
            unsigned remain_bits = 254 - start_bit;
            unsigned bits = remain_bits < window_bits ? remain_bits : window_bits;

            unsigned idx = s.get_nbits(start_bit, bits);
            if (!non_zero_count || idx > 0)
            {
                point_indices[indices_idx] = i;
                bucket_indices[indices_idx] = ((widx << window_bits) * !!idx) | idx;
                indices_idx++;
            }
        }
    }
}

__global__ void msm_core(
    unsigned *buckets_indices,
    unsigned *point_indices,
    unsigned *remain_indices,
    Bn254G1 *remain_acc,
    Bn254G1 *buckets,
    Bn254G1Affine *p,
    unsigned window_bits,
    unsigned windows,
    unsigned n)
{
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned workers = gridDim.x * blockDim.x;
    unsigned tasks = n;
    unsigned tpw = (tasks + workers - 1) / workers;
    unsigned start = tpw * gid > tasks ? tasks : tpw * gid;
    unsigned end = start + tpw > tasks ? tasks : start + tpw;

    unsigned offset_mask = (1 << window_bits) - 1;
    unsigned idx = 0;
    Bn254G1 acc = Bn254G1::identity();

    for (unsigned i = start; i < end; i++)
    {
        unsigned next_idx = buckets_indices[i];
        if (next_idx != idx)
        {
            if (idx > 0)
            {
                unsigned windex = idx >> window_bits;
                unsigned offset = idx & offset_mask;
                buckets[(windex << window_bits) + offset] = acc;
            }
            acc = Bn254G1::identity() + p[point_indices[i]];
            idx = next_idx;
        }
        else if (idx > 0)
        {
            acc = acc + p[point_indices[i]];
        }
    }

    remain_acc[gid] = acc;
    remain_indices[gid] = idx;
}

__global__ void batch_collect_msm_remain(
    unsigned **remain_indices,
    Bn254G1 **remain_acc,
    unsigned **next_remain_indices,
    Bn254G1 **next_remain_acc,
    Bn254G1 **buckets,
    unsigned window_bits,
    unsigned windows,
    unsigned n)
{
    unsigned msm_id = blockIdx.y;
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned workers = gridDim.x * blockDim.x;
    unsigned tasks = n;
    unsigned tpw = (tasks + workers - 1) / workers;
    unsigned start = tpw * gid;
    unsigned end = start + tpw > tasks ? tasks : start + tpw;

    unsigned offset_mask = (1 << window_bits) - 1;
    unsigned idx = 0;
    Bn254G1 acc = Bn254G1::identity();

    for (unsigned i = start; i < end; i++)
    {
        unsigned next_idx = remain_indices[msm_id][i];
        if (next_idx != idx)
        {
            if (idx > 0)
            {
                unsigned windex = idx >> window_bits;
                unsigned offset = idx & offset_mask;
                assert(windex < windows);
                assert(offset < 1 << window_bits);
                buckets[msm_id][(windex << window_bits) + offset] += acc;
            }
            acc = Bn254G1::identity() + remain_acc[msm_id][i];
            idx = next_idx;
        }
        else if (idx > 0)
        {
            acc = acc + remain_acc[msm_id][i];
        }
    }

    if (workers == 1)
    {
        unsigned windex = idx >> window_bits;
        unsigned offset = idx & offset_mask;
        if (idx > 0)
        {
            assert(windex < windows);
            assert(offset < 1 << window_bits);
            buckets[msm_id][(windex << window_bits) + offset] += acc;
        }
    }
    else
    {
        next_remain_indices[msm_id][gid] = idx;
        next_remain_acc[msm_id][gid] = acc;
    }
}

__global__ void batch_collect_bucktes(
    Bn254G1 **buckets,
    unsigned windows,
    unsigned window_bits)
{
    unsigned msm_id = blockIdx.x;

    Bn254G1 acc = buckets[msm_id][(windows - 1) << window_bits];
    for (int i = windows - 2; i >= 0; i--)
    {
        for (int j = 0; j < window_bits; j++)
        {
            acc = acc.ec_double();
        }
        acc = acc + buckets[msm_id][i << window_bits];
    }
    buckets[msm_id][0] = acc;
}

__global__ void batch_reduce_buckets(
    Bn254G1 **buckets,
    unsigned windows,
    unsigned window_bits)
{
    unsigned msm_id = blockIdx.x;
    unsigned window_id = blockIdx.y;
    unsigned id = threadIdx.x;
    unsigned workers = blockDim.x;

    Bn254G1 *buckets_start = &buckets[msm_id][window_id << window_bits];

    extern __shared__ Bn254G1 cache[];

    unsigned len = 1 << window_bits;

    if (id == 0)
    {
        buckets_start[0] = Bn254G1::identity();
    }
    __syncthreads();

    while (len > 2)
    {
        len >>= 1;
        {
            unsigned size = len;
            unsigned tasks_per_worker = (size + workers - 1) / workers;
            unsigned start = tasks_per_worker * id;
            unsigned end = start + tasks_per_worker > size ? size : start + tasks_per_worker;
            for (unsigned i = start; i < end; i++)
            {
                buckets_start[i] = buckets_start[i] + buckets_start[i + len];
            }
        }

        __syncthreads();

        {
            unsigned size = len >> 1;
            unsigned tasks_per_worker = (size + workers - 1) / workers;
            unsigned start = tasks_per_worker * id;
            unsigned end = start + tasks_per_worker > size ? size : start + tasks_per_worker;

            for (unsigned i = start; i < end; i++)
            {
                cache[id] = buckets_start[i + len] + buckets_start[i + len + size];
                buckets_start[1 + i] = buckets_start[1 + i] + cache[id];
                buckets_start[len - 1 - i] = buckets_start[len - 1 - i] + cache[id];
            }
        }

        __syncthreads();
    }

    if (id == 0)
    {
        buckets_start[0] = buckets_start[1];
    }
}

__global__ void init_bucktes(
    Bn254G1 *buckets,
    unsigned n)
{
    unsigned gid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (gid > n)
    {
        return;
    }

    buckets[gid] = Bn254G1::identity();
}

extern "C"
{
#define CHECK_RETURN(x)         \
    {                           \
        cudaError_t err = x;    \
        if (err != cudaSuccess) \
        {                       \
            return err;         \
        }                       \
    }

    cudaError_t msm(
        Bn254G1 *res,
        Bn254G1Affine *points,
        Bn254FrField *scalars,
        unsigned *bucket_indices,
        unsigned *point_indices,
        unsigned *sorted_bucket_indices,
        unsigned *sorted_point_indices,
        unsigned *candidate_sort_indices_temp_storage_bytes,
        unsigned *acc_indices_buf,
        int n,
        int windows,
        int window_bits,
        int threads,
        int workers,
        int prepared_sort_indices_temp_storage_bytes,
        int skip_zero,
        cudaStream_t stream)
    {
        assert(workers % threads == 0);
        unsigned blocks = workers / threads;

        unsigned non_zero_count = n * windows;
        unsigned *non_zero_buckets = NULL;

        if (skip_zero)
        {
            non_zero_buckets = acc_indices_buf + workers;
            CHECK_RETURN(cudaMemsetAsync(non_zero_buckets, 0, workers * sizeof(unsigned), stream));
            count_nonzero_buckets<<<blocks, threads, 0, stream>>>(
                non_zero_buckets,
                bucket_indices, point_indices, scalars, windows, window_bits, n);
            acc_nonzero_buckets<<<1, 64, 64 * sizeof(unsigned), stream>>>(
                non_zero_buckets, workers);
            CHECK_RETURN(cudaMemcpyAsync(&non_zero_count, &non_zero_buckets[workers - 1], sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        }

        fill_buckte_indices<<<blocks, threads, 0, stream>>>(
            non_zero_buckets,
            bucket_indices, point_indices, scalars, windows, window_bits, n);
        cudaStreamSynchronize(stream);

        unsigned *sort_indices_temp_storage{};
        size_t sort_indices_temp_storage_bytes;

        int max_windows_deg = 5;
        assert(windows <= 1 << max_windows_deg);
        CHECK_RETURN(cub::DeviceRadixSort::SortPairsDescending(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
            point_indices, sorted_point_indices, non_zero_count, 0, window_bits + max_windows_deg, stream));

        bool alloc_for_sort = sort_indices_temp_storage_bytes > (unsigned)prepared_sort_indices_temp_storage_bytes;
        if (alloc_for_sort)
        {
            printf("realloc for bytes %lu\n", sort_indices_temp_storage_bytes);
            CHECK_RETURN(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
        }
        else
        {
            sort_indices_temp_storage = candidate_sort_indices_temp_storage_bytes;
        }

        CHECK_RETURN(cub::DeviceRadixSort::SortPairsDescending(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices, sorted_bucket_indices,
            point_indices, sorted_point_indices, non_zero_count, 0, window_bits + max_windows_deg, stream));
        if (alloc_for_sort)
        {
            CHECK_RETURN(cudaFreeAsync(sort_indices_temp_storage, stream));
        }

        Bn254G1 *buckets = res;
        unsigned tasks = windows << window_bits;
        unsigned groups = (tasks + threads - 1) / threads;
        init_bucktes<<<groups, threads, 0, stream>>>(buckets, tasks);

        Bn254G1 *remain_acc = &res[windows << window_bits];
        msm_core<<<blocks, threads, 0, stream>>>(
            sorted_bucket_indices, sorted_point_indices,
            acc_indices_buf, remain_acc,
            buckets, points, window_bits, windows, non_zero_count);

        return cudaGetLastError();
    }

    cudaError_t batch_msm_collect(
        unsigned **remain_indices,
        Bn254G1 **remain_acc,
        unsigned **next_remain_indices,
        Bn254G1 **next_remain_acc,
        Bn254G1 **buckets,
        unsigned workers,
        unsigned windows,
        unsigned window_bits,
        unsigned batch_size,
        cudaStream_t stream)
    {
        if (workers > 128)
        {
            batch_collect_msm_remain<<<dim3(2, batch_size), 64, 0, stream>>>(
                remain_indices, remain_acc, next_remain_indices, next_remain_acc, buckets,
                window_bits, windows, workers);

            batch_collect_msm_remain<<<dim3(1, batch_size), 1, 0, stream>>>(
                next_remain_indices, next_remain_acc, NULL, NULL, buckets,
                window_bits, windows, 2 * 64);
        }
        else
        {
            batch_collect_msm_remain<<<dim3(1, batch_size), 1, 0, stream>>>(
                remain_indices, remain_acc, NULL, NULL, buckets,
                window_bits, windows, workers);
        }

        batch_reduce_buckets<<<dim3(batch_size, windows), 128, 128 * sizeof(Bn254G1), stream>>>(buckets, windows, window_bits);
        batch_collect_bucktes<<<batch_size, 1, 0, stream>>>(buckets, windows, window_bits);

        return cudaGetLastError();
    }
}