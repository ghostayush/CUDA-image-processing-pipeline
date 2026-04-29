/*
 * image_kernels.cu
 * ════════════════════════════════════════════════════════════════════
 * Project 3: GPU Image Processing Pipeline
 *
 * Kernels implemented:
 *   1. gaussian_blur_naive     – 2D convolution, global memory only
 *   2. gaussian_blur_separable – two 1D passes using separable filter
 *   3. gaussian_blur_shmem     – tiled 2D with shared memory + halo
 *   4. sobel_edge_detect       – Gx/Gy gradient magnitude + threshold
 *   5. rgb_to_grayscale        – parallel pixel conversion
 *   6. Full pipeline           – RGB→gray→blur→edge, single stream
 *
 * All kernels process images as float32 in [0, 255] range.
 *
 * Compile (standalone benchmark):
 *   nvcc -O2 -arch=sm_75 --use_fast_math image_kernels.cu -o img_bench
 *
 * Run:
 *   ./img_bench
 * ════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

/* ── tuneable constants ─────────────────────────────────────────── */
#define BLOCK_W     32          /* thread-block width                 */
#define BLOCK_H     8           /* thread-block height                */
#define KERNEL_R    2           /* Gaussian radius  (5×5 = 2r+1)     */
#define KERNEL_SIZE (2*KERNEL_R+1)   /* = 5                          */
#define HALO        KERNEL_R         /* halo border for shared mem    */

/* ── error macro ────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do { \
    cudaError_t e=(x); \
    if(e!=cudaSuccess){ \
        fprintf(stderr,"CUDA error %s:%d  %s\n",__FILE__,__LINE__, \
                cudaGetErrorString(e)); exit(1); } \
} while(0)


/* ══════════════════════════════════════════════════════════════════
   CONSTANT MEMORY — Gaussian kernel coefficients
   __constant__ is cached and broadcast efficiently to all threads.
   This is a key optimisation: kernel weights are read-only and the
   same for every thread, so constant memory avoids L2 pressure.
   ══════════════════════════════════════════════════════════════════ */

/* 5×5 Gaussian kernel (σ=1.0), normalised so all weights sum to 1 */
__constant__ float c_gauss2d[KERNEL_SIZE][KERNEL_SIZE] = {
    {0.00296902f, 0.01330621f, 0.02193823f, 0.01330621f, 0.00296902f},
    {0.01330621f, 0.05963430f, 0.09832033f, 0.05963430f, 0.01330621f},
    {0.02193823f, 0.09832033f, 0.16210282f, 0.09832033f, 0.02193823f},
    {0.01330621f, 0.05963430f, 0.09832033f, 0.05963430f, 0.01330621f},
    {0.00296902f, 0.01330621f, 0.02193823f, 0.01330621f, 0.00296902f}
};

/* Separable 1D kernel [1, 4, 6, 4, 1] / 16 (binomial approximation) */
__constant__ float c_gauss1d[KERNEL_SIZE] = {
    0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f
};

/* Sobel operator kernels (stored for reference — inlined in kernel) */
/* Gx: [-1  0  1; -2  0  2; -1  0  1]                               */
/* Gy: [-1 -2 -1;  0  0  0;  1  2  1]                               */


/* ══════════════════════════════════════════════════════════════════
   KERNEL 1 — RGB to Grayscale
   One thread per pixel. Applies ITU-R BT.601 luminance weights.
   ══════════════════════════════════════════════════════════════════ */
__global__ void rgb_to_grayscale(
    const unsigned char* __restrict__ rgb,   /* [H, W, 3] interleaved */
    float*               __restrict__ gray,  /* [H, W]                */
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int px = (y * W + x) * 3;
    float r = (float)rgb[px + 0];
    float g = (float)rgb[px + 1];
    float b = (float)rgb[px + 2];

    /* ITU-R BT.601 weights */
    gray[y * W + x] = 0.299f * r + 0.587f * g + 0.114f * b;
}


/* ══════════════════════════════════════════════════════════════════
   KERNEL 2 — Naive 2D Gaussian Blur
   One thread per output pixel. Reads from global memory for every
   pixel in the kernel neighbourhood — no data reuse.
   ══════════════════════════════════════════════════════════════════ */
__global__ void gaussian_blur_naive(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    for (int ky = -KERNEL_R; ky <= KERNEL_R; ++ky) {
        for (int kx = -KERNEL_R; kx <= KERNEL_R; ++kx) {
            int nx = min(max(x + kx, 0), W - 1);  /* clamp-to-border */
            int ny = min(max(y + ky, 0), H - 1);
            acc += c_gauss2d[ky + KERNEL_R][kx + KERNEL_R] * in[ny * W + nx];
        }
    }
    out[y * W + x] = acc;
}


/* ══════════════════════════════════════════════════════════════════
   KERNEL 3a — Separable Horizontal Pass
   Applies 1D Gaussian along rows. Each thread computes one pixel.
   Separable filter: O(2r+1) multiplies vs O((2r+1)²) for 2D.
   For r=2: 5 vs 25 multiplies — 5× fewer FLOPs.
   ══════════════════════════════════════════════════════════════════ */
__global__ void gaussian_blur_horiz(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    #pragma unroll
    for (int kx = -KERNEL_R; kx <= KERNEL_R; ++kx) {
        int nx = min(max(x + kx, 0), W - 1);
        acc += c_gauss1d[kx + KERNEL_R] * in[y * W + nx];
    }
    out[y * W + x] = acc;
}


/* ══════════════════════════════════════════════════════════════════
   KERNEL 3b — Separable Vertical Pass
   Applies 1D Gaussian along columns.
   ══════════════════════════════════════════════════════════════════ */
__global__ void gaussian_blur_vert(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    #pragma unroll
    for (int ky = -KERNEL_R; ky <= KERNEL_R; ++ky) {
        int ny = min(max(y + ky, 0), H - 1);
        acc += c_gauss1d[ky + KERNEL_R] * in[ny * W + x];
    }
    out[y * W + x] = acc;
}


/* ══════════════════════════════════════════════════════════════════
   KERNEL 4 — Gaussian Blur with Shared Memory + Halo
   Each block loads a tile of (BLOCK_W + 2*HALO) × (BLOCK_H + 2*HALO)
   pixels into shared memory, including border pixels needed by
   neighbouring threads (the "halo"). Then computes 2D convolution
   entirely from shared memory — far fewer global reads.

   Memory traffic: 1 global read per output pixel (plus halo)
                   vs 25 global reads for naive (with r=2).
   ══════════════════════════════════════════════════════════════════ */
#define SHMEM_W (BLOCK_W + 2*HALO)
#define SHMEM_H (BLOCK_H + 2*HALO)

__global__ void gaussian_blur_shmem(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H)
{
    __shared__ float smem[SHMEM_H][SHMEM_W];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * BLOCK_W + tx;
    int y  = blockIdx.y * BLOCK_H + ty;

    /* ── load tile into shared memory ──────────────────────────── */
    /* Each thread loads its own pixel plus participates in loading halo */
    for (int dy = ty; dy < SHMEM_H; dy += BLOCK_H) {
        for (int dx = tx; dx < SHMEM_W; dx += BLOCK_W) {
            int gx = blockIdx.x * BLOCK_W + dx - HALO;
            int gy = blockIdx.y * BLOCK_H + dy - HALO;
            /* clamp-to-border for out-of-bounds */
            int cx = min(max(gx, 0), W - 1);
            int cy = min(max(gy, 0), H - 1);
            smem[dy][dx] = in[cy * W + cx];
        }
    }
    __syncthreads();  /* all threads done loading → safe to read */

    if (x >= W || y >= H) return;

    /* ── 2D convolution from shared memory (no global reads) ───── */
    float acc = 0.0f;
    #pragma unroll
    for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
            acc += c_gauss2d[ky][kx] * smem[ty + ky][tx + kx];
        }
    }
    out[y * W + x] = acc;
}


/* ══════════════════════════════════════════════════════════════════
   KERNEL 5 — Sobel Edge Detection
   Applies Sobel operator on a grayscale (blurred) image.
   Computes gradient magnitude: G = sqrt(Gx² + Gy²)
   Applies threshold to produce binary edge map.

   Gx = [-1  0  1      Gy = [-1 -2 -1
          -2  0  2            0  0  0
          -1  0  1]           1  2  1]
   ══════════════════════════════════════════════════════════════════ */
__global__ void sobel_edge_detect(
    const float* __restrict__ in,    /* blurred grayscale [H, W] */
    float*       __restrict__ edges, /* edge magnitude   [H, W] */
    unsigned char* __restrict__ edge_binary, /* thresholded [H,W] */
    int W, int H,
    float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* skip border pixels — Sobel needs 1-pixel border */
    if (x == 0 || x >= W-1 || y == 0 || y >= H-1) {
        if (x < W && y < H) { edges[y*W+x] = 0.f; edge_binary[y*W+x] = 0; }
        return;
    }

    /* load 3×3 neighbourhood */
    float p00 = in[(y-1)*W + (x-1)], p01 = in[(y-1)*W + x], p02 = in[(y-1)*W + (x+1)];
    float p10 = in[ y   *W + (x-1)], /* p11 */               p12 = in[ y   *W + (x+1)];
    float p20 = in[(y+1)*W + (x-1)], p21 = in[(y+1)*W + x], p22 = in[(y+1)*W + (x+1)];

    /* Gx and Gy */
    float gx = -p00 + p02 - 2.f*p10 + 2.f*p12 - p20 + p22;
    float gy = -p00 - 2.f*p01 - p02 + p20 + 2.f*p21 + p22;

    float mag = sqrtf(gx*gx + gy*gy);
    edges[y*W+x]        = mag;
    edge_binary[y*W+x]  = (mag > threshold) ? 255 : 0;
}


/* ══════════════════════════════════════════════════════════════════
   KERNEL 6 — Sobel with shared memory (faster for large images)
   Loads 3×3 neighbourhood into shared memory to avoid redundant
   global reads (each pixel is read by up to 9 threads in naive).
   ══════════════════════════════════════════════════════════════════ */
#define SOBEL_BW 32
#define SOBEL_BH  8
#define SOBEL_SW (SOBEL_BW + 2)
#define SOBEL_SH (SOBEL_BH + 2)

__global__ void sobel_edge_shmem(
    const float*   __restrict__ in,
    float*         __restrict__ edges,
    unsigned char* __restrict__ edge_binary,
    int W, int H,
    float threshold)
{
    __shared__ float smem[SOBEL_SH][SOBEL_SW];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * SOBEL_BW + tx;
    int y  = blockIdx.y * SOBEL_BH + ty;

    /* load tile + 1-pixel halo */
    for (int dy = ty; dy < SOBEL_SH; dy += SOBEL_BH) {
        for (int dx = tx; dx < SOBEL_SW; dx += SOBEL_BW) {
            int gx = blockIdx.x * SOBEL_BW + dx - 1;
            int gy = blockIdx.y * SOBEL_BH + dy - 1;
            int cx = min(max(gx, 0), W-1);
            int cy = min(max(gy, 0), H-1);
            smem[dy][dx] = in[cy * W + cx];
        }
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    float p00=smem[ty  ][tx  ], p01=smem[ty  ][tx+1], p02=smem[ty  ][tx+2];
    float p10=smem[ty+1][tx  ],                         p12=smem[ty+1][tx+2];
    float p20=smem[ty+2][tx  ], p21=smem[ty+2][tx+1], p22=smem[ty+2][tx+2];

    float gx = -p00 + p02 - 2.f*p10 + 2.f*p12 - p20 + p22;
    float gy = -p00 - 2.f*p01 - p02 + p20 + 2.f*p21 + p22;

    float mag = sqrtf(gx*gx + gy*gy);
    if (x > 0 && x < W-1 && y > 0 && y < H-1) {
        edges[y*W+x]       = mag;
        edge_binary[y*W+x] = (mag > threshold) ? 255 : 0;
    } else {
        edges[y*W+x] = 0.f; edge_binary[y*W+x] = 0;
    }
}


/* ══════════════════════════════════════════════════════════════════
   TIMING HELPERS
   ══════════════════════════════════════════════════════════════════ */
static float gpu_ms(cudaEvent_t s, cudaEvent_t e) {
    float ms; CUDA_CHECK(cudaEventSynchronize(e));
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e)); return ms;
}


/* ══════════════════════════════════════════════════════════════════
   CPU REFERENCE IMPLEMENTATIONS
   ══════════════════════════════════════════════════════════════════ */
static void cpu_gaussian_blur(const float* in, float* out,
                               int W, int H)
{
    static const float k2d[5][5] = {
        {0.00296902f, 0.01330621f, 0.02193823f, 0.01330621f, 0.00296902f},
        {0.01330621f, 0.05963430f, 0.09832033f, 0.05963430f, 0.01330621f},
        {0.02193823f, 0.09832033f, 0.16210282f, 0.09832033f, 0.02193823f},
        {0.01330621f, 0.05963430f, 0.09832033f, 0.05963430f, 0.01330621f},
        {0.00296902f, 0.01330621f, 0.02193823f, 0.01330621f, 0.00296902f}
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.f;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int nx = (x+kx < 0) ? 0 : (x+kx >= W ? W-1 : x+kx);
                    int ny = (y+ky < 0) ? 0 : (y+ky >= H ? H-1 : y+ky);
                    acc += k2d[ky+2][kx+2] * in[ny*W + nx];
                }
            }
            out[y*W+x] = acc;
        }
    }
}

static void cpu_sobel(const float* in, unsigned char* edges,
                       int W, int H, float threshold)
{
    for (int y = 1; y < H-1; ++y) {
        for (int x = 1; x < W-1; ++x) {
            float p00=in[(y-1)*W+(x-1)], p01=in[(y-1)*W+x], p02=in[(y-1)*W+(x+1)];
            float p10=in[y*W+(x-1)],                          p12=in[y*W+(x+1)];
            float p20=in[(y+1)*W+(x-1)], p21=in[(y+1)*W+x], p22=in[(y+1)*W+(x+1)];
            float gx = -p00+p02-2.f*p10+2.f*p12-p20+p22;
            float gy = -p00-2.f*p01-p02+p20+2.f*p21+p22;
            edges[y*W+x] = (sqrtf(gx*gx+gy*gy) > threshold) ? 255 : 0;
        }
    }
}

static void cpu_rgb_to_gray(const unsigned char* rgb, float* gray, int W, int H)
{
    for (int i = 0; i < W*H; ++i) {
        gray[i] = 0.299f*(float)rgb[i*3+0]
                + 0.587f*(float)rgb[i*3+1]
                + 0.114f*(float)rgb[i*3+2];
    }
}


/* ══════════════════════════════════════════════════════════════════
   BENCHMARK — one image size
   ══════════════════════════════════════════════════════════════════ */
typedef struct {
    int   W, H;
    float naive_ms, sep_ms, shmem_ms, sobel_ms, total_gpu_ms;
    float cpu_blur_ms, cpu_total_ms;
    float max_err_blur, max_err_edges;
} BenchResult;

static BenchResult benchmark_image(int W, int H,
                                    cudaEvent_t evs, cudaEvent_t eve,
                                    int warmup, int reps)
{
    BenchResult res = {W, H, 0};
    size_t npix = (size_t)W * H;
    size_t rgb_bytes  = npix * 3 * sizeof(unsigned char);
    size_t f32_bytes  = npix * sizeof(float);
    size_t u8_bytes   = npix * sizeof(unsigned char);

    /* ── synthesise test image (gradient + checkerboard) ───── */
    unsigned char* h_rgb = (unsigned char*)malloc(rgb_bytes);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int px = (y * W + x) * 3;
            /* R: horizontal gradient, G: vertical, B: checkerboard edges */
            h_rgb[px+0] = (unsigned char)(255 * x / W);
            h_rgb[px+1] = (unsigned char)(255 * y / H);
            h_rgb[px+2] = (unsigned char)(((x/32 + y/32) & 1) * 255);
        }
    }

    /* host output buffers */
    float*         h_gray     = (float*)malloc(f32_bytes);
    float*         h_blurred  = (float*)malloc(f32_bytes);
    unsigned char* h_edges    = (unsigned char*)malloc(u8_bytes);
    float*         h_blur_ref = (float*)malloc(f32_bytes);
    unsigned char* h_edge_ref = (unsigned char*)malloc(u8_bytes);

    /* ── device alloc ─────────────────────────────────────── */
    unsigned char* d_rgb;
    float *d_gray, *d_tmp, *d_blurred, *d_edges_f;
    unsigned char* d_edges_u8;

    CUDA_CHECK(cudaMalloc(&d_rgb,      rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray,     f32_bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp,      f32_bytes)); /* scratch for separable pass */
    CUDA_CHECK(cudaMalloc(&d_blurred,  f32_bytes));
    CUDA_CHECK(cudaMalloc(&d_edges_f,  f32_bytes));
    CUDA_CHECK(cudaMalloc(&d_edges_u8, u8_bytes));

    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((W + BLOCK_W-1)/BLOCK_W, (H + BLOCK_H-1)/BLOCK_H);
    dim3 block_sobel(SOBEL_BW, SOBEL_BH);
    dim3 grid_sobel((W + SOBEL_BW-1)/SOBEL_BW, (H + SOBEL_BH-1)/SOBEL_BH);
    float threshold = 30.f;
    float ms;

    /* ── always compute grayscale first (same for all kernels) ─ */
    rgb_to_grayscale<<<grid, block>>>(d_rgb, d_gray, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ──────────────────────────────────────────────────────────
       Kernel A: Naive 2D blur
       ────────────────────────────────────────────────────────── */
    for (int i=0;i<warmup;++i) gaussian_blur_naive<<<grid,block>>>(d_gray,d_blurred,W,H);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(evs));
    for (int i=0;i<reps;++i) gaussian_blur_naive<<<grid,block>>>(d_gray,d_blurred,W,H);
    CUDA_CHECK(cudaEventRecord(eve));
    res.naive_ms = gpu_ms(evs,eve)/reps;

    /* ──────────────────────────────────────────────────────────
       Kernel B: Separable blur (H-pass then V-pass)
       ────────────────────────────────────────────────────────── */
    for (int i=0;i<warmup;++i) {
        gaussian_blur_horiz<<<grid,block>>>(d_gray,d_tmp,W,H);
        gaussian_blur_vert <<<grid,block>>>(d_tmp,d_blurred,W,H);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(evs));
    for (int i=0;i<reps;++i) {
        gaussian_blur_horiz<<<grid,block>>>(d_gray,d_tmp,W,H);
        gaussian_blur_vert <<<grid,block>>>(d_tmp,d_blurred,W,H);
    }
    CUDA_CHECK(cudaEventRecord(eve));
    res.sep_ms = gpu_ms(evs,eve)/reps;

    /* ──────────────────────────────────────────────────────────
       Kernel C: Shared memory tiled blur
       ────────────────────────────────────────────────────────── */
    for (int i=0;i<warmup;++i) gaussian_blur_shmem<<<grid,block>>>(d_gray,d_blurred,W,H);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(evs));
    for (int i=0;i<reps;++i) gaussian_blur_shmem<<<grid,block>>>(d_gray,d_blurred,W,H);
    CUDA_CHECK(cudaEventRecord(eve));
    res.shmem_ms = gpu_ms(evs,eve)/reps;
    CUDA_CHECK(cudaMemcpy(h_blurred, d_blurred, f32_bytes, cudaMemcpyDeviceToHost));

    /* ──────────────────────────────────────────────────────────
       Kernel D: Sobel edge detection (on shared-mem-blurred image)
       ────────────────────────────────────────────────────────── */
    for (int i=0;i<warmup;++i)
        sobel_edge_shmem<<<grid_sobel,block_sobel>>>(d_blurred,d_edges_f,d_edges_u8,W,H,threshold);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(evs));
    for (int i=0;i<reps;++i)
        sobel_edge_shmem<<<grid_sobel,block_sobel>>>(d_blurred,d_edges_f,d_edges_u8,W,H,threshold);
    CUDA_CHECK(cudaEventRecord(eve));
    res.sobel_ms = gpu_ms(evs,eve)/reps;
    CUDA_CHECK(cudaMemcpy(h_edges, d_edges_u8, u8_bytes, cudaMemcpyDeviceToHost));

    /* ──────────────────────────────────────────────────────────
       Full pipeline timing (gray + shmem_blur + sobel), single stream
       ────────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaEventRecord(evs));
    for (int i=0;i<reps;++i) {
        rgb_to_grayscale  <<<grid,      block      >>>(d_rgb, d_gray, W, H);
        gaussian_blur_shmem<<<grid,     block      >>>(d_gray, d_blurred, W, H);
        sobel_edge_shmem  <<<grid_sobel,block_sobel>>>(d_blurred,d_edges_f,d_edges_u8,W,H,threshold);
    }
    CUDA_CHECK(cudaEventRecord(eve));
    res.total_gpu_ms = gpu_ms(evs,eve)/reps;

    /* ──────────────────────────────────────────────────────────
       CPU reference pipeline
       ────────────────────────────────────────────────────────── */
    {
        /* CPU grayscale */
        cpu_rgb_to_gray(h_rgb, h_gray, W, H);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        cpu_gaussian_blur(h_gray, h_blur_ref, W, H);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        res.cpu_blur_ms = (float)((t1.tv_sec-t0.tv_sec)*1e3 +
                                   (t1.tv_nsec-t0.tv_nsec)*1e-6);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        cpu_rgb_to_gray(h_rgb, h_gray, W, H);
        cpu_gaussian_blur(h_gray, h_blur_ref, W, H);
        cpu_sobel(h_blur_ref, h_edge_ref, W, H, threshold);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        res.cpu_total_ms = (float)((t1.tv_sec-t0.tv_sec)*1e3 +
                                    (t1.tv_nsec-t0.tv_nsec)*1e-6);

        /* correctness: max error on blur output */
        float max_err = 0.f;
        for (int i=0;i<(int)npix;++i) {
            float e = fabsf(h_blurred[i] - h_blur_ref[i]);
            if (e > max_err) max_err = e;
        }
        res.max_err_blur = max_err;

        /* edge pixel agreement */
        int match=0;
        for (int i=0;i<(int)npix;++i)
            if (h_edges[i]==h_edge_ref[i]) match++;
        res.max_err_edges = 100.f * (1.f - (float)match / npix);
    }

    /* cleanup */
    CUDA_CHECK(cudaFree(d_rgb)); CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_edges_f)); CUDA_CHECK(cudaFree(d_edges_u8));
    free(h_rgb); free(h_gray); free(h_blurred); free(h_edges);
    free(h_blur_ref); free(h_edge_ref);

    return res;
}


/* ══════════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════════ */
int main(void)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("══════════════════════════════════════════════════════════\n");
    printf("GPU : %s\n", prop.name);
    printf("SMs : %d   Global mem: %.0f MB   Shared/SM: %.0f KB\n",
           prop.multiProcessorCount,
           prop.totalGlobalMem/1e6f,
           prop.sharedMemPerBlock/1024.f);
    printf("Block: %dx%d   Kernel: %dx%d   Halo: %d\n",
           BLOCK_W, BLOCK_H, KERNEL_SIZE, KERNEL_SIZE, HALO);
    printf("══════════════════════════════════════════════════════════\n\n");

    cudaEvent_t evs, eve;
    CUDA_CHECK(cudaEventCreate(&evs));
    CUDA_CHECK(cudaEventCreate(&eve));

    /* image sizes to benchmark */
    int sizes[][2] = {
        {640,  480},   /* VGA    */
        {1280, 720},   /* 720p   */
        {1920, 1080},  /* 1080p  */
        {3840, 2160},  /* 4K     */
    };
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    FILE* csv = fopen("img_results.csv", "w");
    fprintf(csv, "W,H,megapixels,naive_ms,sep_ms,shmem_ms,sobel_ms,"
                 "total_gpu_ms,cpu_blur_ms,cpu_total_ms,"
                 "speedup_blur,speedup_total,max_err_blur,edge_disagree_pct\n");

    printf("%-10s  %8s  %8s  %8s  %8s  %10s  %10s  %8s  %8s\n",
           "Size", "Naive(ms)", "Sep(ms)", "Shmem(ms)", "Sobel(ms)",
           "GPU tot(ms)", "CPU tot(ms)", "Speedup", "Blur err");
    printf("─────────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < nsizes; ++i) {
        int W = sizes[i][0], H = sizes[i][1];
        BenchResult r = benchmark_image(W, H, evs, eve, 3, 20);

        float speedup_blur  = r.cpu_blur_ms  / r.shmem_ms;
        float speedup_total = r.cpu_total_ms / r.total_gpu_ms;

        printf("%-10s  %8.2f  %8.2f  %8.2f  %8.2f  %10.2f  %10.2f  %7.1fx  err=%.2f\n",
               W==640?"640×480":W==1280?"1280×720":W==1920?"1920×1080":"3840×2160",
               r.naive_ms, r.sep_ms, r.shmem_ms, r.sobel_ms,
               r.total_gpu_ms, r.cpu_total_ms,
               speedup_total, r.max_err_blur);

        fprintf(csv, "%d,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f\n",
                W, H, (float)W*H/1e6f,
                r.naive_ms, r.sep_ms, r.shmem_ms, r.sobel_ms,
                r.total_gpu_ms, r.cpu_blur_ms, r.cpu_total_ms,
                r.cpu_blur_ms/r.shmem_ms, r.cpu_total_ms/r.total_gpu_ms,
                r.max_err_blur, r.max_err_edges);
    }

    fclose(csv);
    CUDA_CHECK(cudaEventDestroy(evs));
    CUDA_CHECK(cudaEventDestroy(eve));

    printf("\nResults saved to img_results.csv\n");
    return 0;
}
