#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// Error-checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSolver error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t matBytes = n*(size_t)n*sizeof(double);
    size_t vecBytes = n*(size_t)sizeof(double);

    // Host buffers
    double *h_A = (double*)malloc(matBytes);
    double *h_b = (double*)malloc(vecBytes);
    double *h_b0 = (double*)malloc(vecBytes);
    double *h_r = (double*)malloc(vecBytes);
    if(!h_A||!h_b||!h_b0||!h_r){fprintf(stderr,"Host alloc failed\n");return EXIT_FAILURE;}

    // Init A and b
    srand(0);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++) h_A[i*(size_t)n+j] = ((double)rand()/RAND_MAX) + (i==j ? n : 0);
        h_b[i] = (double)rand()/RAND_MAX;
        h_b0[i] = h_b[i];
    }

    // Device buffers
    double *d_A, *d_r;
    double *d_b;
    int *d_Ipiv, *d_info;
    double *d_work;
    int lwork;

    CUDA_CHECK(cudaMalloc(&d_A, matBytes));
    CUDA_CHECK(cudaMalloc(&d_b, vecBytes));
    CUDA_CHECK(cudaMalloc(&d_r, vecBytes));
    CUDA_CHECK(cudaMalloc(&d_Ipiv, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, matBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, vecBytes, cudaMemcpyHostToDevice));

    // cuSolver handle
    cusolverDnHandle_t solverH;
    CUSOLVER_CHECK(cusolverDnCreate(&solverH));

    // workspace
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(solverH, n, n, d_A, n, &lwork));
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));

    // Initial factorization and solve
    auto t0 = std::chrono::high_resolution_clock::now();
    CUSOLVER_CHECK(cusolverDnDgetrf(solverH, n, n, d_A, n, d_work, d_Ipiv, d_info));
    CUSOLVER_CHECK(cusolverDnDgetrs(solverH, CUBLAS_OP_N, n, 1, d_A, n, d_Ipiv, d_b, n, d_info));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_b, d_b, vecBytes, cudaMemcpyDeviceToHost));

    // Iterative refinement
    const int max_iter = 5;
    const double tol = 1e-12;
    double rel_res = 0.0;
    int iters;
    for(iters = 0; iters < max_iter; ++iters) {
        // Compute residual r = b0 - A*x
        double err = 0.0, norm = 0.0;
        for(int i=0;i<n;i++){
            double sum = 0.0;
            for(int j=0;j<n;j++) sum += h_A[i*(size_t)n+j] * h_b[j];
            double ri = h_b0[i] - sum;
            h_r[i] = ri;
            err += ri * ri;
            norm += h_b0[i] * h_b0[i];
        }
        rel_res = sqrt(err / norm);
        if(rel_res < tol) break;
        // Solve correction A*dr = r
        CUDA_CHECK(cudaMemcpy(d_r, h_r, vecBytes, cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnDgetrs(solverH, CUBLAS_OP_N, n, 1, d_A, n, d_Ipiv, d_r, n, d_info));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_r, d_r, vecBytes, cudaMemcpyDeviceToHost));
        for(int i=0;i<n;i++) h_b[i] += h_r[i];
        CUDA_CHECK(cudaMemcpy(d_b, h_b, vecBytes, cudaMemcpyHostToDevice));
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("total solve time = %.6f s\n", elapsed);
    printf("relative residual=%.3e\n", rel_res);

    // cleanup
    cudaFree(d_A); cudaFree(d_b); cudaFree(d_r);
    cudaFree(d_Ipiv); cudaFree(d_info); cudaFree(d_work);
    cusolverDnDestroy(solverH);
    free(h_A); free(h_b); free(h_b0); free(h_r);
    return EXIT_SUCCESS;
}
