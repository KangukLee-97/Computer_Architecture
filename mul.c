#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>

#define SIZE 960   // matrix, size*size 
#define UNROLL 4
void dgemm_unop(int n, double* A, double* B, double* C);
void dgemm_parallel(int n, double* A, double* B, double* C);
void dgemm_loop_unroll(int n, double* A, double* B, double* C);

int main(void)
{
    clock_t start, end;
    double elapsed;

    double* A = (double*)malloc(sizeof(double) * SIZE * SIZE);
    double* B = (double*)malloc(sizeof(double) * SIZE * SIZE);
    double* C = (double*)malloc(sizeof(double) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++)
    {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    // Unoptimized test
    printf("*************** Multiplying with unoptimization ***************\n");
    start = clock();
    dgemm_unop(SIZE, A, B, C);
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.4g\n\n", elapsed);

    // AVX (Subword Parallelism) test
    printf("*************** Multiplying with AVX ***************\n");
    start = clock();
    dgemm_parallel(SIZE, A, B, C);
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time by AVX (subword parallelism): %.4g\n\n", elapsed);

    // Loop unrolling test
    printf("*************** Multiplying with Loop unrolling ***************\n");
    start = clock();
    dgemm_loop_unroll(SIZE, A, B, C);
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time by loop unrolling: %.4g\n", elapsed);

    return 0;
}

// Unoptimized 
void dgemm_unop(int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double cij = C[i + j * n];   // cij = C[i][j]
            for (int k = 0; k < n; k++)
                cij += A[i + k * n] * B[k + j * n];   // cij += A[i][k] * B[k][j]

            C[i + j * n] = cij;   // C[i][j] = cij
        }
    }
}

// AVX (Subword Parallelism) 
void dgemm_parallel(int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; i += 4)
    {
        for (int j = 0; j < n; j++)
        {
            __m256d c0 = _mm256_load_pd(C + i + j * n);   // c0 = C[i][j]
            for (int k = 0; k < n; k++)
            {
                c0 = _mm256_add_pd(c0,
                    _mm256_mul_pd(_mm256_load_pd(A + i + k * n),
                        _mm256_broadcast_sd(B + k + j * n)));   // c0 += A[i][k] * B[k][j]
            }

            _mm256_store_pd(C + i + j * n, c0);   // C[i][j] = c0
        }
    }
}

// Loop unrolling + AVX
void dgemm_loop_unroll(int n, double* A, double* B, double* C)
{
    for (int i = 0; i < n; i += UNROLL * 4)
    {
        for (int j = 0; j < n; j++)
        {
            __m256d c[4];
            for (int x = 0; x < UNROLL; x++)
                c[x] = _mm256_load_pd(C + i + x * 4 + j * n);

            for (int k = 0; k < n; k++)
            {
                __m256d b = _mm256_broadcast_sd(B + k + j * n);
                for (int x = 0; x < UNROLL; x++)
                    c[x] = _mm256_add_pd(c[x],
                        _mm256_mul_pd(_mm256_load_pd(A + n * k + x * 4 + i), b));
            }

            for (int x = 0; x < UNROLL; x++)
                _mm256_store_pd(C + i + x * 4 + j * n, c[x]);
        }
    }
}