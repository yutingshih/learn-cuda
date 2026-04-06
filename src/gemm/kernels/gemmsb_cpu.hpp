#ifndef GEMMSB_CPU_HPP
#define GEMMSB_CPU_HPP

template <typename T = float, typename AccT = T>
void gemmsb_cpu(const T *a,
                const T *b,
                T *c,
                int B,
                int M,
                int N,
                int K,
                long long int strideA,
                long long int strideB,
                long long int strideC,
                AccT alpha,
                AccT beta)
{
    for (int i = 0; i < B; i++) {
        for (int x = 0; x < M; x++) {
            for (int y = 0; y < N; y++) {
                AccT sum = 0.0f;
                for (int z = 0; z < K; z++) {
                    sum += static_cast<AccT>(a[x * K + z]) *
                           static_cast<AccT>(b[z * N + y]);
                }
                c[x * N + y] =
                    static_cast<T>(alpha * sum + beta * c[x * N + y]);
            }
        }
        a += strideA;
        b += strideB;
        c += strideC;
    }
}
#endif  // GEMMSB_CPU_HPP
