#ifndef SGEMM_CPU_H
#define SGEMM_CPU_H

template <typename T = float, typename AccT = T>
void gemm_cpu(
    const T* a, const T* b, T* c, int m, int n, int k, AccT alpha, AccT beta) {
    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            AccT sum = 0.0f;
            for (int z = 0; z < k; z++) {
                sum += static_cast<AccT>(a[x * k + z]) *
                       static_cast<AccT>(b[z * n + y]);
            }
            c[x * n + y] = static_cast<T>(alpha * sum + beta * c[x * n + y]);
        }
    }
}

#endif  // SGEMM_CPU_H
