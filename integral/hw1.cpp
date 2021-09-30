#include <mpi.h>

#include <inttypes.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef DEBUG
#define DEBUG_LOG(...) printf(__VA_ARGS__)
#define RELEASE_LOG(...) do {} while(false)
#else
#define DEBUG_LOG(...) do {} while(false)
#define RELEASE_LOG(...) printf(__VA_ARGS__)
#endif

double F(double x) __attribute__((const));

double integrate(double a, double step, int N) __attribute__((const));

double integrate_simple(double a, double b) __attribute__((const));

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int p = 0, id = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc < 2) {
        if (id == 0) {
            DEBUG_LOG(
                "Provide number of partitions. Example:\n\t"
                "./integrate 10  # "
                "Calculate integral using partition of [0, 1] into 10 parts.\n");
        }

        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);  // Get number of pieces in partition
    double step = 1.0 / N;
    int steps_per_process = N / p;

    auto a = 0.0;  // Where the current process should start

    int64_t single_duration = 0, parallel_duration = 0;

    double single_integral = 0;

    if (id == 0) {
        auto start_single = std::chrono::high_resolution_clock::now();
        single_integral = integrate(0, step, N);
        auto end_single = std::chrono::high_resolution_clock::now();
        single_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_single - start_single).count();

        steps_per_process = (N % p == 0) ? steps_per_process : N % p;
        a = (p - 1) * step * steps_per_process;
    } else {
        a = (id - 1) * step * steps_per_process;  // Where current process should start
    }

    auto start_parallel = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);  // Start from here
    
    double result = integrate(a, step, steps_per_process);
    DEBUG_LOG("Integral calculated by process %d on [%lf, %lf]: %.12lf\n", id, a, a + step * steps_per_process, result);

    double parallel_integral = 0;
    MPI_Reduce(&result, &parallel_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(id == 0) {
        auto end_parallel = std::chrono::high_resolution_clock::now();
        parallel_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_parallel).count();

        DEBUG_LOG("Integral computed by single process: %.12lf. Computations took %lf s\n", single_integral, single_duration * 0.000001);
        DEBUG_LOG("Integral computed by %d processes: %.12lf. Computations took %lf s\n", p, parallel_integral, parallel_duration * 0.000001);
        DEBUG_LOG("Speedup is x%lf\n", static_cast<double>(single_duration) / parallel_duration);

        DEBUG_LOG("%" PRId64 " %" PRId64 "\n", single_duration, parallel_duration);
        RELEASE_LOG("%" PRId64 " %" PRId64 "\n", single_duration, parallel_duration);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

// Function to integrate
double F(double x) {
    return 4 / (1 + x * x);
}

// Integrate function
double integrate(double a, double step, int N) {
    double result = 0;

    for (int i = 0; i < N; i++) {
        result += integrate_simple(a + step * i, a + step * (i + 1));
    }

    return result;
}

// Integrate function over [a, b] without partitioning further
double integrate_simple(double a, double b) {
    return (F(a) + F(b)) * (b - a) / 2;
}