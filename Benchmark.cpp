#include <iostream>
#include <immintrin.h>
#include "vec3i128.cpp"
#include <benchmark/benchmark.h>

namespace UniverseStructure
{
    vec3i128 test1(5, 62, 15211253663,0);
    vec3i128 test2(55211, 6211, 1523663,0);
    __m256d zero = _mm256_setzero_pd();
    __m256d Pos1e5 = _mm256_setr_pd(1e5, 1e5, 1e5, 1e5);
    __m256d Neg3e38 = _mm256_setr_pd(-3e38, -3e38, -3e38, -3e38);
    static void ConvertFromDouble_Zero(benchmark::State& state)
    {
        for (auto _ : state) {
            test1.SetDouble_Loss(zero);
        }
    }
    static void ConvertToDouble_Zero(benchmark::State& state)
    {
        for (auto _ : state) {
            zero = test1.ToDouble_Loss();
        }
    }

    static void ConvertFromDouble_1e5(benchmark::State& state)
    {
        for (auto _ : state) {
            test1.SetDouble_Loss(Pos1e5);
        }
    }
    static void ConvertToDouble_1e5(benchmark::State& state)
    {
        for (auto _ : state) {
            Pos1e5 = test1.ToDouble_Loss();
        }
    }
    static void ConvertFromDouble_Neg3e38(benchmark::State& state)
    {
        for (auto _ : state) {
            test1.SetDouble_Loss(Neg3e38);
        }
    }
    static void ConvertToDouble_Neg3e38(benchmark::State& state)
    {
        for (auto _ : state) {
            Neg3e38 = test1.ToDouble_Loss();
        }
    }
    static void Add(benchmark::State& state)
    {
        for (auto _ : state) {
            test1.Add(test2);
        }
    }
    static void Subtract(benchmark::State& state)
    {
        for (auto _ : state) {
            test1.Subtract(test2);
        }
    }

//将该函数注册为基准
BENCHMARK(ConvertFromDouble_Zero);
BENCHMARK(ConvertToDouble_Zero);
BENCHMARK(ConvertFromDouble_1e5);
BENCHMARK(ConvertToDouble_1e5);
BENCHMARK(ConvertFromDouble_Neg3e38);
BENCHMARK(ConvertToDouble_Neg3e38);
BENCHMARK(Add);
BENCHMARK(Subtract);
//运行基准测试
BENCHMARK_MAIN();
}