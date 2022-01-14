#include <iostream>
#include <immintrin.h>
#include "vec3i128.cpp"
#include <benchmark/benchmark.h>
using namespace std;
    vec3i128 test1(5, 62, 15211253663,0);
    vec3i128 test2(55211, 6211, 1523663,0);
    __m256d zero = _mm256_setzero_pd();
    __m256d Pos1e5 = _mm256_set1_pd(1e5);
    __m256d Dotzzzz1 = _mm256_set1_pd(0.000000000003);
    __m256d Neg2p66 = _mm256_set1_pd(-73786976294838206464.1234567);
    /*int main()
    {
        test1.SetDouble_Loss(Dotzzzz1);
        cout << hex <<test1.upper.m256i_u64[1] <<endl;
        cout << hex << test1.lower.m256i_u64[1] <<endl;
        cout << test1.ToDouble_Loss().m256d_f64[1] <<endl;
    }
    */
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
            test1.SetDouble_Loss(Neg2p66);
        }
    }
    static void ConvertToDouble_Neg3e38(benchmark::State& state)
    {
        for (auto _ : state) {
            Neg2p66 = test1.ToDouble_Loss();
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
