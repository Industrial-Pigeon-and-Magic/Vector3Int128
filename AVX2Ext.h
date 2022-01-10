#include <immintrin.h>
#pragma once

const __m256d MajikNumber1 = _mm256_set1_pd(0x0010000000000000);
const __m256d MajikNumber2 = _mm256_set1_pd(0x0018000000000000);
//  Only works for inputs in the range: [0, 2^52)
inline __m256i double_to_uint64_loss(__m256d x) {
    x = _mm256_add_pd(x, MajikNumber1);
    return _mm256_xor_si256(
        _mm256_castpd_si256(x),
        _mm256_castpd_si256(MajikNumber1)
    );
}

//  Only works for inputs in the range: [-2^51, 2^51]
inline __m256i double_to_int64_loss(__m256d x) {
    x = _mm256_add_pd(x, MajikNumber2);
    return _mm256_sub_epi64(
        _mm256_castpd_si256(x),
        _mm256_castpd_si256(MajikNumber2)
    );
}

//  Only works for inputs in the range: [0, 2^52)
inline __m256d uint64_to_double_loss(__m256i x) {
    x = _mm256_or_si256(x, _mm256_castpd_si256(MajikNumber1));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), MajikNumber1);
}

//  Only works for inputs in the range: [-2^51, 2^51]
inline __m256d int64_to_double_loss(__m256i x) {
    x = _mm256_add_epi64(x, _mm256_castpd_si256(MajikNumber2));
    return _mm256_sub_pd(_mm256_castsi256_pd(x), MajikNumber2);
}