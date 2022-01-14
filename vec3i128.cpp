#include <immintrin.h>
#include <climits>
#include <cmath>
#include "AVX2Ext.h"
#include <iostream>

struct vec3i128
{
    const double ULongMaxValue = (double)ULLONG_MAX;
    const double LongMaxValue = (double)LLONG_MAX;
    const __m256i TwoP52Mask = _mm256_srli_epi64(_mm256_slli_epi64(_mm256_set1_epi64x(-1), 12), 12);
    const __m256i SignMask = _mm256_set1_epi64x(LLONG_MIN);
    const __m256d TwoP52 = _mm256_set1_pd(pow(2.0, 52.0));
    const __m256d TwoP52_Dec = _mm256_set1_pd(4503599627370495.1);
    const __m256d TwoP52_Inv = _mm256_set1_pd(1.0 / pow(2.0, 52.0));
    const __m256d TwoP52_InvInc = _mm256_set1_pd(2.2204460492503128E-16);
    const __m256d TwoP38 = _mm256_set1_pd(pow(2.0, 38.0));
    const __m256d TwoP38_Inv = _mm256_set1_pd(1.0 / pow(2.0, 38.0));
public:
    __m256i upper, lower;

    vec3i128(long long x, long long y, long long z, long long w)
    {
        upper = _mm256_setzero_si256();
        lower = _mm256_setr_epi64x(x, y, z, w);
    }

    vec3i128(__m256i upper, __m256i lower)
    {
        this->upper = upper;
        this->lower = lower;
    }

    __m256d ToDouble_Loss()
    {
        //正值计算部分-分解两个2 ^ 52的部分并转换
        auto TwoP52upper = _mm256_or_si256(_mm256_slli_epi64(upper, 12), _mm256_srli_epi64(lower, 52));
        auto TwoP52lower = _mm256_and_si256(lower, TwoP52Mask);
        auto UpperDouble = uint64_to_double_loss(TwoP52upper);
        auto LowerDouble = uint64_to_double_loss(TwoP52lower);
        auto positiveResult = _mm256_fmadd_pd(UpperDouble, TwoP52, LowerDouble);

        //负值计算部分-求反
        auto  negativeUpper = _mm256_xor_si256(upper, _mm256_set1_epi64x(-1));
        auto carry = _mm256_cmpeq_epi64(_mm256_setzero_si256(), lower);
        negativeUpper = _mm256_sub_epi64(negativeUpper, carry);
        auto negativeLower = _mm256_sub_epi64(_mm256_setzero_si256(), lower);
        //负值计算部分-分解两个2 ^ 52的部分并转换
        auto negTwoP52upper = _mm256_or_si256(_mm256_slli_epi64(negativeUpper, 12), _mm256_srli_epi64(negativeLower, 52));
        auto negTwoP52Lower = _mm256_and_si256(negativeLower, TwoP52Mask);
        auto negUpperDouble = uint64_to_double_loss(negTwoP52upper);
        auto negLowerDouble = uint64_to_double_loss(negTwoP52Lower);
        auto negativeResult = _mm256_fnmsub_pd(negUpperDouble, TwoP52, negLowerDouble);

        //判断正负并选择
        auto upperIsNegative = _mm256_cmpgt_epi64(_mm256_setzero_si256(), upper);
        auto result = _mm256_blendv_pd(positiveResult, negativeResult, _mm256_castsi256_pd(upperIsNegative));
        return _mm256_mul_pd(result, TwoP38_Inv);
    }
    //works for (-2^104,2^104)
    void SetDouble_Loss(const __m256d& value)
    {
        //取绝对值
        __m256d abs = _mm256_and_pd(value, _mm256_castsi256_pd(_mm256_set1_epi64x(LLONG_MAX)));
        abs = _mm256_mul_pd(abs,TwoP38);
        //转换高低两部分2的52次方
        __m256d upper2P52_d = _mm256_floor_pd(_mm256_mul_pd(abs, TwoP52_Inv));
        __m256i upper2P52 = double_to_uint64_loss(_mm256_min_pd(TwoP52_Dec, upper2P52_d));
        __m256d lower2P52_d = _mm256_fnmadd_pd(upper2P52_d, TwoP52, abs);
        __m256i lower2P52 = double_to_uint64_loss(_mm256_min_pd(TwoP52_Dec, lower2P52_d));
        //将两部分插起来
        __m256i lower = _mm256_or_si256(_mm256_slli_epi64(upper2P52, 52), lower2P52);
        __m256i upper = _mm256_srli_epi64(upper2P52, 12);
        //取反(用于负值)
        __m256i lowerIsZero = _mm256_cmpeq_epi64(lower, _mm256_setzero_si256());
        auto negUpper = _mm256_xor_si256(upper, _mm256_set1_epi64x(-1));
        negUpper = _mm256_sub_epi64(negUpper, lowerIsZero);
        auto negLower = _mm256_sub_epi64(_mm256_setzero_si256(), lower);
        //判断并选择是否使用负值
        __m256i isNegative = _mm256_castpd_si256(_mm256_cmp_pd(value, _mm256_setzero_pd(), _CMP_LT_OQ));
        this->upper = _mm256_blendv_epi8(upper, negUpper, isNegative);
        this->lower = _mm256_blendv_epi8(lower, negLower, isNegative);
    }

    void Add(vec3i128 value)
    {
        this->lower = _mm256_add_epi64(this->lower, value.lower);
        __m256i carry = _mm256_cmpgt_epi64(_mm256_add_epi64(value.lower, SignMask), _mm256_add_epi64(this->lower, SignMask));
        this->upper = _mm256_sub_epi64(_mm256_add_epi64(this->upper, value.upper), carry);

    }

    void Subtract(vec3i128 value)
    {
        __m256i up = _mm256_sub_epi64(this->upper, value.upper);
        __m256i carry = _mm256_cmpgt_epi64(_mm256_add_epi64(value.lower, SignMask), _mm256_add_epi64(this->lower, SignMask));
        this->lower = _mm256_sub_epi64(this->lower, value.lower);
        this->upper = _mm256_add_epi64(up, carry);
    }
};