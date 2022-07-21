# Vector3Int128

128 bit integer 3D vector structure accelerated by avx2 instruction set

There are only four operations: addition, subtraction, conversion from floating point and conversion to floating point

The converted floating-point only supports iee754-binary64 at present

Due to the limitation of avx2 instruction set, for performance, the numerical range of conversion floating point is (-2^104, 2^104) (of course, this does not represent its representation range)

Of course, from the current use, this scope limit is irrelevant

If avx512 is supported, this problem can be solved and the operation speed can be further improved
