#pragma once
#ifndef DSFMT_AVX_PARAMS_H
#define DSFMT_AVX_PARAMS_H

namespace {
    using namespace MersenneTwister;
    using namespace std;

#define DSFMTAVX_SL1 19
#define DSFMTAVX_SR 12
#define DSFMTAVX_LOW_MASK  UINT64_C(0x000FFFFFFFFFFFFF)
#define DSFMTAVX_HIGH_CONST UINT64_C(0x3FF0000000000000)

#if DSFMTAVX_MEXP == 1279
#define DSFMTAVX_SIZE 5
#define DSFMTAVX_POS1 3
    const w256_t mask1 = {{UINT64_C(0x000ff9ff37fefbbf),
                           UINT64_C(0x0007bebd2cebad7c),
                           UINT64_C(0x000bfffbcfaddfce),
                           UINT64_C(0x0005affefd978fff)}};
    const w256_t fix1 = {{UINT64_C(0xfbc5b6c65831cf64),
                          UINT64_C(0x328ec06650089455),
                          UINT64_C(0xa6ea0cee0217b775),
                          UINT64_C(0x119ade137f700f73)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0xb6c4200000000000)}};
#endif // 1279
#if DSFMTAVX_MEXP == 2281
#define DSFMTAVX_SIZE 10
#define DSFMTAVX_POS1 5
    const w256_t mask1 = {{UINT64_C(0x00053fdfdbdecff4),
                           UINT64_C(0x000fd5edfbdd7fb7),
                           UINT64_C(0x000ffcbfb797f6f5),
                           UINT64_C(0x000f69effd89efac)}};
    const w256_t fix1 = {{UINT64_C(0x7aee281d3142542a),
                          UINT64_C(0x176a4fb0881dafba),
                          UINT64_C(0x13e626916da50640),
                          UINT64_C(0x65b20b60bd5c0674)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0xdbf3c390a41d2200)}};
#endif // 2281
#if DSFMTAVX_MEXP == 4253
#define DSFMTAVX_SIZE 20
#define DSFMTAVX_POS1 9
    const w256_t mask1 = {{UINT64_C(0x000a9feff9ebff9f),
                           UINT64_C(0x000efeffbbf3777f),
                           UINT64_C(0x0001bfffdbfdf7bf),
                           UINT64_C(0x000c7e3fd779bded)}};
    const w256_t fix1 = {{UINT64_C(0xa0cfe3b5ab1db41d),
                          UINT64_C(0xb34f9a51d50878c4),
                          UINT64_C(0xabebbce7898197fe),
                          UINT64_C(0x508ddb4ba74644bf)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000000000001),
                          UINT64_C(0x2ada8a6cc0000000),
                          UINT64_C(0xf566075647a7b032),
                          UINT64_C(0x4a8f059463e81559)}};
#endif // 4253
#if DSFMTAVX_MEXP == 19937
#define DSFMTAVX_SIZE 95
#define DSFMTAVX_POS1 47
    const w256_t mask1 = {{UINT64_C(0x000f7eefaefbd7e9),
                           UINT64_C(0x000cd7fe2ffcfcc3),
                           UINT64_C(0x000ff2fdf7fab37f),
                           UINT64_C(0x000cffffd6adff3c)}};
    const w256_t fix1 = {{UINT64_C(0x42e415394eb76145),
                          UINT64_C(0xb1a401c199de27fd),
                          UINT64_C(0x379b5a55f2680707),
                          UINT64_C(0x1788815e1a4a4cdd)}};
    const w256_t pcv1 = {{UINT64_C(0x0000000001000001),
                          UINT64_C(0x0000000000000000),
                          UINT64_C(0x7c32000000000000),
                          UINT64_C(0x4f259872bc33ab2d)}};
#endif // 19937


//#define CON(a, b) CON2(a, b)
//#define CON2(a, b) a ## b

#if HAVE_AVX2 && HAVE_IMMINTRIN_H
    const w256x32_t perm256 = {{7, 0, 1, 2, 3, 4, 5, 6}};
    const w256_t one256 = {{1, 1, 1, 1}};
    const w256xd_t m_one256 = {{-1.0, -1.0, -1.0, -1.0}};
    const w256xd_t two256 = {{2.0, 2.0, 2.0, 2.0}};
    const w256xd_t m_three256 = {{-3.0, -3.0, -3.0, -3.0}};
#endif
}
/* Local Variables:  */
/* mode: c++         */
/* End:              */
#endif // DSFMT_AVX_PARAMS_H
