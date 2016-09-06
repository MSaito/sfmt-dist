#pragma once
#ifndef MT_SFMT_COMMON_H
#define MT_SFMT_COMMON_H
#include <stdint.h>
#include "w256.h"

namespace MersenneTwister
{
    void sfmt_period_certification(w256_t * vec, const w256_t *pcv);
    void sfmt_seed(w256_t * state, int state_size, uint32_t seed);
    void sfmt_seed(w256_t * state, int state_size,
                   uint32_t init_key[], int key_length);
    void sfmt_initial_mask(w256_t * state, int size);
}
#endif // MT_SFMT_COMMON_H
