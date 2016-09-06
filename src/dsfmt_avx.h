#pragma once
#ifndef MT_DSFMT_AVX_H
#define MT_DSFMT_AVX_H

#include <stdint.h>
#include <string>
#include <sfmt-dist/cpu_feature.h>
#include "w256.h"

namespace MersenneTwister
{
    namespace dSFMT_AVX {
        struct params {
            int mexp;
            int array_size;
            int pos1;
            w256_t mask;
            w256_t fix;
            w256_t pcv;
        };

        params * search_params(int mexp);
        std::string get_id_string(const params& params);
        void initial_mask(w256_t * state, int state_size);
        void fill_state(const cpu_feature_t& cf,
                        const params& params,
                        double * state);
        void fill_array(const cpu_feature_t& cf,
                        const params& params,
                        double * state,
                        double * array,
                        int length);
        void fill_arrayc1o2(const cpu_feature_t& cf,
                            const params& params,
                            double * state,
                            double * array,
                            int length);
        void fill_arrayo0c1(const cpu_feature_t& cf,
                            const params& params,
                            double * state,
                            double * array,
                            int length);
        void fill_arrayo0o1(const cpu_feature_t& cf,
                            const params& params,
                            double * state,
                            double * array,
                            int length);
        void fillarray_maxint(const cpu_feature_t& cf,
                              const params& params,
                              double * state,
                              int32_t * array,
                              uint64_t rmax,
                              int32_t min);
        void fillarray_normaldist(const cpu_feature_t& cf,
                                  const params& params,
                                  double * state,
                                  double * array,
                                  double mu,
                                  double sigma);
        bool self_test(const cpu_feature_t& cf, const params& params,
                       double * state);
        void fill_state256_607(double * state);
        void fill_state64_607(double * state);
        void fill_array256_c0o1_607(double * state, double * array, int length);
        void fill_array64_c0o1_607(double * state, double * array, int length);
        void fill_array256_c1o2_607(double * state, double * array, int length);
        void fill_array64_c1o2_607(double * state, double * array, int length);
        void fill_array256_o0c1_607(double * state, double * array, int length);
        void fill_array64_o0c1_607(double * state, double * array, int length);
        void fill_array256_o0o1_607(double * state, double * array, int length);
        void fill_array64_o0o1_607(double * state, double * array, int length);
        void fill_state256_1279(double * state);
        void fill_state64_1279(double * state);
        void fill_array256_c0o1_1279(double * state,
                                     double * array, int length);
        void fill_array64_c0o1_1279(double * state, double * array, int length);
        void fill_array256_c1o2_1279(double * state,
                                     double * array, int length);
        void fill_array64_c1o2_1279(double * state, double * array, int length);
        void fill_array256_o0c1_1279(double * state,
                                     double * array, int length);
        void fill_array64_o0c1_1279(double * state, double * array, int length);
        void fill_array256_o0o1_1279(double * state,
                                     double * array, int length);
        void fill_array64_o0o1_1279(double * state,
                                    double * array, int length);
        void fill_state256_2281(double * state);
        void fill_state64_2281(double * state);
        void fill_array256_c0o1_2281(double * state,
                                     double * array, int length);
        void fill_array64_c0o1_2281(double * state,
                                    double * array, int length);
        void fill_array256_c1o2_2281(double * state,
                                     double * array, int length);
        void fill_array64_c1o2_2281(double * state,
                                    double * array, int length);
        void fill_array256_o0c1_2281(double * state,
                                     double * array, int length);
        void fill_array64_o0c1_2281(double * state,
                                    double * array, int length);
        void fill_array256_o0o1_2281(double * state,
                                     double * array, int length);
        void fill_array64_o0o1_2281(double * state,
                                    double * array, int length);
        void fill_state256_4253(double * state);
        void fill_state64_4253(double * state);
        void fill_array256_c0o1_4253(double * state,
                                     double * array, int length);
        void fill_array64_c0o1_4253(double * state,
                                    double * array, int length);
        void fill_array256_c1o2_4253(double * state,
                                     double * array, int length);
        void fill_array64_c1o2_4253(double * state,
                                    double * array, int length);
        void fill_array256_o0c1_4253(double * state,
                                     double * array, int length);
        void fill_array64_o0c1_4253(double * state,
                                    double * array, int length);
        void fill_array256_o0o1_4253(double * state,
                                     double * array, int length);
        void fill_array64_o0o1_4253(double * state,
                                    double * array, int length);
        void fill_state256_19937(double * state);
        void fill_state64_19937(double * state);
        void fill_array256_c0o1_19937(double * state,
                                     double * array, int length);
        void fill_array64_c0o1_19937(double * state,
                                    double * array, int length);
        void fill_array256_c1o2_19937(double * state,
                                     double * array, int length);
        void fill_array64_c1o2_19937(double * state,
                                    double * array, int length);
        void fill_array256_o0c1_19937(double * state,
                                     double * array, int length);
        void fill_array64_o0c1_19937(double * state,
                                    double * array, int length);
        void fill_array256_o0o1_19937(double * state,
                                     double * array, int length);
        void fill_array64_o0o1_19937(double * state,
                                     double * array, int length);
        void fillarray256_maxint_19937(double * state, int32_t * array,
                                       uint64_t rmax, int32_t min);
        void fillarray64_maxint_19937(double * state, int32_t * array,
                                      uint64_t rmax, int32_t min);
        int fillarray256_boxmuller_19937(double * state, double * array,
                                       double mu, double sigma);
        int fillarray64_boxmuller_19937(double * state, double * array,
                                         double mu, double sigma);
        bool self_test_19937(const cpu_feature_t& cf, const params& params,
            double * state);
    }
}

#endif //
