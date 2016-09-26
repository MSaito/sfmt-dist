#include "sfmt-dist.h"
#include <inttypes.h>
#include <climits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <random>
#include <sfmt-dist/sfmt19937.h>
#include <sfmt-dist/dSFMT19937.h>
#include <sfmt-dist/mt19937.h>
#include <sfmt-dist/normalFromDouble.hpp>
#include <sfmt-dist/uniformIntFromDouble.hpp>
#include <sfmt-dist/Ziggurat.hpp>
#include <sfmt-dist/KISS.hpp>
#include "getopt.hpp"

using namespace std;
using namespace Ziggurat;

template<typename T, typename D, typename E>
int speed(T p1, T p2, E& engine, uint64_t count)
{
    D dist(p1, p2);
    T sum = 0;
    chrono::system_clock::time_point  start, end;
    start = chrono::system_clock::now();

    for (uint64_t i = 0; i < count; ++i) {
        sum = dist(engine);
    }
    end = chrono::system_clock::now();
    double elapsed = chrono::duration_cast
        <chrono::milliseconds>(end-start).count();
    cout << elapsed << "ms" << endl;
    uint32_t s = static_cast<uint32_t>(sum);
    return s & 1;
}

template<typename T, typename D>
int d_speed(T p1, T p2, uint32_t seed, uint64_t count)
{
    D dist(p1, p2, seed);
    T sum = 0;
    chrono::system_clock::time_point  start, end;
    start = chrono::system_clock::now();

    for (uint64_t i = 0; i < count; ++i) {
        sum = dist();
    }
    end = chrono::system_clock::now();
    double elapsed = chrono::duration_cast
        <chrono::milliseconds>(end-start).count();
    cout << elapsed << "ms" << endl;
    uint32_t s = static_cast<uint32_t>(sum);
    return s & 1;
}


int main(int argc, char * argv[])
{
    options opt;
    opt.count = 100000000;
    opt.seed = 1234;
    opt.start = 0;
    opt.end = 200;
    if (!opt.parse(argc, argv)) {
        return -1;
    }
    uint64_t count = opt.count;
    uint32_t seed = opt.seed;
    int32_t start = opt.start;
    int32_t end = opt.end;
    typedef uniform_int_distribution<uint32_t> unif;
    typedef normal_distribution<> norm;
    typedef MersenneTwister::UniformIntFromDouble
        <MersenneTwister::DSFMT19937> dunif;
    typedef MersenneTwister::NormalFromDouble
        <MersenneTwister::DSFMT19937> dnorm;
    std::mt19937 mt(seed);
    MersenneTwister::MT19937 MT(seed);
    MersenneTwister::SFMT19937 sfmt(seed);
    if (opt.uniform_dist) {
        cout << "#uniform distribution[" << start << ","
             << end << "] time for " << dec << count
             << " generation" << endl;
        switch (opt.generator_kind) {
        case 'm':
            cout << "std::mt19937,";
            speed<uint32_t, unif, std::mt19937>(start, end, mt, count);
            break;
        case 'M':
            cout << "MersenneTwister::MT19937,";
            speed<uint32_t, unif, MersenneTwister::MT19937>(start,
                                                            end, MT, count);
            break;
        case 'S':
            cout << "MersenneTwister::SFMT19937,";
            speed<uint32_t, unif, MersenneTwister::SFMT19937>(start, end,
                                                              sfmt, count);
            break;
        case 'd':
            cout << "MersenneTwister::dSFMT19937,";
            d_speed<uint32_t, dunif>(start, end, seed, count);
            break;
        default:
            break;
        }
    } else if (opt.normal_dist) {
        cout << "#normal distribution for " << dec << count
             << " generation" << endl;
        switch (opt.generator_kind) {
        case 'm':
            cout << "std::mt19937,";
            //std::mt19937 mt(seed);
            speed<double, norm, std::mt19937>(0.0, 1.0, mt, count);
            break;
        case 'M':
            cout << "MersenneTwister::mt19937,";
            //MersenneTwister::MT19937 mt(seed);
            speed<double, norm, MersenneTwister::MT19937>(0.0, 1.0, MT, count);
            break;
        case 'S':
            cout << "MersenneTwister::SFMT19937,";
            speed<double, norm, MersenneTwister::SFMT19937>(0.0, 1.0,
                                                            sfmt, count);
            break;
        case 'd':
            cout << "MersenneTwister::dSFMT19937,";
            d_speed<double, dnorm>(0.0, 1.0, seed, count);
            break;
        default:
            break;
        }
    } else if (opt.ziggurat) {
        cout << "#ziggurat normal distribution for " << dec << count
             << " generation" << endl;
        switch (opt.generator_kind) {
        case 'k':
            cout << "std::mt19937,";
            //std::mt19937 mt(seed);
            d_speed<float, ZigNormal<KISS> >(0.0, 1.0, seed, count);
            break;
        case 'm':
            cout << "std::mt19937,";
            //std::mt19937 mt(seed);
            d_speed<float, ZigNormal<std::mt19937> >(0.0, 1.0, seed, count);
            break;
        case 'M':
            cout << "MersenneTwister::mt19937,";
            //MersenneTwister::MT19937 mt(seed);
            d_speed<float, ZigNormal<MersenneTwister::MT19937> >(0.0,
                                                               1.0, seed, count);
            break;
        case 'S':
            cout << "MersenneTwister::SFMT19937,";
            d_speed<float, ZigNormal<MersenneTwister::SFMT19937> >(0.0, 1.0,
                                                                 seed, count);
            break;
        default:
            break;
        }
    }
    return 0;
}
