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

using namespace std;

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
    uint64_t count = 100000000;
    uint32_t seed = 1234;
    uint32_t start = 0;
    uint32_t end = 200;
    if (argc <= 2) {
        cout << argv[0] << " [-n|-u] [-m|-M|-S|-d] [seed]" << endl;
        cout << "-n     normal distribution" << endl;
        cout << "-u     uniform distribution" << endl;
        cout << "-m     mersennetwister19937" << endl;
        cout << "-M     MersenneTwister19937" << endl;
        cout << "-S     SFMT19937" << endl;
        cout << "-d     dSFMT19937" << endl;
        cout << "seed   seed" << endl;
        return -1;
    }
    bool is_normal = false;
    bool is_uniform = true;
    if (argv[1][1] == 'n') {
        is_normal = true;
        is_uniform = false;
    } else if (argv[1][1] == 'u') {
        is_normal = false;
        is_uniform = true;
    }
    if (argc >= 4) {
        errno = 0;
        seed = strtoul(argv[3], NULL, 10);
        if (errno) {
            cout << "seed must be a number" << endl;
            return -1;
        }
    }
    typedef uniform_int_distribution<uint32_t> unif;
    typedef normal_distribution<> norm;
    typedef MersenneTwister::UniformIntFromDouble
        <uint32_t, MersenneTwister::DSFMT19937> dunif;
    typedef MersenneTwister::NormalFromDouble
        <MersenneTwister::DSFMT19937> dnorm;
    std::mt19937 mt(seed);
    MersenneTwister::MT19937 MT(seed);
    MersenneTwister::SFMT19937 sfmt(seed);
    //unif udist(start, end);
    //dunif dudist(start, end, seed);
    //norm ndist(0, 1.0);
    //dnorm dndist(0, 1.0, seed);
    if (is_uniform) {
        cout << "#uniform distribution time for " << dec << count
             << " generation" << endl;
        switch (argv[2][1]) {
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
    } else if (is_normal) {
        cout << "#normal distribution for " << dec << count
             << " generation" << endl;
        switch (argv[2][1]) {
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
    }
    return 0;
}
