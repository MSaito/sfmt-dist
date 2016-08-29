#include "sfmt-dist.h"
#include <sfmt-dist/dSFMTAVX607.h>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstring>

using MersenneTwister::DSFMTAVX607;
using std::cout;
using std::endl;

namespace {
    extern const char * id_string;
#if 0
    extern double double_data1[];
    extern double double_data2[];

    bool check_double(DSFMT19937& mt, int count, double data[])
    {
        for (int i = 0; i < count; i++) {
            if (fabs(mt.generateClose1Open2() - data[i]) > 0.000000000000001) {
                return false;
            }
        }
        return true;
    }
#endif
}

int main(void)
{
    //uint32_t init[4] = {1, 2, 3, 4};
    //int length = 4;
    DSFMTAVX607 mt;
    //int count = 1000;
    bool r = (strcmp(mt.getIDString().c_str(), id_string) == 0);
    if (!r) {
        cout << "check_ID failed" << endl;
        cout << "expected:" << id_string << endl;
        cout << "returned:" << mt.getIDString() << endl;
        return -1;
    }
#if 0
    mt.seed(0);
    r = check_double(mt, count, double_data1);
    if (!r) {
        cout << "check_double1 failed" << endl;
        return -1;
    }
    mt.seed(init, length);
    r = check_double(mt, count, double_data2);
    if (!r) {
        cout << "check_double2 failed" << endl;
        return -1;
    }
#endif
    r = mt.selfTest();
    if (!r) {
        cout << "self test failed" << endl;
        return -1;
    }
    return 0;
}

namespace {
    const char * id_string
    = "dSFMTAVX-607:1-19:"
        "f7fdedfdaf6f7-ff71b5676ff7f-fb66b7a1ef925-bfaffffff7faf";
#if 0
    double double_data1[1000] = {
    };
    double double_data2[1000] = {
    };
#endif
}
