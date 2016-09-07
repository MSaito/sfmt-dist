#include "sfmt-dist.h"
#include <sfmt-dist/version.h>
#include <iostream>
#include <cstring>

int main()
{
    using namespace std;
    const char * v = MersenneTwister::get_sfmt_dist_version();
    cout << v << endl;
    if (strcmp(v, VERSION) == 0) {
        return 0;
    } else {
        return 1;
    }
}
