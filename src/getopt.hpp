#pragma once
#ifndef GETOPT_HPP
#define GETOPT_HPP
#include <getopt.h>
#include <iostream>
#include <iomanip>
#include <string>

class options {
public:
    int count;
    bool normal_dist;
    bool uniform_dist;
    char generator_kind;
    uint32_t seed;
    int32_t start;
    int32_t end;
    options() {
        count = 0;
        normal_dist = false;
        uniform_dist = false;
        generator_kind = ' ';
        seed = (uint32_t)clock();
    }
    bool parse(int argc, char **argv) {
        using namespace std;
        int c;
        bool error = false;
        string pgm = argv[0];
        static struct option longopts[] = {
            {"normal", no_argument, NULL, 'n'},
            {"uniform", required_argument, NULL, 'u'},
            {"std-mt", optional_argument, NULL, 'm'},
            {"mt-mt", optional_argument, NULL, 'M'},
            {"sfmt", optional_argument, NULL, 'S'},
            {"dsfmt", optional_argument, NULL, 'd'},
            {"dsfmtavx", optional_argument, NULL, 'a'},
            {"seed", required_argument, NULL, 's'},
            {"count", required_argument, NULL, 'c'},
            {"start", required_argument, NULL, 'f'},
            {"end", required_argument, NULL, 'l'},
            {NULL, 0, NULL, 0}};
        errno = 0;
        for (;;) {
            c = getopt_long(argc, argv, "numMSdas:c:f:l:", longopts, NULL);
            if (error) {
                break;
            }
            if (c == -1) {
                break;
            }
            switch (c) {
            case 's':
                seed = strtoul(optarg, NULL, 0);
                if (errno) {
                    error = true;
                    cerr << "seed must be a number" << endl;
                }
                break;
            case 'c':
                count = strtol(optarg, NULL, 10);
                if (errno) {
                    error = true;
                    cerr << "count must be a number" << endl;
                }
                break;
            case 'f':
                start = strtol(optarg, NULL, 10);
                if (errno) {
                    error = true;
                    cerr << "start must be a number" << endl;
                }
                break;
            case 'l':
                end = strtol(optarg, NULL, 10);
                if (errno) {
                    error = true;
                    cerr << "end must be a number" << endl;
                }
                break;
            case 'u':
                uniform_dist = true;
                break;
            case 'n':
                normal_dist = true;
                break;
            case 'm':
            case 'M':
            case 'S':
            case 'd':
            case 'a':
                generator_kind = c;
                break;
            case '?':
            default:
                error = true;
                break;
            }
        }
        if (error) {
            output_help(pgm);
            return false;
        }
        if (!(normal_dist || uniform_dist)) {
            output_help(pgm);
            return false;
        }
        if (generator_kind == ' ') {
            output_help(pgm);
            return false;
        }
        return true;
    }

    void output_help(std::string& pgm) {
        using namespace std;
        cout << pgm << " [-n|-u] [-m|-M|-S|-d] [-s seed] [-c count]"
             << " [-f start] [-l end]" << endl;
        cout << "-n --normal   normal distribution" << endl;
        cout << "-u --uniform  uniform distribution" << endl;
        cout << "-m --std-mt   std::mersennetwister19937" << endl;
        cout << "-M --mt-mt    MersenneTwister19937" << endl;
        cout << "-S --sfmt     SFMT19937" << endl;
        cout << "-d --dsfmt    dSFMT19937" << endl;
        cout << "-a --dsfmtavx dSFMTAVX607" << endl;
        cout << "-s --seed     seed" << endl;
        cout << "-c --count    count" << endl;
        cout << "-f --start    start of uniform range" << endl;
        cout << "-l --end      end of uniform range" << endl;
    }
};
#endif // GETOPT_HPP
