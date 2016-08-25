#pragma once
#ifndef SFMT_DIST_DEBUG_H
#define SFMT_DIST_DEBUG_H

#if defined(DEBUG)
#include <iostream>
#include <iomanip>
#define DMSG(str) do { std::cout << (str) << std::endl;} while(0)
#else
#define DMSG(str)
#endif

#endif // SFMT_DIST_DEBUG_H
