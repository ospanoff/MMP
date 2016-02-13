#include <iostream>

#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>

#include <cstdlib>
#include <ctime>
#include <cmath>


void k_medoids_cpp(const double *points, int points_number,
                   int clusters_number, std::vector<long> &clusterization);