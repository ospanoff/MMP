#include "fast_impl.h"

#include <cmath>

using namespace std;

template<typename T>
inline T Sqr(T x) {
    return x * x;
}

float CalcAvgDist(const vector<float>& px, const vector<float>& py)
{
    float sum = 0.f;
    for (size_t i = 0; i < px.size(); ++i) {
        for (size_t j = i + 1; j < px.size(); ++j) {
            sum += sqrt(Sqr(px[i] - px[j]) + Sqr(py[i] - py[j]));
        }
    }
    return 2 * sum / (px.size() * (px.size() - 1));
}


float CalcAvgDistPtr(const float* px, const float* py, int n) {
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            sum += sqrt(Sqr(px[i] - px[j]) + Sqr(py[i] - py[j]));
        }
    }
    return 2 * sum / (n * (n - 1));
}
