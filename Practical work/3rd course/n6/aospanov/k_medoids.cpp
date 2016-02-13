#include "k_medoids.h"

#define PI 3.14159265359

inline double L_sum(const double *points,
             const std::vector<int> &indxs,
             const int ind)
{
    double sum = 0;
    double r = PI / 180;
    for (int i : indxs) {
        sum += points[3 * i] * 
            acos(sin(r * points[3 * i + 1]) * sin(r * points[3 * ind + 1]) +
                 cos(r * points[3 * i + 1]) * cos(r * points[3 * ind + 1]) *
                 cos(r * (points[3 * i + 2] - points[3 * ind + 2])));
    }
    return sum;
}


inline std::vector<long> L_min(const double *points,
          const std::vector<std::pair<double, double>> &mju,
          const int points_number)
{
    double r = PI / 180;
    std::vector<long> min_indxs(points_number);
    for (int i = 0; i < points_number; i++) {
        double min = acos(sin(r * points[3 * i + 1]) * sin(r * mju[0].first) +
                          cos(r * points[3 * i + 1]) * cos(r * mju[0].first) *
                          cos(r * (points[3 * i + 2] - mju[0].second)));
        min_indxs[i] = 0;
        for (int j = 1; j < mju.size(); j++) {
            double t = acos(sin(r * points[3 * i + 1]) * sin(r * mju[j].first) +
                            cos(r * points[3 * i + 1]) * cos(r * mju[j].first) *
                            cos(r * (points[3 * i + 2] - mju[j].second)));
            if (t < min) {
                min_indxs[i] = j;
                min = t;
            }
        }
    }

    return min_indxs;
}


void k_medoids_cpp(const double *points, int points_number,
               int clusters_number, std::vector<long> &clusterization)
{
    float eps = 1e-5;
    int maxIterNum = 50;
    bool nonIter = true;

    int space = points_number / clusters_number;

    for (int i = 0, j = 0; j < points_number; j++) {
        if ((i + 1) * space < j + 1 && i < clusters_number - 1)
            i++;
        clusterization[j] = i;
    }
    auto engine = std::default_random_engine(unsigned(std::time(0)));
    std::shuffle(clusterization.begin(), clusterization.end(), engine);

    std::vector<std::pair<double, double>> tmp(clusters_number);
    std::vector<std::pair<double, double>> mju(clusters_number);
    int iterNum = 0;

    while (iterNum < maxIterNum) {
        for (auto cn : std::set<int>(clusterization.begin(), clusterization.end())) {
            std::vector<int> indxs;
            for (int i = 0; i < clusterization.size(); i++) {
                if (clusterization[i] == cn)
                    indxs.push_back(i);
            }
            if (indxs.size() == 1) {
                mju[cn] = std::make_pair(points[3 * indxs[0] + 1],
                                         points[3 * indxs[0] + 2]);
                break;
            }

            double s = L_sum(points, indxs, 0);
            int mid = indxs[0];
            for (int i = 1; i < indxs.size(); i++) {
                double t = L_sum(points, indxs, indxs[i]);
                if (t < s) {
                    mid = indxs[i];
                    s = t;
                }
            }
 
            mju[cn] = std::make_pair(points[3 * mid + 1], points[3 * mid + 2]);
        }

        clusterization = L_min(points, mju, points_number);

        iterNum += 1;
        if (nonIter) {
            int i = 0;
            for (; i < mju.size(); i++) {
                if (fabs(tmp[i].first - mju[i].first) > eps ||
                    fabs(tmp[i].second - mju[i].second) > eps)
                    break;
            }

            if (i == mju.size()) {
                break;
            }
        }

        tmp = mju;
    }
        
    // std::cout << "Iterations: " << iterNum;
}
