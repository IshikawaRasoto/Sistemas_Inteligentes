#ifndef SALEMAN_MAP_H
#define SALEMAN_MAP_H
#include <chrono>
#include <vector>
#include <cmath>
#include <random>

struct Map {
    unsigned int width{0}, height{0};
};

struct City {
    unsigned int x{0}, y{0};
    uint16_t tag{0};
};

struct Problem {
    std::vector<City> cities;
    Map map;
    [[nodiscard]] size_t numCities() const noexcept { return cities.size(); }
    std::vector<double> distanceMatrix;
};

struct Path {
    std::vector<uint16_t> order;
    double dist = std::numeric_limits<double>::infinity();
};

struct RNG {
    std::mt19937_64 eng;
    std::uniform_real_distribution<double> real01{0.0, 1.0};

    explicit RNG(const uint64_t seed = std::random_device{}() ^
                                 (uint64_t)
                                     std::chrono::high_resolution_clock::now()
                                         .time_since_epoch()
                                         .count())
        : eng(seed) {}

    size_t randint(const size_t lo, const size_t hi) {
        std::uniform_int_distribution<size_t> d(lo, hi);
        return d(eng);
    }
    double rand01() { return real01(eng); }
};

static double euclid(const unsigned int ax, const unsigned int ay, const unsigned int bx,
                     const unsigned int by) noexcept {
    return std::hypot(static_cast<double>(ax) - static_cast<double>(bx),
                      static_cast<double>(ay) - static_cast<double>(by));
}

inline std::vector<double> buildDistanceMatrix(const Problem &p) {
    const size_t n = p.numCities();
    std::vector<double> m(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        m[i * n + i] = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double d =
                euclid(p.cities[i].x, p.cities[i].y, p.cities[j].x, p.cities[j].y);
            m[i * n + j] = d;
            m[j * n + i] = d;
        }
    }
    return m;
}

static double routeLength(const std::vector<uint16_t> &order,
                          const std::vector<double> &distM) noexcept {
    const size_t n = order.size();
    double acc = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        acc += distM[static_cast<size_t>(order[i]) * n + order[i + 1]];
    }
    acc += distM[static_cast<size_t>(order.back()) * n + order.front()];
    return acc;
}

inline void initializeMap(Map &map, const unsigned int width, const unsigned int height) {
    map.width = width;
    map.height = height;
}

inline void populateCities(Problem &problem, RNG &rng, const Map &map,
                           const unsigned int numCities) {
    problem.cities.clear();
    problem.cities.reserve(numCities);
    for (unsigned int i = 0; i < numCities; ++i) {
        City c;
        c.x = static_cast<unsigned int>(rng.randint(0, map.width - 1));
        c.y = static_cast<unsigned int>(rng.randint(0, map.height - 1));
        c.tag = static_cast<uint16_t>(i);
        problem.cities.push_back(c);
    }
}

#endif //SALEMAN_MAP_H