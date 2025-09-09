#ifndef SALEMAN_GENETIC_H
#define SALEMAN_GENETIC_H
#include <numeric>

#include "map.h"

struct GAParams {
    size_t populationSize = 1000;
    size_t generations = 5000;
    double mutationRate = 0.02;
    size_t tournamentK = 5;
    size_t elitism = 5;
    size_t stallLimit = 500;
};

inline void initPopulation(std::vector<Path> &pop, const size_t nCities, RNG &rng) {
  std::vector<uint16_t> base(nCities);
  std::iota(base.begin(), base.end(), 0);
  for (auto &path : pop) {
    path.order = base;
    for (size_t i = nCities - 1; i > 0; --i) {
      const size_t j = rng.randint(0, i);
      std::swap(path.order[i], path.order[j]);
    }
    path.dist = std::numeric_limits<double>::infinity();
  }
}

inline size_t tournamentSelect(const std::vector<Path> &pop, RNG &rng,
                               const size_t k) {
  size_t best = rng.randint(0, pop.size() - 1);
  for (size_t i = 1; i < k; ++i) {
    const size_t idx = rng.randint(0, pop.size() - 1);
    if (pop[idx].dist < pop[best].dist)
      best = idx;
  }
  return best;
}

inline void orderCrossover(const Path &p1, const Path &p2,
                           Path &child, RNG &rng) {
  const size_t n = p1.order.size();
  child.order.assign(n, std::numeric_limits<uint16_t>::max());
  size_t a = rng.randint(0, n - 1);
  size_t b = rng.randint(0, n - 1);
  if (a > b)
    std::swap(a, b);

  std::vector<char> taken(n, false);
  for (size_t i = a; i <= b; ++i) {
    const uint16_t gene = p1.order[i];
    child.order[i] = gene;
    taken[gene] = true;
  }

  size_t pos = (b + 1) % n;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t gene = p2.order[(b + 1 + i) % n];
    if (!taken[gene]) {
      child.order[pos] = gene;
      pos = (pos + 1) % n;
    }
  }
}

inline void mutateSwap(Path &ind, const double mutationRate, RNG &rng) {
  const size_t n = ind.order.size();
  for (size_t i = 0; i < n; ++i) {
    if (rng.rand01() < mutationRate) {
      const size_t j = rng.randint(0, n - 1);
      std::swap(ind.order[i], ind.order[j]);
    }
  }
}

inline void evaluate(std::vector<Path> &pop, const std::vector<double> &distM) {
  for (auto &path : pop) {
    path.dist = routeLength(path.order, distM);
  }
}

inline Path runGA(Problem &problem, const GAParams &cfg, RNG &rng) {
  const size_t n = problem.numCities();
  if (n < 3)
    throw std::runtime_error("Need at least 3 cities.");

  problem.distanceMatrix = buildDistanceMatrix(problem);

  std::vector<Path> pop(cfg.populationSize);
  initPopulation(pop, n, rng);
  evaluate(pop, problem.distanceMatrix);
  std::sort(pop.begin(), pop.end(),
            [](const auto &a, const auto &b) { return a.dist < b.dist; });

  Path best = pop.front();
  size_t stall = 0;

  std::vector<Path> next(pop.size());
  for (size_t gen = 0; gen < cfg.generations; ++gen) {
    for (size_t e = 0; e < cfg.elitism; ++e)
      next[e] = pop[e];

    for (size_t i = cfg.elitism; i < pop.size(); ++i) {
      const Path &p1 = pop[tournamentSelect(pop, rng, cfg.tournamentK)];
      const Path &p2 = pop[tournamentSelect(pop, rng, cfg.tournamentK)];
      orderCrossover(p1, p2, next[i], rng);
      mutateSwap(next[i], cfg.mutationRate, rng);
    }

    pop.swap(next);
    evaluate(pop, problem.distanceMatrix);
    std::sort(pop.begin(), pop.end(),
              [](const auto &a, const auto &b) { return a.dist < b.dist; });

    if (pop.front().dist + 1e-9 < best.dist) {
      best = pop.front();
      stall = 0;
    } else {
      if (++stall >= cfg.stallLimit)
        break;
    }
  }
  return best;
}

#endif //SALEMAN_GENETIC_H