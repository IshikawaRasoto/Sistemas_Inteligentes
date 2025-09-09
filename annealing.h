#ifndef SALEMAN_ANNEALING_H
#define SALEMAN_ANNEALING_H

#include "map.h"

struct AnnealingParams {
    double initialTemp = 1000.0;
    double finalTemp = 1e-3;
    double alpha = 0.0;
    double actualTemp = initialTemp;
};

struct AnnealingState {
    Path bestPath;
    Path currentPath;
    AnnealingParams params;
    Problem problem;
    double bestDist = std::numeric_limits<double>::infinity();
    unsigned int iterations = 0;
    unsigned int currentIterations = 0;
};

inline Path twoOptSwap(Path &path, RNG &rng) {
    if (path.order.size() < 2) return path;

    size_t i = rng.randint(0, path.order.size() - 1);
    size_t j = rng.randint(0, path.order.size() - 1);
    if (i > j) std::swap(i, j);

    Path swap = path;
    std::reverse(swap.order.begin() + i, swap.order.begin() + j + 1);
    return swap;
}

inline bool runAnnealing(AnnealingState &state, RNG &rng) {
    if (state.params.actualTemp < state.params.finalTemp) {
        return false;
    }

    for (unsigned int i = 0; i < state.iterations; ++i) {
        Path candidate = twoOptSwap(state.currentPath, rng);
        const double current_dist = routeLength(state.currentPath.order, state.problem.distanceMatrix);
        const double candidate_dist = routeLength(candidate.order, state.problem.distanceMatrix);

        if (candidate_dist < current_dist) {
            state.currentPath = candidate;
        } else {
            const double delta = candidate_dist - current_dist;
            const double acceptance_prob = std::exp(-delta / state.params.actualTemp);
            if (rng.rand01() < acceptance_prob) {
                state.currentPath = candidate;
            }
        }

        double currentPathDist = routeLength(state.currentPath.order, state.problem.distanceMatrix);
        if (currentPathDist < state.bestDist) {
            state.bestDist = currentPathDist;
            state.bestPath = state.currentPath;
        }
    }
    state.currentIterations += state.iterations;
    state.params.actualTemp = state.params.actualTemp / (1 + state.params.alpha * state.params.actualTemp);

    return true;
}

#endif //SALEMAN_ANNEALING_H