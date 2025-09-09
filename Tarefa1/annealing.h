#ifndef SALEMAN_ANNEALING_H
#define SALEMAN_ANNEALING_H

#include "map.h"

struct AnnealingParams {
    double initialTemp = 1000.0;
    double finalTemp = 1e-3;
    double alpha = 0.0;
    double actualTemp = initialTemp;
    unsigned int neighborsPerTemp = 10;
    unsigned int stallLimit = 500;
};

struct AnnealingState {
    Path bestPath;
    Path currentPath;
    AnnealingParams params;
    Problem problem;
    double bestDist = std::numeric_limits<double>::infinity();
    unsigned int iterations = 0;
    unsigned int currentIterations = 0;
	unsigned int stallCounter = 0;
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

inline double routePathLength(const Path& path, const Problem& problem) {
    return routeLength(path.order, problem.distanceMatrix);
}

inline bool runAnnealing(AnnealingState& state, RNG& rng) {
    if (state.params.actualTemp < state.params.finalTemp ||
        state.stallCounter >= state.params.stallLimit) {
        return false; 
    }

    for (unsigned int i = 0; i < state.params.neighborsPerTemp; ++i) {
        Path candidate = twoOptSwap(state.currentPath, rng);

        double current_dist = routePathLength(state.currentPath, state.problem);
        double candidate_dist = routePathLength(candidate, state.problem);

        if (candidate_dist < current_dist) {
            state.currentPath = candidate;
        }
        else {
            const double delta = candidate_dist - current_dist;
            const double acceptance_prob = std::exp(-delta / state.params.actualTemp);
            if (rng.rand01() < acceptance_prob) {
                state.currentPath = candidate;
            }
        }

        double currLen = routePathLength(state.currentPath, state.problem);
        if (currLen < state.bestDist) {
            state.bestDist = currLen;
            state.bestPath = state.currentPath;
            state.stallCounter = 0; // resetar se melhorou
        }
        else {
            state.stallCounter++;
        }

        state.iterations++;
        state.currentIterations++;
    }

    // resfriamento da temperatura
    state.params.actualTemp =
        state.params.actualTemp / (1.0 + state.params.alpha * state.params.actualTemp);

    return !(state.params.actualTemp < state.params.finalTemp ||
        state.stallCounter >= state.params.stallLimit);
}

#endif //SALEMAN_ANNEALING_H