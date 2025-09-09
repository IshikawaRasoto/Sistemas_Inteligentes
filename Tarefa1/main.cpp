#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <raylib.h>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "map.h"
#include "genetic.h"
#include "annealing.h"
#include "logger.h"

#define NUM_CITIES 250

class AlgorithmVisualization
{
private:
    Logger logger;
    unsigned int loggerCounter = 0;

    Problem problem;
    RNG gaRng;
    RNG saRng;

    std::vector<Path> population;
    GAParams gaParams;
    Path gaBestPath;
    size_t generation = 0;
    size_t stallCounter = 0;
    bool gaFinished = false;

    AnnealingState saState;
    bool saFinished = false;

    std::thread saThread;
    std::thread gaThread;
    mutable std::mutex saMutex;
    mutable std::mutex gaMutex;
    std::condition_variable cv;
    std::mutex cvMutex;
    std::atomic<bool> running{ false };
    bool stepSignal = false;

    int screenWidth, screenHeight;
    int mapX, mapY, mapW, mapH;
    bool showGA = true;
    bool showSA = true;

    [[nodiscard]] std::vector<City> PathToCity(const std::vector<uint16_t>& order) const {
        std::vector<City> result;
        result.reserve(order.size());
        for (const uint16_t idx : order)
        {
            result.push_back(problem.cities[idx]);
        }
        return result;
    }

public:
    AlgorithmVisualization(const int width, const int height)
        : logger("tsp_comparison0.csv"), gaRng(std::random_device{}()), saRng(std::random_device{}()), screenWidth(width),
        screenHeight(height) {
        mapX = 50;
        mapY = 50;
        mapW = (screenWidth - 100) / 2 - 50;
        mapH = screenHeight - 300;

        InitializeCities(NUM_CITIES);
        InitializeAlgorithms();
        StartThreads();
    }

    ~AlgorithmVisualization()
    {
        StopThreads();
    }

    void InitializeCities(const int numCities)
    {
        initializeMap(problem.map, mapW, mapH);
        populateCities(problem, gaRng, problem.map, numCities);

        for (auto& city : problem.cities)
        {
            city.x = mapX + 20 + (city.x % (mapW - 40));
            city.y = mapY + 20 + (city.y % (mapH - 40));
        }
    }

    void InitializeAlgorithms()
    {
        std::lock(gaMutex, saMutex);
        std::lock_guard<std::mutex> lg1(gaMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lg2(saMutex, std::adopt_lock);

        problem.distanceMatrix = buildDistanceMatrix(problem);

        gaParams.populationSize = NUM_CITIES * 10;
        gaParams.generations = NUM_CITIES * 500;
        gaParams.elitism = static_cast<int>(static_cast<double>(gaParams.populationSize) * 0.03f);
        gaParams.tournamentK = std::max(2, static_cast<int>(static_cast<double>(gaParams.populationSize) * 0.001f));
        gaParams.mutationRate = 0.1;
        gaParams.stallLimit = gaParams.generations / 10;

        population.resize(gaParams.populationSize);
        initPopulation(population, problem.numCities(), gaRng);
        evaluate(population, problem.distanceMatrix);
        std::sort(population.begin(), population.end(),
            [](const auto& a, const auto& b) { return a.dist < b.dist; });

        gaBestPath = population[0];
        generation = 0;
        stallCounter = 0;
        gaFinished = false;

        saState.problem = problem;
        saState.params.initialTemp = 1000.0;
        saState.params.finalTemp = 1e-3;
        saState.params.alpha = 1.0 / (0.2 * NUM_CITIES);
        saState.params.actualTemp = saState.params.initialTemp;
        saState.iterations = 5;

        saState.currentPath.order.resize(NUM_CITIES);
        std::iota(saState.currentPath.order.begin(), saState.currentPath.order.end(), 0);
        std::shuffle(saState.currentPath.order.begin(), saState.currentPath.order.end(), saRng.eng);
        saState.currentPath.dist = routeLength(saState.currentPath.order, problem.distanceMatrix);
        saState.bestPath = saState.currentPath;
        saState.bestDist = saState.currentPath.dist;
        saFinished = false;
    }

    void StartThreads()
    {
        StopThreads();
        running = true;

        saThread = std::thread([this]() {
            while (running)
            {
                std::unique_lock<std::mutex> lk(cvMutex);
                cv.wait(lk, [this]() { return stepSignal || !running; });
                if (!running) break;
                stepSignal = false;

                {
                    std::lock_guard<std::mutex> lk(saMutex);
                    if (!saFinished) StepSA();
                }
            }
            });

        gaThread = std::thread([this]() {
            while (running)
            {
                std::unique_lock<std::mutex> lk(cvMutex);
                cv.wait(lk, [this]() { return stepSignal || !running; });
                if (!running) break;
                stepSignal = false;

                {
                    std::lock_guard<std::mutex> lk(gaMutex);
                    if (!gaFinished) StepGA();
                }
            }
            });
    }

    void StopThreads()
    {
        running = false;
        cv.notify_all();
        if (saThread.joinable()) saThread.join();
        if (gaThread.joinable()) gaThread.join();
    }

    void StepSA()
    {
        if (saFinished || saState.params.actualTemp <= saState.params.finalTemp)
        {
            saFinished = true;
            return;
        }

        if (!runAnnealing(saState, saRng))
        {
            saFinished = true;
            return;
        }

        logger.AddSAValue(saState.currentIterations, saState.bestDist);
    }

    void StepGA()
    {
        if (gaFinished || generation >= gaParams.generations || stallCounter >= gaParams.stallLimit)
        {
            gaFinished = true;
            return;
        }

        std::vector<Path> nextPop(population.size());

        for (size_t e = 0; e < gaParams.elitism; ++e)
            nextPop[e] = population[e];

        for (size_t i = gaParams.elitism; i < population.size(); ++i)
        {
            const Path& p1 = population[tournamentSelect(population, gaRng, gaParams.tournamentK)];
            const Path& p2 = population[tournamentSelect(population, gaRng, gaParams.tournamentK)];
            orderCrossover(p1, p2, nextPop[i], gaRng);
            mutateSwap(nextPop[i], gaParams.mutationRate, gaRng);
        }

        population.swap(nextPop);
        evaluate(population, problem.distanceMatrix);
        std::sort(population.begin(), population.end(),
            [](const auto& a, const auto& b) { return a.dist < b.dist; });

        if (population[0].dist + 1e-9 < gaBestPath.dist)
        {
            gaBestPath = population[0];
            stallCounter = 0;
        }
        else
        {
            stallCounter++;
        }

        generation++;
        logger.AddGAValue(generation, gaBestPath.dist);
    }

    static void DrawPath(const std::vector<City>& path, const Color c, const float thick, const int offsetX = 0)
    {
        for (size_t i = 0; i < path.size(); i++)
        {
            const City& a = path[i];
            const City& b = path[(i + 1) % path.size()];
            DrawLineEx({ static_cast<float>(a.x + offsetX), static_cast<float>(a.y) },
                { static_cast<float>(b.x + offsetX), static_cast<float>(b.y) }, thick, c);
        }
    }

    static void DrawCities(const std::vector<City>& cities, const int offsetX = 0)
    {
        for (const auto& c : cities)
        {
            DrawCircle(c.x + offsetX, c.y, 8, DARKBLUE);
            DrawCircle(c.x + offsetX, c.y, 6, SKYBLUE);
            std::string s = std::to_string(c.tag);
            const int tw = MeasureText(s.c_str(), 10);
            DrawText(s.c_str(), c.x + offsetX - tw / 2, c.y - 5, 10, WHITE);
        }
    }

    void DrawUI() const {
        int uiY = screenHeight - 240;
        int uiHeight = 220;
        int leftX = 20;
        int rightX = screenWidth / 2 + 20;
        int uiWidth = screenWidth / 2 - 40;

        DrawRectangle(leftX, uiY, uiWidth, uiHeight, Fade(BLACK, 0.85f));
        DrawRectangleLines(leftX, uiY, uiWidth, uiHeight, RAYWHITE);
        DrawText("Simulated Annealing", leftX + 12, uiY + 8, 20, YELLOW);

        std::ostringstream tss;
        tss << "Temperature: " << std::scientific << std::setprecision(2) << saState.params.actualTemp;
        DrawText(tss.str().c_str(), leftX + 12, uiY + 35, 12, YELLOW);

        std::ostringstream bss;
        bss << "Best Distance: " << std::fixed << std::setprecision(1) << saState.bestDist;
        DrawText(bss.str().c_str(), leftX + 12, uiY + 55, 12, GREEN);

        std::ostringstream css;
        css << "Current Distance: " << std::fixed << std::setprecision(1) << saState.currentPath.dist;
        DrawText(css.str().c_str(), leftX + 12, uiY + 75, 12, ORANGE);

        std::ostringstream its;
        its << "Iterations: " << saState.currentIterations;
        DrawText(its.str().c_str(), leftX + 12, uiY + 95, 12, LIGHTGRAY);

        if (saFinished)
        {
            DrawText("FINISHED", leftX + 12, uiY + 115, 14, GREEN);
        }

        DrawRectangle(rightX, uiY, uiWidth, uiHeight, Fade(BLACK, 0.85f));
        DrawRectangleLines(rightX, uiY, uiWidth, uiHeight, RAYWHITE);
        DrawText("Genetic Algorithm", rightX + 12, uiY + 8, 20, LIME);

        std::ostringstream gss;
        gss << "Generation: " << generation;
        DrawText(gss.str().c_str(), rightX + 12, uiY + 35, 12, LIME);

        std::ostringstream gbss;
        gbss << "Best Distance: " << std::fixed << std::setprecision(1) << gaBestPath.dist;
        DrawText(gbss.str().c_str(), rightX + 12, uiY + 55, 12, GREEN);

        std::ostringstream gcss;
        gcss << "Current Distance: " << std::fixed << std::setprecision(1) << population[0].dist;
        DrawText(gcss.str().c_str(), rightX + 12, uiY + 75, 12, ORANGE);

        std::ostringstream stalls;
        stalls << "Stall Counter: " << stallCounter;
        DrawText(stalls.str().c_str(), rightX + 12, uiY + 95, 12, LIGHTGRAY);

        if (gaFinished)
        {
            DrawText("FINISHED", rightX + 12, uiY + 115, 14, GREEN);
        }

        int compY = uiY + 140;
        DrawText("Comparison:", leftX, compY, 16, WHITE);

        if (saState.bestDist < gaBestPath.dist)
        {
            DrawText("SA is winning!", leftX, compY + 25, 14, YELLOW);
        }
        else if (gaBestPath.dist < saState.bestDist)
        {
            DrawText("GA is winning!", leftX, compY + 25, 14, LIME);
        }
        else
        {
            DrawText("Tie!", leftX, compY + 25, 14, WHITE);
        }

        std::ostringstream diffs;
        diffs << "Difference: " << std::fixed << std::setprecision(1)
            << std::abs(saState.bestDist - gaBestPath.dist);
        DrawText(diffs.str().c_str(), leftX, compY + 45, 12, GRAY);
    }

    void Update()
    {
        if (IsKeyPressed(KEY_R))
        {
            StopThreads();
            // InitializeCities(NUM_CITIES);
            InitializeAlgorithms();
            loggerCounter++;
            logger = Logger("tsp_comparison" + std::to_string(loggerCounter) + ".csv");
            StartThreads();
        }

        if (IsKeyPressed(KEY_ONE))
        {
            showSA = !showSA;
        }

        if (IsKeyPressed(KEY_TWO))
        {
            showGA = !showGA;
        }

    }

    void Draw() const {
        BeginDrawing();
        ClearBackground(Color{ 15, 20, 35, 255 });

        const int centerX = screenWidth / 2;
        DrawLine(centerX, 0, centerX, screenHeight - 250, GRAY);

        if (showSA)
        {
            DrawText("Simulated Annealing", mapX, 20, 20, YELLOW);
            std::vector<City> saCurrentPathCities;
            std::vector<City> saBestPathCities;
            {
                std::lock_guard<std::mutex> lk(saMutex);
                saCurrentPathCities = PathToCity(saState.currentPath.order);
                saBestPathCities = PathToCity(saState.bestPath.order);
            }
            DrawPath(saCurrentPathCities, ORANGE, 2.0f);
            DrawPath(saBestPathCities, RED, 3.0f);
            DrawCities(saBestPathCities);
        }

        if (showGA)
        {
            const int gaOffsetX = centerX;
            DrawText("Genetic Algorithm", mapX + gaOffsetX, 20, 20, LIME);
            std::vector<City> gaCurrentPathCities;
            std::vector<City> gaBestPathCities;
            {
                std::lock_guard<std::mutex> lg(gaMutex);
                if (!population.empty()) gaCurrentPathCities = PathToCity(population[0].order);
                gaBestPathCities = PathToCity(gaBestPath.order);
            }
            DrawPath(gaCurrentPathCities, LIGHTGRAY, 2.0f, gaOffsetX);
            DrawPath(gaBestPathCities, GREEN, 3.0f, gaOffsetX);
            DrawCities(gaBestPathCities, gaOffsetX);
        }

        DrawUI();

        DrawText("Press R to restart, 1 to toggle SA, 2 to toggle GA", 20, screenHeight - 20, 12, WHITE);

        EndDrawing();
    }

    void Run()
    {
        while (!WindowShouldClose())
        {
            {
                std::lock_guard<std::mutex> lk(cvMutex);
                stepSignal = true;
            }
            cv.notify_all();

            Update();
            Draw();
        }
    }
};

int main()
{
    constexpr int screenWidth = 1680;
    constexpr int screenHeight = 720;

    InitWindow(screenWidth, screenHeight, "TSP: Simulated Annealing vs Genetic Algorithm");

    AlgorithmVisualization app(screenWidth, screenHeight);
    app.Run();

    CloseWindow();
    return 0;
}

