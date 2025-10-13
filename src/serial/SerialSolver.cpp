//
// Created by Matthew Krueger on 10/13/25.
//

#include "SerialSolver.hpp"

#include <algorithm>
#include <memory>
#include <ranges>
#include <unordered_set>
#include <__ranges/iota_view.h>
#include <__ranges/transform_view.h>

#include "../shared/Logging.hpp"

namespace kmeans {

    SerialSolver::SerialSolver(Config &config) {

        m_DataSet = config.dataSet;

        DEBUG_PRINT("Starting Centroid Count: " << config.startingCentroidCount);
        DEBUG_PRINT("m_DataSet Size: " << m_DataSet.size());

        // guard config size:
        if (config.startingCentroidCount > m_DataSet.size()) {
            throw std::invalid_argument("Cannot select more centroids than data points");
        }

        // generate starting centroids based on DataSet Properties

        // copy config appropriately.
        m_MaxIterations = config.maxIterations;
        m_ConvergenceThreshold = config.convergenceThreshold;
        m_CurrentCentroids = std::vector<Point>();

        size_t dimensionality = m_DataSet[0].numDimensions();
        size_t numCentroids = config.startingCentroidCount;
        size_t seed = config.startingCentroidSeed;

        // reserve space for centroids appropriately. No need to reserve previous as it'll get dumped anyway as soon as we start the run
        m_CurrentCentroids.reserve(config.startingCentroidCount);

        // now, we generate our centroids
        DEBUG_PRINT("Copied Solver Configs");

        // first, create our RNG
        std::mt19937 rng(seed);
        std::uniform_int_distribution<size_t> dist(0, m_DataSet.size() - 1);

        // instead of generating new centroids, we'll just randomly pull existing points BY COPY!
        // So, we'll use an unordered set. There's not a great functional way to do this, and the standard way makes much more sense.
        std::unordered_set<size_t> indices;
        while (indices.size() < numCentroids) {
            size_t index = dist(rng);
            indices.emplace(index);
        }


        std::ranges::transform(indices,
                               std::back_inserter(m_CurrentCentroids),
                               [&](const size_t index) { return Point(m_DataSet[index]); } // explicitly copy the data so we know *FOR SURE* it's unique.
        );

    }


    void SerialSolver::run() {

        // so the algorithm is roughly this
        // Calculate the *closest* centroid and class the point as this centroid
        // Then calculate the vector average of all the points
        // the vector average becomes the new

        std::cout << "Initial Centroids" << std::endl;
        for (auto &centroid : m_CurrentCentroids) {
            std::cout << centroid << std::endl;
        }


        size_t iteration = 0;
        while (iteration < m_MaxIterations) { // test if we have reached convergence or max samples
            DEBUG_PRINT("SerialSolver iteration " << iteration << " of " << m_MaxIterations);

            // in each iteration, we have to class the centroid, then accumulate the centroid to the new average. Generally speaking,
            // while this class -> reduction operation is two separate operations, in this case it may be advantageous to interleave these operations

            // Ergo, we will reserve a new array of current centroids and move the old one to previous
            m_PreviousCentroids = std::move(m_CurrentCentroids);

            // and zero the new one AND
            // since m_CurrentCentroids has counts, we can fill the vector with zero
            //m_CurrentCentroids = std::vector<Point>(m_PreviousCentroids.size(), Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0));

            // now that we have that, we can now accumulate
            // again, this uses move semantics to pass the *same* value back and forth,
            // so the accumulation is a zero cost abstraction that matches the reduction pattern more closely
            // than "just" a for loop
            m_CurrentCentroids = std::accumulate(
                m_DataSet.begin(),
                m_DataSet.end(),
                std::vector<Point>(m_PreviousCentroids.size(), Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0), 0)),
                [&](std::vector<Point> acc, Point &point) {
                    auto closestCentroidInPrevious = point.findClosestPointInVector(m_PreviousCentroids);

                    if (closestCentroidInPrevious != m_PreviousCentroids.end()) {
                        size_t centroidIndex = std::distance(m_PreviousCentroids.begin(), closestCentroidInPrevious);
                        acc[centroidIndex] += point;
                        acc[centroidIndex].setCount(acc[centroidIndex].getCount()+1);
                        return acc;
                    }else {
                        throw std::runtime_error("Centroid not found in previous centroids");
                    }
                }
            );

            // transform the m_CurrentCentroids by the scalar
            // so that we have the actual average
            // THERE's a SEGFAULT here
            std::ranges::for_each(m_CurrentCentroids, [](Point& centroid) {
                if (centroid.getCount() > 0) {
                    centroid /= static_cast<double>(centroid.getCount());
                }
                // If getCount() is 0, the centroid sum is already {0,0,...}, which is correct for an empty cluster.
            });

            // now we're done with an iteration.
            std::cout << "Iteration " << iteration << std::endl;
            for (auto &centroid : m_CurrentCentroids) {
                std::cout << centroid << std::endl;
            }

            iteration++;
        }

    }

} // kmeans