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

namespace kmeans {

    SerialSolver::SerialSolver(Config &config) {

        // guard config size:
        if (config.startingCentroidCount > m_DataSet.size()) {
            throw std::invalid_argument("Cannot select more centroids than data points");
        }

        // generate starting centroids based on DataSet Properties

        // copy config appropriately.
        m_MaxIterations = config.maxIterations;
        m_ConvergenceThreshold = config.convergenceThreshold;
        m_DataSet = std::move(config.dataSet);
        m_PreviousCentroids = std::vector<PerCountCentroid>();
        m_CurrentCentroids = std::vector<PerCountCentroid>();

        size_t dimensionality = m_DataSet[0].numDimensions();
        size_t numCentroids = config.startingCentroidCount;
        size_t seed = config.startingCentroidSeed;

        // reserve space for centroids appropriately. No need to reserve previous as it'll get dumped anyway as soon as we start the run
        m_CurrentCentroids.reserve(config.startingCentroidCount);

        // now, we generate our centroids

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
                               [&](const size_t index) { return PerCountCentroid{1, Point(m_DataSet[index])}; } // explicitly copy the data so we know *FOR SURE* it's unique.
        );

    }


    void SerialSolver::run() {

        // so the algorithm is roughly this
        // Calculate the *closest* centroid and class the point as this centroid
        // Then calculate the vector average of all the points
        // the vector average becomes the new

        size_t iteration = 0;
        while (iteration < m_MaxIterations) { // test if we have reached convergence or max samples

            // in each iteration, we have to class the centroid, then accumulate the centroid to the new average. Generally speaking,
            // while this class -> reduction operation is two separate operations, in this case it may be advantageous to interleave these operations

            // Ergo, we will reserve a new array of current centroids and move the old one to previous
            m_PreviousCentroids = std::move(m_CurrentCentroids);

            // and zero the new one AND
            // since m_CurrentCentroids has counts, we can fill the vector with zero
            m_CurrentCentroids = std::vector<PerCountCentroid>(m_PreviousCentroids.size(), {0, Point(std::vector<double>(m_DataSet[0].numDimensions(), 0.0))});

            // now that we have that, we can now accumulate



            iteration++;
        }

    }

} // kmeans