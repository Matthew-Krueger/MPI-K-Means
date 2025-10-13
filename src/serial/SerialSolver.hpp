//
// Created by Matthew Krueger on 10/13/25.
//

#ifndef KMEANS_MPI_SERIALSOLVER_HPP
#define KMEANS_MPI_SERIALSOLVER_HPP

#include "../shared/DataSet.hpp"

namespace kmeans {
    class SerialSolver {
    public:
        struct Config {
            size_t maxIterations;
            double convergenceThreshold;
            DataSet &&dataSet;
            size_t startingCentroidSeed;
            size_t startingCentroidCount;
        };

        struct PerCountCentroid {
            size_t count;
            Point centroid;
        };

        SerialSolver(Config &config);
        ~SerialSolver() = default;

        void run();

        inline DataSet& getDataSet() { return m_DataSet; }
        inline const DataSet& getDataSet() const { return m_DataSet; }
        inline const std::optional<std::vector<Point>>& getCalculatedCentroidsAtCompletion() const { return m_CalculatedCentroidsAtCompletion; }

    private:
        DataSet m_DataSet;
        std::vector<PerCountCentroid> m_CurrentCentroids;
        std::vector<PerCountCentroid> m_PreviousCentroids;
        size_t m_MaxIterations;
        double m_ConvergenceThreshold;
        std::optional<std::vector<Point>> m_CalculatedCentroidsAtCompletion = std::nullopt;


    };
} // kmeans

#endif //KMEANS_MPI_SERIALSOLVER_HPP