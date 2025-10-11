//
// Created by Matthew Krueger on 10/10/25.
//

#ifndef KMEANS_MPI_DATASET_HPP
#define KMEANS_MPI_DATASET_HPP
#include <vector>
#include <string>
#include <expected>
#include <random>

#include "Point.hpp"

namespace kmeans {
    class DataSet {
    public:
        struct Config {
            size_t numTotalSamples;
            size_t numDimensions;
            size_t numTrueClusters; // K_true, the number of blobs to generate
            double clusterSpread;   // Standard deviation for Gaussian noise within blobs
            long seed;              // Random seed for reproducibility
        };

        DataSet() = default;
        explicit DataSet(std::vector<Point> points) : m_Points(std::move(points)) {}
        explicit DataSet(const Config& config);

        inline std::expected<Point::FlattenedPoints, std::string> flattenDataset() { return Point::flattenPoints(m_Points); };
        inline static std::expected<DataSet, std::string> unflattenDataset(const Point::FlattenedPoints &flattenedPoints) {
            return Point::unflattenPoints(flattenedPoints).transform([](std::vector<Point>&& points) {
                return DataSet(std::move(points));
            });
        };

        using iterator = std::vector<Point>::iterator;
        using const_iterator = std::vector<Point>::const_iterator;

        inline iterator begin() { return m_Points.begin(); }
        [[nodiscard]] inline const_iterator begin() const { return m_Points.begin(); }
        inline iterator end() { return m_Points.end(); }
        [[nodiscard]] inline const_iterator end() const { return m_Points.end(); }

    private:
        inline void reserve(size_t size) { m_Points.reserve(size); }
        inline void emplace_back(const Point &toEmplace) { m_Points.emplace_back(toEmplace); }

        void generateCluster(const Point& clusterCenter, size_t numberPoints, double clusterSpread, std::mt19937 &rng);

        std::vector<Point> m_Points;


    };
} // kmeans

#endif //KMEANS_MPI_DATASET_HPP