//
// Created by mpiuser on 10/10/25.
// This file defines the Point class, which represents a data point in a multi-dimensional space.

#ifndef KMEANS_MPI_POINT_HPP
#define KMEANS_MPI_POINT_HPP
#include <vector>
#include <cstddef>
#include <expected>
#include <string>

namespace kmeans{
    class Point{
    public:
        /**
         * @brief A structure to hold flattened point data for efficient serialization/deserialization.
         *
         * This structure is used to represent a collection of points as a single contiguous
         * vector of doubles, along with metadata about the original dimensions and number of points.
         */
        struct FlattenedPoints
        {
            /// The number of dimensions for each point.
            size_t numDimensionsPerPoint;
            /// The total number of points.
            size_t numPoints;
            /// A flattened vector containing all the double values of all points.
            /// The data for each point is stored contiguously.
            std::vector<double> points;
        };

        Point() = default;

        Point(std::vector<double> data) noexcept: m_Data(std::move(data)){}

        /**
         * @brief Copy constructor.
         * @param other The Point object to copy from.
         */
        Point(const Point& other) : m_Data{other.m_Data}{}

        /**
         * @brief Move constructor.
         * @param other The Point object to move from.
         */
        Point(Point&& other) noexcept : m_Data{std::move(other.m_Data)}{}

        /**
         * @brief Copy-and-swap assignment operator.
         * @param other The Point object to assign from.
         * @return A reference to the assigned Point object.
         */
        Point& operator=(Point other){
            using std::swap;
            swap(*this, other);
            return *this;
        }

        /**
         * @brief Default destructor.
         */
        ~Point() = default;

        /**
         * @brief Gets the underlying data vector of the point.
         * @return A const reference to the internal std::vector<double> representing the point's coordinates.
         */
        const std::vector<double>& getData() { return m_Data; };

        /**
         * @brief Sets the underlying data vector of the point.
         * @param data A std::vector<double> to set as the point's coordinates.
         */
        void setData(std::vector<double> data) { m_Data = std::move(data); };

        /**
         * @brief Calculates the Euclidean distance between this point and another point.
         * @param other The other point to calculate the distance to.
         * @return An expected value containing the Euclidean distance as a double, or a string error message if dimensions do not match.
         */
        std::expected<double, std::string> calculateEuclideanDistance(const Point& other);

        static std::expected<FlattenedPoints, std::string> flattenPoints(const std::vector<Point>& points);
        static std::expected<std::vector<Point>, std::string> unflattenPoints(const FlattenedPoints &flattenedPoints);


        // Iterator support to allow for range-based for loops over the data, among other syntax magic
        using iterator = std::vector<double>::iterator;
        using const_iterator = std::vector<double>::const_iterator;

        inline iterator begin() { return m_Data.begin(); }
        inline const_iterator begin() const { return m_Data.begin(); }
        inline iterator end() { return m_Data.end(); }
        inline const_iterator end() const { return m_Data.end(); }

    private:
        std::vector<double> m_Data;
    };
} // kmeans

#endif //KMEANS_MPI_POINT_HPP