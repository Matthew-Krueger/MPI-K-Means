//
// Created by mpiuser on 10/10/25.
//

#ifndef KMEANS_MPI_POINT_HPP
#define KMEANS_MPI_POINT_HPP
#include <vector>
#include <cstddef>
#include <expected>
#include <string>

namespace kmeans{
    class Point{
    public:
        struct FlattenedPoints
        {
            size_t numDimensionsPerPoint;
            size_t numPoints;
            std::vector<double> points;
        };

        Point() = default;

        Point(std::vector<double> data) noexcept: m_Data(std::move(data)){}

        Point(const Point& other) : m_Data{other.m_Data}{}

        Point(Point&& other) noexcept : m_Data{std::move(other.m_Data)}{}

        Point& operator=(Point other){
            using std::swap;
            swap(*this, other);
            return *this;
        }

        ~Point() = default;

        const std::vector<double>& getData() { return m_Data; };
        void setData(std::vector<double> data) { m_Data = std::move(data); };

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