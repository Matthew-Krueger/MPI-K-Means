//
// Created by mpiuser on 10/10/25.
//

#include "Point.hpp"

#include <expected>
#include <numeric>
#include <cmath>
#include <ranges>

namespace kmeans{

    Point::~Point(){

    }

    std::expected<double, std::string> Point::calculateEuclideanDistance(const Point& other){

        // guard against dimension mismatch
        if (m_Data.size() != other.m_Data.size()){
            return std::unexpected("Dimensions mismatch. This has " + std::to_string(m_Data.size()) + " dimensions, that has " + std::to_string(other.m_Data.size()) + " dimensions.");
        }

        auto squaredifference = [](double first, double second) -> double{
            return std::pow(first - second, 2);
        };

        double totalSum = std::transform_reduce(
            m_Data.begin(),
            m_Data.end(),
            other.m_Data.begin(),
            0.0,
            std::plus<double>(),
            [squaredifference](auto first, auto second){
                return squaredifference(first,second);
            }

        );

        return std::sqrt(totalSum);

    }

    std::expected<Point::FlattenedPoints, std::string> Point::flattenPoints(const std::vector<Point>& points){

        // get the sum of the lengths of the vectors, while ensuring the dimensionality matches
        const double expectedNumberDimensions = points[0].m_Data.size();
        std::expected<size_t, std::string> result = std::transform_reduce(
            points.begin(),
            points.end(),
            0.0,
            std::plus<size_t>(),
            [expectedNumberDimensions](const Point& point) -> std::expected<size_t, std::string> {
                size_t dimensions = point.m_Data.size();
                if(expectedNumberDimensions != dimensions) return std::unexpected("Dimensions mismatch in a point. Expected dimensionality is " + std::to_string(expectedNumberDimensions) + ", but got " + std::to_string(dimensions));
                return dimensions;
            }
        )


    }
} // kmeans