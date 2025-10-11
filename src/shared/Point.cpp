//
// Created by mpiuser on 10/10/25.
//

#include "Point.hpp"

#include <algorithm>
#include <expected>
#include <numeric>
#include <cmath>
#include <ranges>

namespace kmeans{

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

    std::expected<Point::FlattenedPoints, std::string> Point::flattenPoints(const std::vector<Point>& points) {

        if (points.empty()) {
            return std::unexpected("No points provided");
        }

        // validate that all points have the same dimensionality
        const size_t expectedNumberDimensions = points[0].m_Data.size();
        bool allHaveRequiredNumberDimensions = std::all_of(points.begin(), points.end(), [expectedNumberDimensions](const Point& point) {
            return expectedNumberDimensions == point.m_Data.size();
        });
        if (!allHaveRequiredNumberDimensions) {
            return std::unexpected("All points must have the same number of dimensions");
        }

        // construct a result view
        const auto resultView = points | std::ranges::views::join;

        // now use that view to construct a flattened vector
        std::vector<double> flattenedPoints(resultView.begin(), resultView.end());

        return Point::FlattenedPoints{
            expectedNumberDimensions,
            points.size(),
            flattenedPoints
        };

    }

    std::expected<std::vector<Point>, std::string> Point::unflattenPoints(const Point::FlattenedPoints &flattenedPoints) {
        // validate the data makes sense at all
        const size_t totalEntries = flattenedPoints.numDimensionsPerPoint * flattenedPoints.numPoints;
        if (flattenedPoints.points.size() != totalEntries) {
            return std::unexpected(
                "Illogical number of points or dimensionality. Expected " +
                std::to_string(totalEntries) + " points, but got " + std::to_string(flattenedPoints.points.size()) + " instead. "
                "Should have " + std::to_string(flattenedPoints.numPoints) + " points and " + std::to_string(flattenedPoints.numDimensionsPerPoint) + " dimensions per point");
        }

        std::vector<Point> result;
        result.reserve(flattenedPoints.numPoints);

        // go through the points, and unflatten them into result
        for (size_t currentPointStartingIndex = 0; currentPointStartingIndex < totalEntries; currentPointStartingIndex += flattenedPoints.numDimensionsPerPoint) {
            std::vector<double> pointData(flattenedPoints.points.begin() + currentPointStartingIndex, flattenedPoints.points.begin() + currentPointStartingIndex + flattenedPoints.numDimensionsPerPoint);
            result.emplace_back(pointData);
        }

        return result;

    }


} // kmeans