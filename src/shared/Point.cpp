//
// Created by mpiuser on 10/10/25.
//

#include "Point.hpp"

#include <algorithm>
#include <expected>
#include <numeric>
#include <cmath>
#include <ranges>

namespace kmeans {
    std::expected<double, std::string> Point::calculateEuclideanDistance(const Point &other) {
        // Guard against dimension mismatch.
        if (m_Data.size() != other.m_Data.size()) {
            return std::unexpected(
                "Dimensions mismatch. This has " + std::to_string(m_Data.size()) + " dimensions, that has " +
                std::to_string(other.m_Data.size()) + " dimensions.");
        }

        // Lambda function to calculate the squared difference between two doubles.
        auto squaredifference = [](double first, double second) -> double {
            return std::pow(first - second, 2);
        };

        // Calculate the sum of squared differences using std::transform_reduce.
        // m_Data.begin(), m_Data.end(): The range of elements from the current Point.
        // other.m_Data.begin(): The starting iterator for the other Point's data.
        // 0.0: The initial value for the sum.
        // std::plus<double>(): The binary operation to combine the results of the transform.
        // [squaredifference](auto first, auto second){ ... }: The transform operation, applying squaredifference to corresponding elements.
        double totalSum = std::transform_reduce(
            m_Data.begin(),
            m_Data.end(),
            other.m_Data.begin(),
            0.0,
            std::plus<>(),
            squaredifference
        );

        return std::sqrt(totalSum);
    }

    std::expected<Point::FlattenedPoints, std::string> Point::flattenPoints(const std::vector<Point> &points) {
        // Return an error if no points are provided.
        if (points.empty()) {
            return std::unexpected("No points provided");
        }

        // Get the first point's dimensionality, which becomes the expected dimensionality
        const size_t expectedNumberDimensions = points[0].m_Data.size();

        if (expectedNumberDimensions == 0) {
            return std::unexpected("Expected Dimensionality cannot be zero");
        }

        // Check if all points in the vector have the expected number of dimensions.
        bool allHaveRequiredNumberDimensions = std::all_of(points.begin(), points.end(),
                                                           [expectedNumberDimensions](const Point &point) {
                                                               return expectedNumberDimensions == point.m_Data.size();
                                                           });

        // Return an error if not all points have the same number of dimensions.
        if (!allHaveRequiredNumberDimensions) {
            return std::unexpected("All points must have the same number of dimensions");
        }

        // Construct a result view by joining the data of all points.
        const auto resultView = points | std::ranges::views::join;

        // now use that view to construct a flattened vector
        std::vector<double> flattenedPoints(resultView.begin(), resultView.end());

        return Point::FlattenedPoints{
            expectedNumberDimensions,
            points.size(),
            flattenedPoints
        };
    }

    std::expected<std::vector<Point>, std::string>
    Point::unflattenPoints(const Point::FlattenedPoints &flattenedPoints) {
        // Validate that the total number of elements in the flattened vector matches the expected count.
        // totalEntries: The expected total number of elements (numPoints * numDimensionsPerPoint).
        const size_t totalEntries = flattenedPoints.numDimensionsPerPoint * flattenedPoints.numPoints;
        // Check if the actual size of the flattened points vector matches the expected total entries.
        if (flattenedPoints.points.size() != totalEntries) {
            return std::unexpected(
                "Flattened points vector size mismatch. Expected " +
                std::to_string(totalEntries) + " (" +
                std::to_string(flattenedPoints.numPoints) + " points * " +
                std::to_string(flattenedPoints.numDimensionsPerPoint) + " dims) but got " +
                std::to_string(flattenedPoints.points.size()) + "."
            );
        }

        // result: A vector to store the unflattened Point objects.
        std::vector<Point> result;
        // Reserve memory for the expected number of points to avoid reallocations.
        result.reserve(flattenedPoints.numPoints);

        // Iterate through the flattened points data and reconstruct individual Point objects.
        for (size_t currentPointStartingIndex = 0; currentPointStartingIndex < totalEntries;
             currentPointStartingIndex += flattenedPoints.numDimensionsPerPoint) {
            // Extract the data for a single point.
            std::vector<double> pointData(flattenedPoints.points.begin() + static_cast<long>(currentPointStartingIndex),
                                          flattenedPoints.points.begin() + static_cast<long>(currentPointStartingIndex) + static_cast<long>(flattenedPoints.
                                              numDimensionsPerPoint));
            result.emplace_back(pointData);
        }

        return result;
    }

    std::expected<ClusterLocalAggregateSum, std::string> ClusterLocalAggregateSum::calculateCentroidLocalSum(const std::vector<Point> &points) {
        // Return an error if no points are provided. This should still run in critical paths.
        if (points.empty()) {
            return std::unexpected("No points provided");
        }

        // Get the first point's dimensionality, which becomes the expected dimensionality
        const size_t expectedNumberDimensions = points[0].getData().size();

        // since this will be an actual critical path, we should wrap this in ifdef ndebug so we can strip it out in release builds
#ifndef NDEBUG


        if (expectedNumberDimensions == 0) {
            return std::unexpected("Expected Dimensionality cannot be zero");
        }

        // Check if all points in the vector have the expected number of dimensions.
        bool allHaveRequiredNumberDimensions = std::all_of(points.begin(), points.end(),
                                                           [expectedNumberDimensions](const Point &point) {
                                                               return expectedNumberDimensions == point.getData().size();
                                                           });

        // Return an error if not all points have the same number of dimensions.
        if (!allHaveRequiredNumberDimensions) {
            return std::unexpected("All points must have the same number of dimensions");
        }

#endif

        // Initialize a vector to store the sum of each dimension


        // Sum coordinates element-wise using accumulate
        // this approach uses a range based for loop and std::ranges::transform to transform each vector into the accumulator
        // so effectively, we calculate a vector sum by accumulating every point into the vector,
        // such that if we have v1, v2 and v3, each with [a,b,c]
        // we get [v1.a + v2.a + v3.a, v1.b + v2.b + v3.b, ...]
        // we cannot use std::accumulate as it follows strict semantics per the documentation which may not be optimized away

        std::vector<double> centroidLocalSum(expectedNumberDimensions, 0.0);
        for (const auto &point:points) {
            std::ranges::transform(
                centroidLocalSum,
                point.getData(),
                centroidLocalSum.begin(),
                std::plus<>()
            );
        }


        // Leaving this to remind me how to do the average
        // We don't need to do this right now because MPI will handle it or we will externally. This is just for reference.
        // // Divide by the number of points to get the mean
        // std::ranges::transform(
        //     centroidData,
        //     centroidData.begin(),
        //     [numPoints = points.size()](double sum) { return sum / static_cast<double>(numPoints); });

        return ClusterLocalAggregateSum(Point(centroidLocalSum),points.size());

    }
} // kmeans
