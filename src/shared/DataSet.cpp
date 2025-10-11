//
// Created by Matthew Krueger on 10/10/25.
//

#include "DataSet.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <ranges>

namespace kmeans {
    DataSet::DataSet(const Config& config) {

        m_Points.reserve(config.numTotalSamples);

    }

    void DataSet::generateCluster(const Point &clusterCenter, size_t numberPoints, double clusterSpread, std::mt19937 &rng) {
        // realloc if not big enough but it should always be
        if (m_Points.capacity()-m_Points.size()<numberPoints) {
            m_Points.reserve(m_Points.size()+numberPoints);
        }

        size_t clusterNumberDimensions = clusterCenter.getData().size();
        std::vector<boost::normal_distribution<double>> distributions;
        distributions.reserve(clusterNumberDimensions);

        // use std::ranges::transform to populate distributions with gauntness distributions *BASED ON* the cluster center
        // back inserter is used to make sure the distributions iterator stays valid
        std::ranges::transform(clusterCenter,
                               std::back_inserter(distributions),
                               [clusterSpread](double dimension) {
                                   return boost::normal_distribution<double>(dimension, clusterSpread);
                               }
        );

        // we will now generate points with each index in the vector using its own corresponding boost distribution.
        auto generate_single_point = [&]() {

            // back inserter doesn't play nice so we will make a vector then move it to the
            std::vector<double> dimensions_for_point;
            dimensions_for_point.reserve(distributions.size()); // Pre-reserve for dimensions.

            std::ranges::transform(distributions,
                                   std::back_inserter(dimensions_for_point),
                                   [&](auto& dist){ return dist(rng); }); // Sample each dimension

            return Point(std::move(dimensions_for_point)); // Assuming Point can be constructed from std::vector<double>.
            // If Point *is* std::vector<double>, just 'return dimensions_for_point;'
        };

        auto generated_points_view = std::ranges::views::iota((size_t)0, numberPoints)
                           | std::ranges::views::transform([&](int){ return generate_single_point(); });

        m_Points.insert(m_Points.end(), generated_points_view.begin(), generated_points_view.end());

    }



} // kmeans