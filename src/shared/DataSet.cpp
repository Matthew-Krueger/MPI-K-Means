//
// Created by Matthew Krueger on 10/10/25.
//

#include "DataSet.hpp"

#include <algorithm>
#include <ranges>
#include <boost/random/normal_distribution.hpp>

namespace kmeans {

    // Choose treachery and evil auto return,
    // It's more FUN!
    // Uncle Iroh in The Ember Island Players (probably)
    auto DataSet::generateCluster(const Point &clusterCenter, size_t numberPoints, double clusterSpread,
                                  std::mt19937 &rng) {

        std::vector<Point> cluster;
        cluster.reserve(numberPoints);

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
        auto generatedPointsView = std::ranges::views::iota((size_t)0, numberPoints)
                           | std::ranges::views::transform([distributions, &rng](int) mutable { return generateSinglePoint(distributions, rng); });

        return generatedPointsView;

    }

    DataSet::DataSet(const Config& config) {

        // reserve the right number of samples
        m_Points.reserve(config.numTotalSamples);

        // create our random
        std::mt19937 rng(config.seed);

        // scope the generation of our known good centroids
        // so we can dump the distribution generators ASAP
        {
            // create our distributions (we are using linear distributions for now
            // This is on a PER DIMENSION BASIS.
            // ESSENTIALLY, WE NEED TO HAVE A VECTOR OF DISTRIBUTIONS SO THAT WE CAN THEN MAKE A VECTOR OF POINTS, RANDOMIZED WITH PER DIMENSION RANDOM NUMBERS
            std::vector<boost::normal_distribution<double>> distributions;
            distributions.reserve(config.numDimensions);

            // and actually create them
            auto clusterCentroidGeneratorDistributionView = std::ranges::views::iota((size_t)0, config.numDimensions - 1)
                | std::ranges::views::transform([&config](size_t dimension) {
                    return std::uniform_real_distribution<double>(config.clusterDimensionDistributions[dimension].low, config.clusterDimensionDistributions[dimension].high);
                });

            // and expand their pipeline
            auto clusterCentroidGeneratorDistribution = std::vector<std::uniform_real_distribution<double>>(clusterCentroidGeneratorDistributionView.begin(), clusterCentroidGeneratorDistributionView.end());

            // now, we can create our *actual* centroids with them
            // We are not using pipelining for this one since the generation is inherently stateful and thus problematic
            // if we attempt to assemble a data pipeline. Trust me, it makes your head explode
            m_KnownGoodCentroids = std::vector<Point>();
            m_KnownGoodCentroids.value().reserve(config.numTrueClusters);
            for (size_t sample = 0; sample < config.numTrueClusters; ++sample) {
                std::vector<double> coordinates;
                coordinates.reserve(config.numDimensions);
                for (auto& dist : clusterCentroidGeneratorDistribution) {
                    coordinates.push_back(dist(rng));
                }
                m_KnownGoodCentroids.value().emplace_back(coordinates);
            }
        }

        // now that we have known good centroids FROM WHICH we can generate our clusters, we can actually generate the cluster
        // First we need to get a list of the NUMBER of samples per centroid
        // In other words, the "stride" of the data
        size_t samplesPerCentroid = config.numTotalSamples / config.numTrueClusters;
        size_t samplesLeftover = config.numTotalSamples % config.numTrueClusters;
        // Create a view for the number of samples per cluster, accounting for leftovers.
        auto samplesPerClusterView = std::ranges::views::iota(size_t{0}, config.numTrueClusters)
            | std::ranges::views::transform([samplesPerCentroid, samplesLeftover](size_t clusterIdx) {
                // Distribute leftovers to the first few clusters.
                return samplesPerCentroid + (clusterIdx < samplesLeftover ? 1 : 0);
            });

        // Now that we know how many per centroid, we can call DataSet::generateCluster for each centroid, with the number of samples, the RNG, the cluster center as the centroid.
        // Generate clusters using a ranges pipeline.
        auto clustersView = std::ranges::views::iota(static_cast<size_t>(0), config.numTrueClusters)
            | std::ranges::views::transform([this, &rng, samplesPerClusterView, config](const size_t clusterIdx) {
                // Access the number of samples for the current cluster.
                const size_t numSamples = *(samplesPerClusterView.begin() + static_cast<long>(clusterIdx));
                // Generate a cluster around the known centroid.
                return generateCluster(
                    m_KnownGoodCentroids->at(clusterIdx),
                    numSamples,
                    config.clusterSpread,
                    rng
                );
            })
            | std::ranges::views::join; // Flatten the clusters into a single range of points.

        // Collect the generated points into m_Points
        std::ranges::move(clustersView, std::back_inserter(m_Points));
    }

    Point DataSet::generateSinglePoint(std::vector<boost::normal_distribution<double>>& distributions, std::mt19937 &rng) {

        // back inserter doesn't play nice so we will make a vector then move it to the
        std::vector<double> dimensionsForPoint;
        dimensionsForPoint.reserve(distributions.size()); // Pre-reserve for dimensions.

        std::ranges::transform(distributions,
                               std::back_inserter(dimensionsForPoint),
                               [&](auto& dist){ return dist(rng); }); // Sample each dimension

        return Point(std::move(dimensionsForPoint)); // Assuming Point can be constructed from std::vector<double>.
        // If Point *is* std::vector<double>, just 'return dimensionsForPoint;'

    }


} // kmeans