//
// Created by Matthew Krueger on 10/13/25.
//

#ifndef KMEANS_MPI_UTILS_HPP
#define KMEANS_MPI_UTILS_HPP
#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>
#include <optional>

namespace kmeans {

    inline bool areVectorsPracticallyTheSame(const std::vector<double>& lhs, const std::vector<double>& rhs, std::optional<double> epsilon = std::nullopt) {

        // guard against unequal vectors
        // since we don't care the *reason* it's invalid, we can just return false
        if (lhs.size()!=rhs.size()) {
            return false;
        }

        // set epsilon to numeric limits episilon if not set
        if (!epsilon.has_value()) {
            epsilon = std::numeric_limits<double>::epsilon();
        }

        // zip up the two vectors to iterate through them together.
        auto vectorView = std::ranges::views::zip(lhs, rhs);

        // accumulate the truth of the similarity, according to the formula in the lambda
        return std::ranges::all_of(
            vectorView.begin(),
            vectorView.end(),
            [epsilon](auto && pairElements) {
                // subtract the absolute values of each element, and they are "equal" if they are smaller than epsilon
                return (std::fabs(std::get<0>(pairElements)) - std::fabs(std::get<1>(pairElements))) < epsilon;
            }
        );

    }

}

#endif //KMEANS_MPI_UTILS_HPP