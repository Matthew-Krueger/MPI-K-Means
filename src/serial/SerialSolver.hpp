//
// Created by Matthew Krueger on 10/13/25.
//

#ifndef KMEANS_MPI_SERIALSOLVER_HPP
#define KMEANS_MPI_SERIALSOLVER_HPP

#include "../shared/DataSet.hpp"

namespace kmeans {
    class SerialSolver {
    public:
        SerialSolver(DataSet &&dataSet);
        ~SerialSolver() = default;
    };
} // kmeans

#endif //KMEANS_MPI_SERIALSOLVER_HPP