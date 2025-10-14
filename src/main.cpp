
#include <boost/mpi.hpp>

#include "serial/SerialSolver.hpp"
#include "shared/DataSet.hpp"
#include "shared/Logging.hpp"
#include "shared/Point.hpp"

int main(int argc, char** argv)
{

    DEBUG_PRINT("Creating MPI Environment");
    boost::mpi::environment mpiEnvironment(argc, argv);
    boost::mpi::communicator worldCommunicator;

    DEBUG_PRINT("Finished creating MPI Environment");
    DEBUG_PRINT("Creating Dataset ");

    kmeans::DataSet::Config datasetConfig{
        {{0,10}, {10,20}, {20,30}},
        50000000,
        3,
        35,
        3.5,
        1
    };


    kmeans::DataSet dataSet(datasetConfig);

    DEBUG_PRINT("Finished creating Dataset");

    DEBUG_PRINT("Printing known good centroids");
    for (auto&point : dataSet.getKnownGoodCentroids().value()) {
        std::cout << point << std::endl;
    }

    DEBUG_PRINT("Creating solver");
    kmeans::SerialSolver::Config solverConfig(
            1000,
            0.0001,
            std::move(dataSet),
            1234,
            35
        );

    DEBUG_PRINT("Created Solver Config");
    kmeans::SerialSolver solver(solverConfig);

    solver.run();

    std::cout << "I am rank " << worldCommunicator.rank() << " of a world size " << worldCommunicator.size() << std::endl;

}
