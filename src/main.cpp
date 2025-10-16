
#include <boost/mpi.hpp>

#include "mpi/MPISolver.hpp"
#include "serial/SerialSolver.hpp"
#include "shared/DataSet.hpp"
#include "shared/Logging.hpp"
#include "shared/Point.hpp"
#include "shared/Instrumentation.hpp"

int main(int argc, char** argv)
{

    DEBUG_PRINT("Creating MPI Environment");
    boost::mpi::environment mpiEnvironment(argc, argv);
    boost::mpi::communicator worldCommunicator;

    auto writer = new instrumentation::MPIWriter(instrumentation::MPIWriter::Config{
        "log.json",
        0,
        5020,
        0
    });

    PROFILE_BEGIN_SESSION(std::unique_ptr<instrumentation::MPIWriter>(writer));

    {
        PROFILE_SCOPE("Serial Run");
        DEBUG_PRINT("Finished creating MPI Environment");
        DEBUG_PRINT("Creating Dataset ");

        kmeans::DataSet::Config datasetConfig{
            {{0,10}, {10,20}, {20,30}},
            10,
            3,
            2,
            3.5,
            1
        };


        kmeans::DataSet dataSet = kmeans::DataSet();
        if (worldCommunicator.rank() == 0) {
            dataSet = kmeans::DataSet(datasetConfig);
        }

        DEBUG_PRINT("Finished creating Dataset");

        DEBUG_PRINT("Printing known good centroids");
        if (dataSet.getKnownGoodCentroids().has_value()) {
            for (auto&point : dataSet.getKnownGoodCentroids().value()) {
                std::cout << point << std::endl;
            }
        }

        /*DEBUG_PRINT("Creating solver");
        kmeans::SerialSolver::Config solverConfig(
                1000,
                0.0001,
                std::move(dataSet),
                1234,
                35
            );

        DEBUG_PRINT("Created Solver Config");
        kmeans::SerialSolver solver(solverConfig);

        solver.run();*/

        DEBUG_PRINT("Parallel Solver");

        kmeans::MPISolver::Config solverConfig({
            1000,
            0.0001,
            std::move(dataSet),
            1234,
            2,
            0,
            1
        });

        DEBUG_PRINT("Created Solver Config");
        DEBUG_PRINT("Rank " << worldCommunicator.rank() << ". Initalize Solver");
        kmeans::MPISolver mpiSolver(std::move(solverConfig), worldCommunicator);

        mpiSolver.run();

        std::cout << "I am rank " << worldCommunicator.rank() << " of a world size " << worldCommunicator.size() << std::endl;



        PROFILE_END_SESSION();

    }
}
