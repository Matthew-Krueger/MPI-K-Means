
#include <boost/mpi.hpp>

#include "shared/DataSet.hpp"
#include "shared/Logging.hpp"
#include "shared/Point.hpp"

int main(int argc, char** argv)
{

    DEBUG_PRINT("Creating MPI Environment");
    boost::mpi::environment mpiEnvironment(argc, argv);
    boost::mpi::communicator worldCommunicator;

    kmeans::DataSet::Config datasetConfig{
        {{0,1}, {1,2}, {2,3}},
        50000,
        3,
        3000,
        0.01,
        1
    };


    kmeans::DataSet dataSet(datasetConfig);

    for (auto& point:dataSet) {
        std::cout << point << std::endl;
    }

    std::cout << "I am rank " << worldCommunicator.rank() << " of a world size " << worldCommunicator.size() << std::endl;

}
