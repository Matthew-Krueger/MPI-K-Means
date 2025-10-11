
#include <boost/mpi.hpp>

#include "shared/Point.hpp"

int main(int argc, char** argv)
{

    boost::mpi::environment mpiEnvironment(argc, argv);
    boost::mpi::communicator worldCommunicator;

    kmeans::Point point({3});
    kmeans::Point point2({4});

    std::cout << point << std::endl;
    std::cout << point2 << std::endl;

    std::cout << kmeans::ClusterLocalAggregateSum::calculateCentroidLocalSum({point,point2}).value().localSumData << std::endl;

    std::cout << point.calculateEuclideanDistance(point2).value() << std::endl;

    std::cout << "I am rank " << worldCommunicator.rank() << " of a world size " << worldCommunicator.size() << std::endl;

}
