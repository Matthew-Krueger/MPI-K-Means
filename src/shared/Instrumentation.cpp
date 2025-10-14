//
// Created by Matthew Krueger on 10/14/25.
//

#include "Instrumentation.hpp"

#include <algorithm>
#include <fstream>
#include <boost/tuple/detail/tuple_basic.hpp>

#define DEBUG_INSTRUMENTATION


#ifdef DEBUG_INSTRUMENTATION
#define INSTRUMENTATION_DEBUG_INSTRUMENTATION true
#else
#define INSTRUMENTATION_DEBUG_INSTRUMENTATION false
#endif

namespace instrumentation {
    std::shared_ptr<Instrumentor> Instrumentor::s_GlobalInstrumentor = nullptr;

    std::weak_ptr<Instrumentor> Instrumentor::getGlobalInstrumentor() {
        return Instrumentor::s_GlobalInstrumentor;
    }

    Instrumentor::Instrumentor(std::unique_ptr<Writer> &&writer) {
        m_Writer = std::move(writer);
    }

    void Instrumentor::recordEntry(Entry &&entry) {
        m_LocalLog.push_back(std::move(entry));

        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Wrote an instrumentation entry. Local Log Size: " << m_LocalLog.size() << " entries" <<
                    std::endl;
        }

        if (m_LocalLog.size() > m_Writer->getTargetBufferSize()) {
            flush();
        }
    }

    Instrumentor::~Instrumentor() {
        flush();
        m_Writer->flush();
        // flush the writer too since it will go out of scope when this does. The writer will finalize itself on destruct however since it is not a global singleton
    }


    void Instrumentor::flush() {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Flushing Instrumentation Log" << std::endl;
        }
        m_Writer->write(m_LocalLog);
        m_LocalLog.clear();
    }

    void Instrumentor::initializeGlobalInstrumentor(std::unique_ptr<Writer> &&writer) {
        if (s_GlobalInstrumentor == nullptr) {
            s_GlobalInstrumentor = std::make_shared<Instrumentor>(std::move(writer));
        } else {
            std::cout << "Global Instrumentor already exists. Ignoring request." << std::endl;
        }
    }

    void Instrumentor::finalizeGlobalInstrumentor() {
        s_GlobalInstrumentor.reset(); // invalidate *all* instances of this instrumentor
    }
}
#ifdef BUILD_WITH_MPI
#include <mpi.h>
#endif

namespace instrumentation {

    std::variant<std::chrono::time_point<std::chrono::high_resolution_clock>, double> Session::getTimePoint() {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Getting time point" << std::endl;
        }

        #ifdef BUILD_WITH_MPI
        return MPI_Wtime();
        #else
        return std::chrono::high_resolution_clock::now();
        #endif
    }

#ifdef BUILD_WITH_MPI
    MPIWriter::MPIWriter(const Config &config) : Writer(config.targetBufferSize) {
        m_MyRank = std::numeric_limits<int>::max();
        m_Config = config;

        MPI_Comm_rank(MPI_COMM_WORLD, &m_MyRank);

        m_WriteBuffer = std::vector<char>(); // initialize the buffer

        m_IsFirstFlush = true;
    }

    MPIWriter::~MPIWriter() {
        MPIWriter::flush();
        if (std::ofstream file(m_Config.logFileName, std::ios::app); file.is_open()) {
            MPIWriter::write_tail(file);
        }
    }

    void MPIWriter::write(const std::vector<Entry> &entries) {
        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Writing an MPIWriter entry." << std::endl;
        }

        // we can just directly copy to the buffer
        for (const auto &entry: entries) {
            const std::string &entryText = entry.to_string();
            if (!entryText.empty()) {
                auto currentDisplacement = static_cast<uint32_t>(m_WriteBuffer.size());
                m_Displacements.push_back(currentDisplacement);

                // Reserve space and copy the string
                size_t oldSize = m_WriteBuffer.size();
                m_WriteBuffer.reserve(oldSize + entryText.size());
                m_WriteBuffer.insert(m_WriteBuffer.begin(), entryText.begin(), entryText.end());
            }
        }

        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Wrote an MPIWriter entry. Local Log Size: " << m_WriteBuffer.size() << " characters" << std::endl;
            std::cout << "Displacements: " << m_Displacements.size() << std::endl;
            std::cout << "Contents: " << m_WriteBuffer.data() << std::endl;
        }
    }

    void MPIWriter::flush() {

        if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
            std::cout << "Flushing MPIWriter. Local buffer size: " << m_WriteBuffer.size() << std::endl;
        }

        if (m_MyRank == m_Config.mainRank) {

            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Main rank flushing" << std::endl;
            }

            // Main rank: receive from all processes and write to file/stderr/stdout
            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            // Receive counts from all processes
            std::vector<int> recv_counts(world_size, 0);
            const int size = static_cast<int>(m_Displacements.size());
            MPI_Gather(&size, 1, MPI_INT,
                       recv_counts.data(), 1, MPI_INT,
                       m_Config.mainRank, MPI_COMM_WORLD);

            // Calculate total size and displacements
            std::vector<int> global_displacements(world_size, 0);
            int total_size = 0;
            for (int i = 0; i < world_size; ++i) {
                global_displacements[i] = total_size;
                total_size += recv_counts[i];
            }

            // Allocate receive buffer
            std::vector<char> recv_buffer(total_size);

            // Gather the actual data
            MPI_Gatherv(m_WriteBuffer.data(), static_cast<int>(m_Displacements.size()), MPI_CHAR,
                        recv_buffer.data(), recv_counts.data(), global_displacements.data(), MPI_CHAR,
                        m_Config.mainRank, MPI_COMM_WORLD);

            // Now write to the appropriate destination
            std::ofstream file(m_Config.logFileName, std::ios::app); // Append to avoid overwriting

            if (file.is_open()) {
                if (m_IsFirstFlush) {
                    write_preamble(file);
                    m_IsFirstFlush = false;
                }

                writeEntriesToFile(recv_buffer, global_displacements, file);
                file.flush();

            } else {
                std::cerr << "Failed to open file: " << m_Config.logFileName << std::endl;
            }

            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Main rank finished flushing" << std::endl;
            }

        } else {

            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Non-main rank flushing" << std::endl;
            }

            // Non-main ranks: send data to main rank
            int local_count = static_cast<int>(m_Displacements.size());
            MPI_Gather(&local_count, 1, MPI_INT,
                       nullptr, 0, MPI_CHAR,
                       m_Config.mainRank, MPI_COMM_WORLD);

            if (local_count > 0) {
                MPI_Gatherv(m_WriteBuffer.data(), local_count, MPI_CHAR,
                            nullptr, nullptr, nullptr, MPI_CHAR,
                            m_Config.mainRank, MPI_COMM_WORLD);
            }

            if constexpr (INSTRUMENTATION_DEBUG_INSTRUMENTATION) {
                std::cout << "Non-main rank finished flushing" << std::endl;
            }

        }

        // Clear buffers after flush
        m_WriteBuffer.clear();
        m_Displacements.clear();
    }

    void MPIWriter::writeEntriesToFile(const std::vector<char> &all_data, const std::vector<int> &global_displacements,
                                       std::ostream &file) {
        if (all_data.empty() || global_displacements.empty()) {
            return;
        }

        size_t offset = 0;
        bool first_entry = true;

        // this is just conceptually easier to understand as a for loop rather than functionally
        for (size_t i = 0; i < global_displacements.size(); ++i) {
            const char *entry_start = all_data.data() + global_displacements[i];
            const char *entry_end = (i + 1 < global_displacements.size())
                                        ? entry_start + (global_displacements[i + 1] - global_displacements[i])
                                        : entry_start + (all_data.size() - global_displacements[i]);

            if (entry_end > entry_start && entry_end - entry_start > 0) {
                if (!first_entry) {
                    file << COMMA_NEWLINE;
                }

                // Write the entry as a string
                file.write(entry_start, entry_end - entry_start);
                first_entry = false;
            }
        }
    }

#endif
}
