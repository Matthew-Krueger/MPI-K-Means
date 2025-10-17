//
// Created by Matthew Krueger on 10/17/25.
//

#ifndef KMEANS_MPI_TIMER_HPP
#define KMEANS_MPI_TIMER_HPP
#include <memory>
#include <variant>


    namespace timer {
        template<typename T>
        /**
         * Represents the result of a function execution along with the time taken, measured in microseconds.
         *
         * @tparam T Type of the function result.
         */
        struct TimeResult {
            /**
             * Stores the result of the executed function.
             * This value represents the output of the function being timed.
             */
            T functionResult;
            /**
             * Represents the elapsed time in microseconds. This value can be used to
             * derive time in other units such as milliseconds or seconds.
             */
            uint64_t timeMicroseconds;

            /**
             * Retrieves the elapsed time in milliseconds by converting the stored microseconds value.
             * The conversion is performed by dividing the internal time value, in microseconds, by 10^3.
             * @return The elapsed time in milliseconds as a 64-bit unsigned integer.
             */
            inline uint64_t getTimeMilliseconds() const { return timeMicroseconds/1000UL; }
            /**
             * Retrieves the elapsed time in seconds by dividing the stored time in microseconds by 1,000,000.
             * Provides a second-level granularity representation of the total recorded time.
             * @return The elapsed time in seconds as an unsigned 64-bit integer.
             */
            inline uint64_t getTimeSeconds() const { return timeMicroseconds/1000000UL; }
            /**
             * Converts the time, stored in microseconds, to a double representation in seconds.
             * This provides a higher precision floating-point representation of elapsed time.
             * @return The elapsed time in seconds as a double value.
             */
            inline double getTimeSecondsDouble() const { return timeMicroseconds/static_cast<double>(1e6); }
        };

        class Timer {
        public:
            /**
             * When called with a weak pointer, this will write the time elapsed, in microseconds, to m_TimeReference.
             * If m_TimeReference no longer is in scope, nothing will happen.
             * @param timeReference A pointer in which to write the resultant time. If it goes out of scope, nothing will be written
             */
            Timer(std::weak_ptr<uint64_t> timeReference);
            ~Timer();

        private:
            /**
             * Contains a start timepoint, long is in microseconds, double is in seconds
             */
            std::variant<uint64_t, double> m_StartTimePoint = 0.0;
            std::weak_ptr<uint64_t> m_TimeReference;
        };

        template<typename FuncToTime>
        /**
         * Measures the execution time of a given function and returns the result along with the time taken.
         *
         * @tparam FuncToTime The type of the function whose execution time is to be measured.
         *                    The function should take no arguments and return a value.
         * @param toTime A callable function whose execution time will be measured.
         * @return A TimeResult object containing the result of the function execution along with the time taken in microseconds.
         */
        TimeResult<decltype(std::declval<FuncToTime>()())> time(FuncToTime toTime) {
            TimeResult<decltype(std::declval<FuncToTime>()())> result;
            std::shared_ptr<uint64_t> timeReference = 0;
            {
                Timer timer(timeReference);
                result.functionResult = toTime();
            }
            result.timeMicroseconds = *timeReference;
            return result;

        }

    }


#endif //KMEANS_MPI_TIMER_HPP