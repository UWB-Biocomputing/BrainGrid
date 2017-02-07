/**
 * @brief Represents a CPU thread barrier
 * @note The barrier automatically resets after all threads are synced
 */

#include <mutex>
#include <condition_variable>

class Barrier
{
private:
    std::mutex m_mutex;
    std::condition_variable m_cv;

    size_t m_count;
    const size_t m_initial;

    enum State : unsigned char {
        Up, Down
    };
    State m_state;

public:
    explicit Barrier(std::size_t count) : m_count{ count }, m_initial{ count }, m_state{ State::Down } { }

    /// Blocks until all N threads reach here
    void Sync()
    {
        std::unique_lock<std::mutex> lock{ m_mutex };

        if (m_state == State::Down)
        {
            // Counting down the number of syncing threads
            if (--m_count == 0) {
                m_state = State::Up;
                m_cv.notify_all();
            }
            else {
                m_cv.wait(lock, [this] { return m_state == State::Up; });
            }
        }

        else // (m_state == State::Up)
        {
            // Counting back up for Auto reset
            if (++m_count == m_initial) {
                m_state = State::Down;
                m_cv.notify_all();
            }
            else {
                m_cv.wait(lock, [this] { return m_state == State::Down; });
            }
        }
    }
};  
