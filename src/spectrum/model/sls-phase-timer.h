/*
 * Lightweight phase-timing instrumentation. Off by default; compile
 * with `-DSLS_PROFILE_INSTRUMENT=1` to enable. Adds a few ns of overhead
 * per scope entry/exit when on, zero when off.
 *
 * Usage:
 *     #include "sls-phase-timer.h"
 *     ...
 *     {
 *         SLS_PHASE_SCOPE("GenerateChannelParameters");
 *         ... work ...
 *     }
 *
 * At program exit, the accumulated counters are dumped to stderr.
 */

#ifndef SLS_PHASE_TIMER_H
#define SLS_PHASE_TIMER_H

#if SLS_PROFILE_INSTRUMENT

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>

namespace sls
{

struct PhaseStats
{
    const char* name = nullptr;
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> totalNanos{0};
};

namespace detail
{

// Fixed-size table of named phase counters. The first call to
// `getOrCreate(name)` for a given pointer claims a slot; subsequent
// calls with the same string literal pointer reuse it. We hash by
// pointer identity (since SLS_PHASE_SCOPE takes string literals).
inline PhaseStats&
getOrCreate(const char* name)
{
    static std::array<PhaseStats, 32> table;
    for (auto& s : table)
    {
        const char* nm = s.name;
        if (nm == name)
            return s;
        if (nm == nullptr)
        {
            // Claim the slot. Race-tolerant: worst case two threads
            // claim adjacent slots for the same name; benchmark is
            // single-threaded anyway.
            s.name = name;
            return s;
        }
    }
    // Table full — return the last slot; we'll under-report but won't
    // crash. Bump the table size if this fires in practice.
    return table.back();
}

// Dumps the accumulated counters at program exit.
struct AutoDumper
{
    ~AutoDumper()
    {
        std::fprintf(stderr, "\n==== SLS phase timing (ns total / calls / ns avg) ====\n");
        for (uint32_t i = 0; i < 32; ++i)
        {
            // We can't loop the same array from here without a getter;
            // call getOrCreate(nullptr) is unsafe. Instead use a wrapper.
        }
    }
};

inline std::array<PhaseStats, 32>&
table()
{
    static std::array<PhaseStats, 32> t;
    return t;
}

inline PhaseStats&
slot(const char* name)
{
    auto& t = table();
    for (auto& s : t)
    {
        const char* nm = s.name;
        if (nm == name)
            return s;
        if (nm == nullptr)
        {
            s.name = name;
            return s;
        }
    }
    return t.back();
}

struct Dumper
{
    ~Dumper()
    {
        std::fprintf(stderr,
                     "\n==== SLS phase timing (calls / total ms / us avg) ====\n");
        for (auto& s : table())
        {
            const char* nm = s.name;
            uint64_t calls = s.calls.load(std::memory_order_relaxed);
            if (!nm || calls == 0)
                continue;
            uint64_t totalNs = s.totalNanos.load(std::memory_order_relaxed);
            std::fprintf(stderr,
                         "  %-44s %10llu  %10.2f  %10.2f\n",
                         nm,
                         static_cast<unsigned long long>(calls),
                         totalNs / 1.0e6,
                         (totalNs / 1.0e3) / static_cast<double>(calls));
        }
    }
};

inline Dumper&
dumper()
{
    static Dumper d;
    return d;
}

} // namespace detail

class Scope
{
  public:
    explicit Scope(const char* name)
        : m_stats(detail::slot(name)),
          m_start(std::chrono::steady_clock::now())
    {
        // Force the dumper to be created so its destructor fires at
        // program exit.
        (void)detail::dumper();
    }
    ~Scope()
    {
        auto end = std::chrono::steady_clock::now();
        uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - m_start).count());
        m_stats.calls.fetch_add(1, std::memory_order_relaxed);
        m_stats.totalNanos.fetch_add(ns, std::memory_order_relaxed);
    }

  private:
    PhaseStats& m_stats;
    std::chrono::steady_clock::time_point m_start;
};

} // namespace sls

#define SLS_PHASE_SCOPE(name) ::sls::Scope _sls_phase_scope_##__LINE__(name)

#else // !SLS_PROFILE_INSTRUMENT

#define SLS_PHASE_SCOPE(name) ((void)0)

#endif

#endif // SLS_PHASE_TIMER_H
