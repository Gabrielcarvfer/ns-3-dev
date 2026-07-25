/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

/**
 * @file
 * Microbenchmark of the per-invocation cost of the callback dispatch
 * mechanisms used on ns-3 fast paths, in nanoseconds per call:
 *
 *  - direct (non-inlinable) member function call, as a floor
 *  - member function pointer call
 *  - std::function holding a lambda
 *  - ns3::Callback created with MakeCallback (the fast-path mechanism)
 *  - a two-word object/trampoline delegate, prototyping a possible
 *    Callback fast path
 *  - ns3::TracedCallback with zero and one connected sinks
 *
 * Example: ./ns3 run bench-callbacks
 *
 * Example output, from a default profile build with GCC on an AMD Ryzen 9
 * 7940HS. Absolute values depend on the machine and build profile; the
 * ratios between rows are the point of interest.
 *
 * @verbatim
   mechanism                            ns/call
   direct member call                      1.03
   member function pointer                 1.03
   std::function(lambda)                   1.88
   ns3::Callback (MakeCallback)            3.35
   object+trampoline delegate              1.02
   TracedCallback, 0 sinks                 1.23
   TracedCallback, 1 sink                  4.63
   @endverbatim
 */

#include "ns3/callback.h"
#include "ns3/command-line.h"
#include "ns3/traced-callback.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>

using namespace ns3;

/// A call target whose implementation the compiler cannot inline.
class Target
{
  public:
    /**
     * The benchmarked operation.
     * @param x An argument.
     */
    __attribute__((noinline)) void Consume(uint32_t x);

    uint64_t m_acc{0}; //!< Accumulator defeating dead code elimination.
};

void
Target::Consume(uint32_t x)
{
    m_acc += x;
}

/// Two-word delegate: object pointer plus stamped-out trampoline.
struct Delegate
{
    void* m_obj;                           //!< The bound object.
    void (*m_trampoline)(void*, uint32_t); //!< The stamped-out call adapter.

    /**
     * Create a delegate bound to a member function.
     * @tparam T The object type.
     * @tparam MEM The bound member function.
     * @param obj The object.
     * @return The delegate.
     */
    template <typename T, void (T::*MEM)(uint32_t)>
    static Delegate Make(T* obj)
    {
        return {obj, [](void* o, uint32_t x) { (static_cast<T*>(o)->*MEM)(x); }};
    }

    /**
     * Invoke the delegate.
     * @param x The argument.
     */
    void operator()(uint32_t x) const
    {
        m_trampoline(m_obj, x);
    }
};

/**
 * Time a callable over many iterations.
 * @tparam F The callable type.
 * @param label Row label.
 * @param iterations Number of calls.
 * @param f The callable, taking the iteration counter.
 */
template <typename F>
void
Bench(const char* label, uint32_t iterations, F&& f)
{
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; i++)
    {
        f(i);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iterations;
    std::cout << std::left << std::setw(34) << label << std::right << std::setw(10) << std::fixed
              << std::setprecision(2) << ns << std::endl;
}

int
main(int argc, char* argv[])
{
    uint32_t iterations = 100000000;
    CommandLine cmd(__FILE__);
    cmd.Usage("Microbenchmark of the per-invocation cost of the callback dispatch\n"
              "mechanisms used on ns-3 fast paths, reported in nanoseconds per call.");
    cmd.AddValue("iterations", "Calls per benchmark", iterations);
    cmd.Parse(argc, argv);

    Target target;
    Target* obj = &target;

    void (Target::*memFn)(uint32_t) = &Target::Consume;
    std::function<void(uint32_t)> stdFn = [obj](uint32_t x) { obj->Consume(x); };
    Callback<void, uint32_t> cb = MakeCallback(&Target::Consume, obj);
    Delegate delegate = Delegate::Make<Target, &Target::Consume>(obj);
    TracedCallback<uint32_t> tracedEmpty;
    TracedCallback<uint32_t> tracedOne;
    tracedOne.ConnectWithoutContext(cb);

    std::cout << std::left << std::setw(34) << "mechanism" << std::right << std::setw(10)
              << "ns/call" << std::endl;

    Bench("direct member call", iterations, [&](uint32_t i) { obj->Consume(i); });
    Bench("member function pointer", iterations, [&](uint32_t i) { (obj->*memFn)(i); });
    Bench("std::function(lambda)", iterations, [&](uint32_t i) { stdFn(i); });
    Bench("ns3::Callback (MakeCallback)", iterations, [&](uint32_t i) { cb(i); });
    Bench("object+trampoline delegate", iterations, [&](uint32_t i) { delegate(i); });
    Bench("TracedCallback, 0 sinks", iterations, [&](uint32_t i) { tracedEmpty(i); });
    Bench("TracedCallback, 1 sink", iterations, [&](uint32_t i) { tracedOne(i); });

    // Keep the accumulator observable.
    std::cout << "(accumulator " << target.m_acc << ")" << std::endl;
    return 0;
}
