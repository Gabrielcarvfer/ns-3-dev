/*
 * Copyright (c) 2018 Lawrence Livermore National Laboratory
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "time-printer.h"

#include "assert.h"
#include "log.h"
#include "nstime.h"
#include "simulator.h" // Now()

#include <charconv>
#include <cmath>
#include <iterator>

/**
 * @file
 * @ingroup time
 * ns3::DefaultTimePrinter implementation.
 */

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("TimePrinter");

void
DefaultTimePrinter(std::ostream& os)
{
    int precision;
    switch (Time::GetResolution())
    {
    case Time::US:
        precision = 6;
        break;
    case Time::NS:
        precision = 9;
        break;
    case Time::PS:
        precision = 12;
        break;
    case Time::FS:
        precision = 15;
        break;

    default:
        // default C++ precision of 5
        precision = 5;
    }

    // Create the same Time temporaries as the historical
    // `os << Simulator::Now().As(Time::S)` so the Time marking bookkeeping
    // (and the Time:Mark/Clear logs it emits before the simulation starts)
    // is unchanged.
    Time now = Simulator::Now();
    [[maybe_unused]] TimeWithUnit inSeconds = now.As(Time::S);

    // Same output as streaming inSeconds with std::fixed, std::showpos and
    // the resolution-dependent precision, but bypassing the ostream
    // formatting machinery, which is significantly slower.
    char buf[64];
    char* p = buf;
    double seconds = now.GetSeconds();
    if (!std::signbit(seconds))
    {
        *p++ = '+';
    }
    auto [end, ec] =
        std::to_chars(p, std::end(buf) - 1, seconds, std::chars_format::fixed, precision);
    NS_ASSERT(ec == std::errc());
    *end++ = 's';
    os.write(buf, end - buf);
}

} // namespace ns3
