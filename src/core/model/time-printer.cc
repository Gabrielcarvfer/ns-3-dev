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
    // Digits after the decimal point when printing seconds, indexed by
    // Time::Unit (Y, D, H, MIN, S, MS, US, NS, PS, FS); the default C++
    // precision of 5 is kept for the coarser resolutions.
    static constexpr int precisions[Time::LAST] = {5, 5, 5, 5, 5, 5, 6, 9, 12, 15};
    const int precision = precisions[Time::GetResolution()];

    // Same output as streaming inSeconds with std::fixed, std::showpos and
    // the resolution-dependent precision, but bypassing the ostream
    // formatting machinery, which is significantly slower.
    double seconds = Simulator::Now().GetSeconds();
    char buf[64];
    char* p = buf;
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
