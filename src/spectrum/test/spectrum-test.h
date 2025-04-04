/*
 * Copyright (c) 2009 University of Washington
 * Copyright (c) 2011 CTTC
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Nicola Baldo <nbaldo@cttc.es>
 * part of the code copied from test.h
 */

#ifndef SPECTRUM_TEST_H
#define SPECTRUM_TEST_H

#include "ns3/spectrum-value.h"
#include "ns3/test.h"

/**
 * @ingroup spectrum
 * @defgroup spectrum-test Spectrum module tests
 */

/**
 * @ingroup spectrum-tests
 *
 * @brief Test if two SpectrumModel instances are equal within a given tolerance.
 *
 * This test compares component-by-component the two SpectrumModel
 * instances; if any pair of components differs by more than the given
 * tolerance, the test fails.
 *
 * @param actual the actual value obtained by the simulator
 * @param expected the expected value obtained by off-line calculations
 * @param tol the tolerance
 * @param msg the message to print if the test fails
 *
 */
#define NS_TEST_ASSERT_MSG_SPECTRUM_MODEL_EQ_TOL(actual, expected, tol, msg)                       \
    do                                                                                             \
    {                                                                                              \
        auto i = (actual).Begin();                                                                 \
        auto j = (expected).Begin();                                                               \
        uint32_t k = 0;                                                                            \
        while (i != (actual).End() && j != (expected).End())                                       \
        {                                                                                          \
            if ((i->fl > j->fl + (tol)) || (i->fl < j->fl - (tol)) || (i->fc > j->fc + (tol)) ||   \
                (i->fc < j->fc - (tol)) || (i->fh > j->fh + (tol)) || (i->fh < j->fh - (tol)))     \
            {                                                                                      \
                ASSERT_ON_FAILURE;                                                                 \
                std::ostringstream indexStream;                                                    \
                indexStream << "[" << k << "]";                                                    \
                std::ostringstream msgStream;                                                      \
                msgStream << (msg);                                                                \
                std::ostringstream actualStream;                                                   \
                actualStream << i->fl << " <-- " << i->fc << " --> " << i->fh;                     \
                std::ostringstream expectedStream;                                                 \
                expectedStream << j->fl << " <-- " << j->fc << " --> " << j->fh;                   \
                ReportTestFailure(std::string(#actual) + indexStream.str() +                       \
                                      " == " + std::string(#expected) + indexStream.str(),         \
                                  actualStream.str(),                                              \
                                  expectedStream.str(),                                            \
                                  msgStream.str(),                                                 \
                                  (__FILE__),                                                      \
                                  (__LINE__));                                                     \
                CONTINUE_ON_FAILURE;                                                               \
            }                                                                                      \
            ++i;                                                                                   \
            ++j;                                                                                   \
            ++k;                                                                                   \
        }                                                                                          \
        if (i != (actual).End() || j != (expected).End())                                          \
        {                                                                                          \
            std::ostringstream msgStream;                                                          \
            msgStream << (msg);                                                                    \
            std::ostringstream actualStream;                                                       \
            actualStream << (i != (actual).End());                                                 \
            std::ostringstream expectedStream;                                                     \
            expectedStream << (j != (expected).End());                                             \
            ReportTestFailure("Bands::iterator == End ()",                                         \
                              actualStream.str(),                                                  \
                              expectedStream.str(),                                                \
                              msgStream.str(),                                                     \
                              (__FILE__),                                                          \
                              (__LINE__));                                                         \
        }                                                                                          \
    } while (false);

/**
 * @ingroup spectrum-tests
 *
 * @brief Test if two SpectrumValue instances are equal within a given tolerance.
 *
 * This test compares component-by-component the two SpectrumValue
 * instances; if any pair of components differs by more than the given
 * tolerance, the test fails.
 *
 * @param actual the actual value obtained by the simulator
 * @param expected the expected value obtained by off-line calculations
 * @param tol the tolerance
 * @param msg the message to print if the test fails
 *
 */
#define NS_TEST_ASSERT_MSG_SPECTRUM_VALUE_EQ_TOL(actual, expected, tol, msg)                       \
    do                                                                                             \
    {                                                                                              \
        auto i = (actual).ConstValuesBegin();                                                      \
        auto j = (expected).ConstValuesBegin();                                                    \
        uint32_t k = 0;                                                                            \
        while (i != (actual).ConstValuesEnd() && j != (expected).ConstValuesEnd())                 \
        {                                                                                          \
            if ((*i) > (*j) + (tol) || (*i) < (*j) - (tol))                                        \
            {                                                                                      \
                ASSERT_ON_FAILURE;                                                                 \
                std::ostringstream indexStream;                                                    \
                indexStream << "[" << k << "]";                                                    \
                std::ostringstream msgStream;                                                      \
                msgStream << msg;                                                                  \
                std::ostringstream actualStream;                                                   \
                actualStream << actual;                                                            \
                std::ostringstream expectedStream;                                                 \
                expectedStream << expected;                                                        \
                ReportTestFailure(std::string(#actual) + indexStream.str() +                       \
                                      " == " + std::string(#expected) + indexStream.str(),         \
                                  actualStream.str(),                                              \
                                  expectedStream.str(),                                            \
                                  msgStream.str(),                                                 \
                                  __FILE__,                                                        \
                                  __LINE__);                                                       \
                CONTINUE_ON_FAILURE;                                                               \
            }                                                                                      \
            ++i;                                                                                   \
            ++j;                                                                                   \
            ++k;                                                                                   \
        }                                                                                          \
        if (i != (actual).ConstValuesEnd() || j != (expected).ConstValuesEnd())                    \
        {                                                                                          \
            std::ostringstream msgStream;                                                          \
            msgStream << (msg);                                                                    \
            std::ostringstream actualStream;                                                       \
            actualStream << (i != (actual).ConstValuesEnd());                                      \
            std::ostringstream expectedStream;                                                     \
            expectedStream << (j != (expected).ConstValuesEnd());                                  \
            ReportTestFailure("Values::const_iterator == ConstValuesEnd ()",                       \
                              actualStream.str(),                                                  \
                              expectedStream.str(),                                                \
                              msgStream.str(),                                                     \
                              (__FILE__),                                                          \
                              (__LINE__));                                                         \
        }                                                                                          \
    } while (false);

#endif // SPECTRUM_TEST_H
