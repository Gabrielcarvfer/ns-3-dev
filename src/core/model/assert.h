/*
 * Copyright (c) 2006 INRIA, 2010 NICTA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *         Quincy Tse <quincy.tse@nicta.com.au>
 */
#ifndef NS_ASSERT_H
#define NS_ASSERT_H

/**
 * @file
 * @ingroup assert
 * NS_ASSERT() and NS_ASSERT_MSG() macro definitions
 */

/**
 * @ingroup debugging
 * @defgroup assert Assertions
 *
 * @brief Assert functions and macros
 *
 * The assert macros are used to verify
 * at runtime that a certain condition is true. If it is
 * not true, the program halts. These checks are built
 * into the program only in debugging builds. They are
 * removed in optimized builds.
 *
 * These macro are intended to check certain conditions
 * to be true. Do not put code that also have side effects
 * that your program relies on (eg. code that advances
 * an iterator and return false at end of file) because
 * the code will not be executed on release builds!!
 *
 * If assertion-style checks are required for release
 * builds, use NS_ABORT_UNLESS and NS_ABORT_MSG_UNLESS.
 */

#ifdef NS3_ASSERT_ENABLE

#include "fatal-error.h"

#include <iostream>

/**
 * @ingroup assert
 *
 * At runtime, in debugging builds, if this condition is not
 * true, the program prints the source file, line number and
 * unverified condition and halts by calling std::terminate
 *
 * @param [in] condition Condition to verify.
 */
#define NS_ASSERT(condition)                                                                       \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::cerr << "NS_ASSERT failed, cond=\"" << #condition << "\", ";                      \
            NS_FATAL_ERROR_NO_MSG();                                                               \
        }                                                                                          \
    } while (false)

/**
 * @ingroup assert
 *
 * At runtime, in debugging builds, if this condition is not
 * true, the program prints the message to output and
 * halts by calling std::terminate.
 *
 * @param [in] condition Condition to verify.
 * @param [in] message Message to output
 */
#define NS_ASSERT_MSG(condition, message)                                                          \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::cerr << "NS_ASSERT failed, cond=\"" << #condition << "\", ";                      \
            NS_FATAL_ERROR(message);                                                               \
        }                                                                                          \
    } while (false)

#else /* NS3_ASSERT_ENABLE */

// NOTE: The no-op macros are not inserted into the final code.
// However, the use of sizeof() allows the compiler to silently check if the condition is
// syntactically valid.

#define NS_ASSERT(condition)                                                                       \
    do                                                                                             \
    {                                                                                              \
        (void)sizeof(condition);                                                                   \
    } while (false)

#define NS_ASSERT_MSG(condition, message)                                                          \
    do                                                                                             \
    {                                                                                              \
        (void)sizeof(condition);                                                                   \
    } while (false)

#endif /* NS3_ASSERT_ENABLE */

#endif /* ASSERT_H */
