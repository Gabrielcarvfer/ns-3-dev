/*
 * Copyright (c) 2006,2007 INRIA
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef NS3_LOG_H
#define NS3_LOG_H

#include "log-macros-disabled.h"
#include "log-macros-enabled.h"
#include "node-printer.h"
#include "time-printer.h"

#include <iostream>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

/**
 * @file
 * @ingroup logging
 * Debug message logging
 */

/**
 * @ingroup debugging
 * @defgroup logging Logging
 *
 * @brief Logging functions and macros
 *
 * LOG functionality: macros which allow developers to
 * send information to the \c std::clog output stream.
 *
 * All logging messages are disabled by default. To enable selected logging
 * messages, use the ns3::LogComponentEnable
 * function or use the NS_LOG environment variable
 *
 * Use the environment variable NS_LOG to define a ':'-separated list of
 * logging components to enable. For example (using bash syntax),
 * @code
 *   $ NS_LOG="OlsrAgent" ./ns3 run ...
 * @endcode
 * would enable one component at all log levels.
 * @code
 *   $NS_LOG="OlsrAgent:Ipv4L3Protocol" ./ns3 run ...
 * @endcode
 * would enable two components, at all log levels, etc.
 * \c NS_LOG="*" will enable all available log components at all levels.
 *
 * To control more selectively the log levels for each component, use
 * this syntax:
 * @code
 *   $ NS_LOG='Component1=func|warn:Component2=error|debug'
 * @endcode
 * This example would enable the \c func, and \c warn log
 * levels for 'Component1' and the \c error and \c debug log levels
 * for 'Component2'.  The wildcard '*' can be used here as well.  For example
 * \c NS_LOG='*=level_all|prefix' would enable all log levels and prefix all
 * prints with the component and function names.
 *
 * A note on NS_LOG_FUNCTION() and NS_LOG_FUNCTION_NOARGS():
 * generally, use of (at least) NS_LOG_FUNCTION(this) is preferred,
 * with the any function parameters added:
 * @code
 *   NS_LOG_FUNCTION (this << arg1 << args);
 * @endcode
 * Use NS_LOG_FUNCTION_NOARGS() only in static functions with no arguments.
 */
/** @{ */

namespace ns3
{

/**
 *  Logging severity classes and levels.
 */
enum LogLevel
{
    LOG_NONE = 0x00000000, //!< No logging.

    LOG_ERROR = 0x00000001,       //!< Serious error messages only.
    LOG_LEVEL_ERROR = 0x00000001, //!< LOG_ERROR and above.

    LOG_WARN = 0x00000002,       //!< Warning messages.
    LOG_LEVEL_WARN = 0x00000003, //!< LOG_WARN and above.

    LOG_INFO = 0x00000004,       //!< Something happened to change state.
    LOG_LEVEL_INFO = 0x00000007, //!< LOG_INFO and above.

    LOG_FUNCTION = 0x00000008,       //!< Function tracing for non-trivial function calls.
    LOG_LEVEL_FUNCTION = 0x0000000f, //!< LOG_FUNCTION and above.

    LOG_LOGIC = 0x00000010,       //!< Debugging logs for key branches and decisions in a function.
    LOG_LEVEL_LOGIC = 0x0000001f, //!< LOG_LOGIC and above.

    LOG_DEBUG = 0x00000020,       //!< Full voluminous logging to support debugging.
    LOG_LEVEL_DEBUG = 0x0000003f, //!< LOG_DEBUG and above.

    LOG_ALL = 0x0fffffff,    //!< Print everything.
    LOG_LEVEL_ALL = LOG_ALL, //!< Print everything.

    LOG_PREFIX_FUNC = 0x80000000,  //!< Prefix all trace prints with function.
    LOG_PREFIX_TIME = 0x40000000,  //!< Prefix all trace prints with simulation time.
    LOG_PREFIX_NODE = 0x20000000,  //!< Prefix all trace prints with simulation node.
    LOG_PREFIX_LEVEL = 0x10000000, //!< Prefix all trace prints with log level (severity).
    LOG_PREFIX_ALL = 0xf0000000    //!< All prefixes.
};

/**
 * Enable the logging output associated with that log component.
 *
 * The logging output can be later disabled with a call
 * to ns3::LogComponentDisable.
 *
 * Same as running your program with the NS_LOG environment
 * variable set as NS_LOG='name=level'.
 *
 * @param [in] name The log component name.
 * @param [in] level The logging level.
 */
void LogComponentEnable(const std::string& name, LogLevel level);

/**
 * Enable the logging output for all registered log components.
 *
 * Same as running your program with the NS_LOG environment
 * variable set as NS_LOG='*=level'
 *
 * @param [in] level The logging level.
 */
void LogComponentEnableAll(LogLevel level);

/**
 * Disable the logging output associated with that log component.
 *
 * The logging output can be later re-enabled with a call
 * to LogComponentEnable.
 *
 * @param [in] name The log component name.
 * @param [in] level The logging level.
 */
void LogComponentDisable(const std::string& name, LogLevel level);

/**
 * Disable all logging for all components.
 *
 * @param [in] level The logging level.
 */
void LogComponentDisableAll(LogLevel level);

} // namespace ns3

/**
 * Define a Log component with a specific name.
 *
 * This macro should be used at the top of every file in which you want
 * to use the NS_LOG macro. This macro defines a new
 * "log component" which can be later selectively enabled
 * or disabled with the ns3::LogComponentEnable and
 * ns3::LogComponentDisable functions or with the NS_LOG
 * environment variable.
 *
 * LogComponent names should be simple string tokens, _i.e._,
 * "ArfWifiManager", not "ns3::ArfWifiManager".
 *
 * This macro should be placed within namespace ns3.  If functions
 * outside of namespace ns3 require access to logging, the preferred
 * solution is to add the following 'using' directive at file scope,
 * outside of namespace ns3, and after the inclusion of
 * NS_LOG_COMPONENT_DEFINE, such as follows:
 * @code
 *   namespace ns3 {
 *     NS_LOG_COMPONENT_DEFINE ("...");
 *
 *     // Definitions within the ns3 namespace
 *
 *   } // namespace ns3
 *
 *   using ns3::g_log;
 *
 *   // Further definitions outside of the ns3 namespace
 * @endcode
 *
 * @param [in] name The log component name.
 */
#define NS_LOG_COMPONENT_DEFINE(name)                                                              \
    static ns3::LogComponent g_log = ns3::LogComponent(name, __FILE__)

/**
 * Define a logging component with a mask.
 *
 * See LogComponent().
 *
 * @param [in] name The log component name.
 * @param [in] mask The default mask.
 */
#define NS_LOG_COMPONENT_DEFINE_MASK(name, mask)                                                   \
    static ns3::LogComponent g_log = ns3::LogComponent(name, __FILE__, mask)

/**
 * Declare a reference to a Log component.
 *
 * This macro should be used in the declaration of template classes
 * to allow their methods (defined in an header file) to make use of
 * the NS_LOG_* macros. This macro should be used in the private
 * section to prevent subclasses from using the same log component
 * as the base class.
 */
#define NS_LOG_TEMPLATE_DECLARE LogComponent& g_log

/**
 * Initialize a reference to a Log component.
 *
 * This macro should be used in the constructor of template classes
 * to allow their methods (defined in an header file) to make use of
 * the NS_LOG_* macros.
 *
 * @param [in] name The log component name.
 */
#define NS_LOG_TEMPLATE_DEFINE(name) g_log(GetLogComponent(name))

/**
 * Declare and initialize a reference to a Log component.
 *
 * This macro should be used in static template methods to allow their
 * methods (defined in an header file) to make use of the NS_LOG_* macros.
 *
 * @param [in] name The log component name.
 */
#define NS_LOG_STATIC_TEMPLATE_DEFINE(name)                                                        \
    static LogComponent& g_log [[maybe_unused]] = GetLogComponent(name)

/**
 * Use \ref NS_LOG to output a message of level LOG_ERROR.
 *
 * @param [in] msg The message to log.
 */
#define NS_LOG_ERROR(msg) NS_LOG(ns3::LOG_ERROR, msg)

/**
 * Use \ref NS_LOG to output a message of level LOG_WARN.
 *
 * @param [in] msg The message to log.
 */
#define NS_LOG_WARN(msg) NS_LOG(ns3::LOG_WARN, msg)

/**
 * Use \ref NS_LOG to output a message of level LOG_DEBUG.
 *
 * @param [in] msg The message to log.
 */
#define NS_LOG_DEBUG(msg) NS_LOG(ns3::LOG_DEBUG, msg)

/**
 * Use \ref NS_LOG to output a message of level LOG_INFO.
 *
 * @param [in] msg The message to log.
 */
#define NS_LOG_INFO(msg) NS_LOG(ns3::LOG_INFO, msg)

/**
 * Use \ref NS_LOG to output a message of level LOG_LOGIC
 *
 * @param [in] msg The message to log.
 */
#define NS_LOG_LOGIC(msg) NS_LOG(ns3::LOG_LOGIC, msg)

namespace ns3
{

/**
 * Print the list of logging messages available.
 * Same as running your program with the NS_LOG environment
 * variable set as NS_LOG=print-list
 */
void LogComponentPrintList();

/**
 * Set the TimePrinter function to be used
 * to prepend log messages with the simulation time.
 *
 * The default is DefaultTimePrinter().
 *
 * @param [in] lp The TimePrinter function.
 */
void LogSetTimePrinter(TimePrinter lp);
/**
 * Get the LogTimePrinter function currently in use.
 * @returns The current LogTimePrinter function.
 */
TimePrinter LogGetTimePrinter();

/**
 * Set the LogNodePrinter function to be used
 * to prepend log messages with the node id.
 *
 * The default is DefaultNodePrinter().
 *
 * @param [in] np The LogNodePrinter function.
 */
void LogSetNodePrinter(NodePrinter np);
/**
 * Get the LogNodePrinter function currently in use.
 * @returns The current LogNodePrinter function.
 */
NodePrinter LogGetNodePrinter();

/**
 * A single log component configuration.
 */
class LogComponent
{
  public:
    /**
     * Constructor.
     *
     * @param [in] name The user-visible name for this component.
     * @param [in] file The source code file which defined this LogComponent.
     * @param [in] mask LogLevels blocked for this LogComponent.  Blocking
     *                  a log level helps prevent recursion by logging in
     *                  functions which help implement the logging facility.
     */
    LogComponent(const std::string& name, const std::string& file, const LogLevel mask = LOG_NONE);

    /**
     * Check if this LogComponent is enabled for \c level
     *
     * @param [in] level The level to check for.
     * @return \c true if we are enabled at \c level.
     *
     * @internal
     * This function is defined in the header to enable inlining for better performance. See:
     * https://gitlab.com/nsnam/ns-3-dev/-/merge_requests/2448#note_2527898962
     */
    bool IsEnabled(const LogLevel level) const
    {
        return level & m_levels;
    }

    /**
     * Check if all levels are disabled.
     *
     * @return \c true if all levels are disabled.
     */
    bool IsNoneEnabled() const;
    /**
     * Enable this LogComponent at \c level
     *
     * @param [in] level The LogLevel to enable.
     */
    void Enable(const LogLevel level);
    /**
     * Disable logging at \c level for this LogComponent.
     *
     * @param [in] level The LogLevel to disable.
     */
    void Disable(const LogLevel level);
    /**
     * Get the name of this LogComponent.
     *
     * @return The name of this LogComponent.
     */
    std::string Name() const;
    /**
     * Get the compilation unit defining this LogComponent.
     * @returns The file name.
     */
    std::string File() const;
    /**
     * Get the string label for the given LogLevel.
     *
     * @param [in] level The LogLevel to get the label for.
     * @return The string label for \c level.
     */
    static std::string GetLevelLabel(const LogLevel level);
    /**
     * Prevent the enabling of a specific LogLevel.
     *
     * @param [in] level The LogLevel to block.
     */
    void SetMask(const LogLevel level);

    /**
     * LogComponent name map.
     *
     * @internal
     * This should really be considered an internal API.
     * It is exposed here to allow print-introspected-doxygen.cc
     * to generate a list of all LogComponents.
     */
    using ComponentList = std::unordered_map<std::string, LogComponent*>;

    /**
     * Get the list of LogComponents.
     *
     * @internal
     * This should really be considered an internal API.
     * It is exposed here to allow print-introspected-doxygen.cc
     * to generate a list of all LogComponents.
     *
     * @returns The list of LogComponents.
     */
    static ComponentList* GetComponentList();

  private:
    /**
     * Parse the `NS_LOG` environment variable for options relating to this
     * LogComponent.
     */
    void EnvVarCheck();

    int32_t m_levels;   //!< Enabled LogLevels.
    int32_t m_mask;     //!< Blocked LogLevels.
    std::string m_name; //!< LogComponent name.
    std::string m_file; //!< File defining this LogComponent.

    // end of class LogComponent
};

/**
 * Get the LogComponent registered with the given name.
 *
 * @param [in] name The name of the LogComponent.
 * @return a reference to the requested LogComponent
 */
LogComponent& GetLogComponent(const std::string name);

/**
 * Insert `, ` when streaming function arguments.
 */
class ParameterLogger
{
  public:
    /**
     * Constructor.
     *
     * @param [in] os Underlying output stream.
     */
    ParameterLogger(std::ostream& os);

    /**
     * Write a function parameter on the output stream,
     * separating parameters after the first by `,` strings.
     *
     * @param [in] param The function parameter.
     * @return This ParameterLogger, so it's chainable.
     */
    template <typename T>
    ParameterLogger& operator<<(const T& param);

    /**
     * Overload for vectors, to print each element.
     *
     * @param [in] vector The vector of parameters
     * @return This ParameterLogger, so it's chainable.
     */
    template <typename T>
    ParameterLogger& operator<<(const std::vector<T>& vector);

  private:
    /** Add `, ` before every parameter after the first. */
    void CommaRest();

    bool m_first{true}; //!< First argument flag, doesn't get `, `.
    std::ostream& m_os; //!< Underlying output stream.

    // end of class ParameterLogger
};

template <typename T>
ParameterLogger&
ParameterLogger::operator<<(const T& param)
{
    CommaRest();

    if constexpr (std::is_convertible_v<T, std::string>)
    {
        m_os << "\"" << param << "\"";
    }
    else if constexpr (std::is_arithmetic_v<T>)
    {
        // Use + unary operator to cast uint8_t / int8_t to uint32_t / int32_t, respectively
        m_os << +param;
    }
    else if constexpr (std::is_pointer_v<T>)
    {
        m_os << static_cast<const void*>(&param);
    }
    else
    {
        m_os << param;
    }

    return *this;
}

template <typename T>
ParameterLogger&
ParameterLogger::operator<<(const std::vector<T>& vector)
{
    for (const auto& i : vector)
    {
        *this << i;
    }
    return *this;
}

} // namespace ns3

/**@}*/ // \ingroup logging

#endif /* NS3_LOG_H */
