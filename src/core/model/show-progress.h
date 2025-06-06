/*
 * Copyright (c) 2017 Lawrence Livermore National Laboratory
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Gustavo Carneiro <gjc@inescporto.pt>
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#ifndef SHOW_PROGRESS_H
#define SHOW_PROGRESS_H

/**
 * @file
 * @ingroup core
 * ns3::ShowProgress declaration.
 */

#include "event-id.h"
#include "nstime.h"
#include "system-wall-clock-ms.h"
#include "system-wall-clock-timestamp.h"
#include "time-printer.h"

#include <iostream>

namespace ns3
{

/**
 * @ingroup core
 * @ingroup debugging
 *
 * Periodically print a status message indicating simulator progress.
 *
 * Print a status message showing the simulator time and
 * execution speed, relative to wall clock time, as well as the number
 * of events executed each interval of wall clock time.
 *
 * The output interval is based on wall clock time, rather than simulation
 * time, to avoid too many events for fast simulations, and too long of
 * a reporting interval for very slow simulations.
 *
 * The target update interval (and output stream) can be configured
 * at construction.  The default output stream, if unspecified, is cout.
 *
 * Example usage:
 *
 * @code
 *     int main (int arg, char ** argv)
 *     {
 *       // Create your model
 *
 *       ShowProgress progress (Seconds (10), std::cerr);
 *       Simulator::Run ();
 *       Simulator::Destroy ();
 *     }
 * @endcode
 *
 * This generates output similar to the following:
 *
 * @code
 *     Start wall clock: Tue May 19 10:24:07 2020
 *     +17.179869183s (   4.937x real time) 29559 events processed
 *     +25.769803775s (   4.965x real time) 45179 events processed
 *     +51.539607551s (   8.810x real time) 49421 events processed
 *     +90.324463739s (   7.607x real time) 58021 events processed
 *     +129.770882188s (   7.576x real time) 53850 events processed
 *     +170.958751090s (   8.321x real time) 56473 events processed
 *     +194.339562435s (  12.776x real time) 30957 events processed
 *     End wall clock:  Tue May 19 10:24:23 2020
 *     Elapsed wall clock: 16s
 * @endcode
 *
 * A more extensive example of use is provided in sample-show-progress.cc.
 *
 * Based on a python version by Gustavo Carneiro <gjcarneiro@gmail.com>,
 * as released here:
 *
 * https://mailman.isi.edu/pipermail/ns-developers/2009-January/005039.html
 */
class ShowProgress
{
  public:
    /**
     * Constructor.
     * @param [in] interval The target wallclock interval to show progress.
     * @param [in] os The stream to print on.
     */
    ShowProgress(const Time interval = Seconds(1), std::ostream& os = std::cout);

    /** Destructor. */
    ~ShowProgress();

    /**
     * Set the target update interval, in wallclock time.
     * @param [in] interval The target wallclock interval to show progress.
     */
    void SetInterval(const Time interval);

    /**
     * Set the TimePrinter function to be used
     * to prepend progress messages with the simulation time.
     *
     * The default is DefaultTimePrinter().
     *
     * @param [in] lp The TimePrinter function.
     */
    void SetTimePrinter(TimePrinter lp);

    /**
     * Set the output stream to show progress on.
     * @param [in] os The output stream; defaults to std::cerr.
     */
    void SetStream(std::ostream& os);

    /**
     * Set verbose mode to print real and virtual time intervals.
     *
     * @param [in] verbose \c true to turn on verbose mode
     */
    void SetVerbose(bool verbose);

  private:
    /**
     * Start the elapsed wallclock timestamp and print the start time.
     * This is triggered by the constructor.
     */
    void Start();

    /**
     * Stop the elapsed wallclock timestamp and print the total elapsed time.
     * This is triggered by the destructor.
     */
    void Stop();

    /**
     * Schedule the next CheckProgress.
     */
    void ScheduleCheckProgress();

    /**
     * Check on execution progress.
     * This function is executed periodically, updates the internal
     * state on rate of progress, and decides if it's time to generate
     * output.
     */
    void CheckProgress();

    /**
     * Show execution progress.
     * This function actually generates output, when directed by CheckProgress().
     * @param [in] nEvents The actual number of events processed since the last
     *             progress output.
     * @param [in] ratio The current ratio of elapsed wall clock time to the
     *             target update interval.
     * @param [in] speed The execution speed relative to wall clock time.
     */
    void GiveFeedback(uint64_t nEvents, int64x64_t ratio, int64x64_t speed);

    /**
     * Hysteresis factor.
     * @see Feedback()
     */
    static const int64x64_t HYSTERESIS;
    /**
     * Maximum growth factor.
     * @see Feedback()
     */
    static const int64x64_t MAXGAIN;

    SystemWallClockMs m_timer;        //!< Wallclock timer
    SystemWallClockTimestamp m_stamp; //!< Elapsed wallclock time.
    Time m_elapsed;                   //!< Total elapsed wallclock time since last update.
    Time m_interval;                  //!< The target update interval, in wallclock time
    Time m_vtime;                     //!< The virtual time interval.
    EventId m_event;                  //!< The next progress event.
    uint64_t m_eventCount;            //!< Simulator event count

    TimePrinter m_printer; //!< The TimePrinter to use
    std::ostream* m_os;    //!< The output stream to use.
    bool m_verbose;        //!< Verbose mode flag
    uint64_t m_repCount;   //!< Number of CheckProgress events
};

} // namespace ns3

#endif /* SHOW_PROGRESS_H */
