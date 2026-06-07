/*
 * Copyright (c) 2010 The Boeing Company
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Tom Goff <thomas.goff@boeing.com>
 */

// winsock2.h must come before any header that might pull in windows.h
// (ns-3 headers include ostream which transitively includes windows.h on
// some toolchains).  WIN32_LEAN_AND_MEAN prevents winsock.h re-inclusion.
#define WIN32_LEAN_AND_MEAN
#include "fatal-error.h"
#include "fd-reader.h"
#include "log.h"
#include "simple-ref-count.h"
#include "simulator.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <io.h>
#include <thread>
#include <windows.h>
#include <winsock2.h>

/**
 * @file
 * @ingroup system
 * ns3::FdReader implementation.
 */

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("FdReader");

// conditional compilation to avoid Doxygen errors
#ifdef _WIN32
bool FdReader::winsock_initialized = false;
#endif

FdReader::FdReader()
    : m_fd(-1),
      m_stop(false),
      m_destroyEvent(),
      m_eventsignal(nullptr)
{
    NS_LOG_FUNCTION(this);
    m_evpipe[0] = -1;
    m_evpipe[1] = -1;
}

FdReader::~FdReader()
{
    NS_LOG_FUNCTION(this);
    Stop();
}

void
FdReader::Start(int fd, Callback<void, uint8_t*, ssize_t> readCallback)
{
    NS_LOG_FUNCTION(this << fd << &readCallback);
    int tmp;

    if (!winsock_initialized)
    {
        WSADATA wsaData;
        tmp = WSAStartup(MAKEWORD(2, 2), &wsaData);
        NS_ASSERT_MSG(tmp == NO_ERROR, "Error at WSAStartup()");
        winsock_initialized = true;
    }

    NS_ASSERT_MSG(!m_readThread.joinable(), "read thread already exists");

    m_fd = fd;
    m_readCallback = readCallback;

    //
    // We're going to spin up a thread soon, so we need to make sure we have
    // a way to tear down that thread when the simulation stops.  Do this by
    // scheduling a "destroy time" method to make sure the thread exits before
    // proceeding.
    //
    if (!m_destroyEvent.IsPending())
    {
        // hold a reference to ensure that this object is not
        // deallocated before the destroy-time event fires
        this->Ref();
        m_destroyEvent = Simulator::ScheduleDestroy(&FdReader::DestroyEvent, this);
    }

    //
    // Now spin up a thread to read from the fd.  On Windows, the discrete-event
    // simulator can finish processing 0..N ms of simulation time before the OS
    // ever grants the new thread a CPU time slice.  We use a one-shot Windows
    // Event so that Start() (which executes on the simulator's main thread, i.e.
    // inside a scheduled event callback) blocks until Run() has completed its
    // first loop iteration.  If data was already in the pipe, the thread will
    // have called ReceiveCallback / ScheduleWithContext before signaling, so the
    // simulator will find the forwarded packet event in the queue when it
    // resumes.
    //
    NS_LOG_LOGIC("Spinning up read thread");

    m_eventsignal = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    m_readThread = std::thread(&FdReader::Run, this);
    WaitForSingleObject(m_eventsignal, INFINITE);
    CloseHandle(m_eventsignal);
    m_eventsignal = nullptr;
}

void
FdReader::DestroyEvent()
{
    NS_LOG_FUNCTION(this);
    Stop();
    this->Unref();
}

void
FdReader::Stop()
{
    NS_LOG_FUNCTION(this);
    m_stop = true;

    if (m_readThread.joinable())
    {
        m_readThread.join();
    }

    m_fd = -1;
    m_readCallback.Nullify();
    m_stop = false;
}

// This runs in a separate thread
void
FdReader::Run()
{
    NS_LOG_FUNCTION(this);

    // On Windows, Winsock's select() only accepts actual SOCKETs — it cannot
    // monitor POSIX file descriptors such as those created by _pipe().
    // Anonymous pipes do not support WaitForMultipleObjects either.
    // The only portable way to check readiness of an anonymous pipe handle
    // without blocking is PeekNamedPipe().  We poll at 1 ms intervals, which
    // is fine for simulation purposes and keeps CPU usage negligible.
    HANDLE hFd = (HANDLE)_get_osfhandle(m_fd);

    // Signal Start() exactly once after our first loop iteration so it can
    // unblock.  For the pipe case with pre-queued data this happens after the
    // first ReceiveCallback / ScheduleWithContext, so the simulator sees the
    // event immediately.  For the non-pipe (TAP device) case we signal before
    // blocking in DoRead so Start() is not held up indefinitely.
    bool startSignaled = false;
    auto signalStart = [&]() {
        if (!startSignaled && m_eventsignal)
        {
            SetEvent(m_eventsignal);
            startSignaled = true;
        }
    };

    for (;;)
    {
        if (m_stop)
        {
            signalStart(); // unblock Start() if we exit early
            break;
        }

        DWORD avail = 0;
        BOOL ok = PeekNamedPipe(hFd, nullptr, 0, nullptr, &avail, nullptr);
        if (!ok)
        {
            // Not a pipe (e.g., TAP/TUN device handle).  Signal Start() now so
            // it is not blocked while we wait for the first frame.
            signalStart();
            FdReader::Data data = DoRead();
            if (data.m_len == 0)
            {
                break;
            }
            if (data.m_len > 0)
            {
                m_readCallback(data.m_buf, data.m_len);
            }
        }
        else if (avail > 0)
        {
            FdReader::Data data = DoRead();
            if (data.m_len == 0)
            {
                break;
            }
            if (data.m_len > 0)
            {
                m_readCallback(data.m_buf, data.m_len);
            }
            // Signal after callback so any ScheduleWithContext call is visible
            // to the simulator before Start() returns.
            signalStart();
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            signalStart();
        }
    }
}

} // namespace ns3
