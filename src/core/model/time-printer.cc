/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Lawrence Livermore National Laboratory
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "ns3/log.h"
#include "ns3/time-printer.h"
#include "ns3/simulator.h"  // Now()
#include "ns3/nstime.h"

#include <iomanip>

/**
 * \file
 * \ingroup time
 * ns3::DefaultTimePrinter implementation.
 */

namespace ns3 {

  NS_LOG_COMPONENT_DEFINE ("TimePrinter");

void
DefaultTimePrinter (std::ostream &os)
{
  std::ios_base::fmtflags ff = os.flags (); // Save stream flags
  std::streamsize oldPrecision = os.precision ();
  os << std::fixed;
  switch (Time::GetResolution ())
    {
    case Time::US :    os << std::setprecision (6);   break;
    case Time::NS :    os << std::setprecision (9);   break;
    case Time::PS :    os << std::setprecision (12);  break;
    case Time::FS :    os << std::setprecision (15);  break;
    default :          
      // default C++ precision of 5
      os << std::setprecision (5);
    }
  os << Simulator::Now ().As (Time::S);
  
  os << std::setprecision (oldPrecision);
  os.flags (ff); // Restore stream flags
}

  
} // namespace ns3

