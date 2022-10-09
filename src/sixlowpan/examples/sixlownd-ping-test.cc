
/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze, Italy
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
 * Author: Adnan Rashid <adnan.rashid@unifi.it>
 *         Tommaso Pecorella <tommaso.pecorella@unifi.it>.
 *
 *  Sixlowpan Mesh-under network topology
 *
 *  n0---------n1---------n2
 *  |          |          |
 *  |          |          |      6LBR = n4
 *  n3---------n4---------n5     6LN = rest nodes
 *  |          |          |
 *  |          |          |
 *  n6---------n7---------n8
 *
 * ./waf --run "scratch/udp-test.cc --Mesh --Position=Grid NumberOfNodes=9 --StopTime=200 --Interval=1"
 *
 */
#include <fstream>
#include <map>
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/ipv6-static-routing-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/log.h"
#include "ns3/ipv6-routing-table-entry.h"
#include "ns3/sixlowpan-module.h"
#include "ns3/lr-wpan-module.h"
#include "ns3/energy-module.h"
#include "ns3/wifi-module.h"
#include "ns3/gnuplot.h"
#include "ns3/applications-module.h"

using namespace ns3;

uint32_t pktCount = 0;
uint64_t pktTotalSize = 0;
uint32_t ackCount = 0;
uint64_t ackTotalSize = 0;
uint32_t unkCount = 0;
uint64_t unkTotalSize = 0;

std::map<uint8_t, uint32_t> icmpTypeCount;
uint32_t udpCount = 0;
uint32_t otherL4Count = 0;

uint32_t unicastcount = 0;
uint32_t multicastcount = 0;

void
PrintResults (Time interval)
{
  std::cout << Now ().GetSeconds () << "\t";
  std::cout << pktCount << "\t" << pktTotalSize << "\t";
  std::cout << ackCount << "\t" << ackTotalSize << "\t";
  std::cout << unkCount << "\t" << unkTotalSize << "\t";
  std::cout << unicastcount << "\t" << multicastcount << std::endl;
  //	std::cout << udpCount << "\t" <<otherL4Count << std::endl;

  pktCount = 0;
  pktTotalSize = 0;
  ackCount = 0;
  ackTotalSize = 0;
  unkCount = 0;
  unkTotalSize = 0;

  Simulator::Schedule (interval, &PrintResults, interval);
}

void
PhyCallback (std::string path, Ptr<const Packet> packet)
{
  lrwpan::LrWpanMacHeader lrWpanHdr;
  Ptr<Packet> pktCpy = packet->Copy ();
  pktCpy->RemoveHeader (lrWpanHdr);

  if (lrWpanHdr.IsAcknowledgment ())
    {
      ackCount++;
      ackTotalSize += packet->GetSize ();
    }
  else if (lrWpanHdr.IsData ())
    {
      pktCount++;
      pktTotalSize += packet->GetSize ();
    }
  else
    {
      unkCount++;
      unkTotalSize += packet->GetSize ();
      std::cout << *packet << std::endl;
    }
}

// Here you see only the packets that IP did send to 6LoWPAN.
// It can NOT tell you shit about the real packet size after compression, if there has been a fragmentation, etc.
// for that you need the function above.

void
SixLowCallback (std::string path, Ptr<const Packet> packet, Ptr<SixLowPanNetDevice> netDev, uint32_t index)
{
  Ipv6Header ipv6Hdr;
  Ptr<Packet> pktCpy = packet->Copy ();
  pktCpy->RemoveHeader (ipv6Hdr);
  if (ipv6Hdr.GetNextHeader () == UdpL4Protocol::PROT_NUMBER)
    {
      udpCount++;
    }
  else if (ipv6Hdr.GetNextHeader () == Icmpv6L4Protocol::PROT_NUMBER)
    {
      Icmpv6Header icmpHdr;
      pktCpy->RemoveHeader (icmpHdr);
      icmpTypeCount[icmpHdr.GetType ()]++;
    }
  else
    {
      otherL4Count++;
    }

  if (ipv6Hdr.GetDestination () == Ipv6Address::GetAllRoutersMulticast ())
    {
      multicastcount++;
    }
  else
    {
      unicastcount++;
    }
  //std::cout << Now ().As (Time::S) << " ******* "<< packet->GetSize () << " ******* " << *packet << std::endl;
}

int main (int argc, char** argv)
{
  bool useMeshUnder = false;
  //	bool useLLA = false;
  //	bool useGUA = false;
  bool printNeighborCache = false;
  std::string useUdpFrom = "";
  std::string usePingOn = "";
  double stopTime = 100;
  Time interval = Seconds (1);
  std::string position = "Grid";
  uint8_t numberOfNodes = 9;

  CommandLine cmd;
  cmd.AddValue ("Mesh", "Use mesh-under in the network", useMeshUnder);
  //	cmd.AddValue ("Udp", "Send one UDP packet from (6LBR, 6LN, nothing)", useUdpFrom);
  //	cmd.AddValue ("Ping", "Install Ping app on (6LBR, 6LN, nothing)", usePingOn);
  cmd.AddValue ("NeighborCache", "Print the neighbor cache entries", printNeighborCache);
  //	cmd.AddValue ("LLA", "Use link-local addresses for the communication", useLLA);
  //	cmd.AddValue ("GUA", "Use global addresses for the communication", useGUA);
  cmd.AddValue ("StopTime", "Simulation stop time (seconds)", stopTime);
  cmd.AddValue ("Interval", "Sampling interval", interval);
  cmd.AddValue ("Position", "Grid or Circle", position);
  cmd.AddValue ("NumberOfNodes", "Number of nodes", numberOfNodes);
  cmd.Parse (argc, argv);

  if (useMeshUnder)
    {
      Config::SetDefault ("ns3::SixLowPanNetDevice::UseMeshUnder", BooleanValue (true));
    }

  Packet::EnablePrinting ();

#if 0
  LogComponentEnable ("Ping6Application", LOG_LEVEL_ALL);
  LogComponentEnable ("LrWpanMac",LOG_LEVEL_ALL);
  LogComponentEnable ("LrWpanPhy",LOG_LEVEL_ALL);
  LogComponentEnable ("LrWpanNetDevice", LOG_LEVEL_ALL);
  LogComponentEnable ("SixLowPanNetDevice", LOG_LEVEL_ALL);
#endif
  //  LogComponentEnable ("OnOffApplication", LOG_LEVEL_ALL);

  NodeContainer lo_nodes;
  lo_nodes.Create (numberOfNodes);
  uint8_t sixLbrNodeNum;

  if (position == "Grid")
    {
      MobilityHelper mobility;
      mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                     "MinX", DoubleValue (0.0),
                                     "MinY", DoubleValue (0.0),
                                     "DeltaX", DoubleValue (60),
                                     "DeltaY", DoubleValue (60),
                                     "GridWidth", UintegerValue (3),
                                     "LayoutType", StringValue ("RowFirst"));
      mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
      mobility.Install (lo_nodes);
      sixLbrNodeNum = 4;
    }
  else if (position == "Circle")
    {
      MobilityHelper mobility;
      mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

      Ptr<ListPositionAllocator> nodesPositionAlloc = CreateObject<ListPositionAllocator> ();
      sixLbrNodeNum = 0;
      nodesPositionAlloc->Add (Vector (0.0, 0.0, 0.0));
      for (uint8_t index = 1; index < numberOfNodes; index++)
        {
          double x, y;
          x = 90 * sin (2 * M_PI / (numberOfNodes - 1) * (index - 1));
          y = 90 * cos (2 * M_PI / (numberOfNodes - 1) * (index - 1));
          nodesPositionAlloc->Add (Vector (x, y, 0.0));
        }
      mobility.SetPositionAllocator (nodesPositionAlloc);
      mobility.Install (lo_nodes);

    }
  else
    {
      std::cout << "Invalid position type " << position << std::endl;
      exit (0);
    }

  // this means that the registration is valid for 2 days (and the re-registration is performed after 1 day).
  Config::SetDefault ("ns3::SixLowPanNdProtocol::RegistrationLifeTime", UintegerValue (2880));
  // Config::SetDefault ("ns3::Icmpv6L4Protocol::SolicitationJitter", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"));

  LrWpanHelper lrWpanHelper;
  NetDeviceContainer lrwpanDevices = lrWpanHelper.Install (lo_nodes);

  lrWpanHelper.CreateAssociatedPan (lrwpanDevices, 0);

  InternetStackHelper internetv6;
  internetv6.Install (lo_nodes);

  SixLowPanHelper sixlowpan;
  NetDeviceContainer devices = sixlowpan.Install (lrwpanDevices);

  for (int var = 0; var < numberOfNodes; var++)
    {
      if (var == sixLbrNodeNum)
        {
          sixlowpan.InstallSixLowPanNdBorderRouter (devices.Get (var), "2001::");
          sixlowpan.SetAdvertisedPrefix (devices.Get (var), Ipv6Prefix ("2001::", 64));
          sixlowpan.AddAdvertisedContext (devices.Get (var), Ipv6Prefix ("2002::", 64));
        }
      else
        {
          sixlowpan.InstallSixLowPanNdNode (devices.Get (var));
        }
    }


  uint32_t MaxPacketSize = 16;
  uint32_t maxPacketCount = 10;
  Time interPacketInterval = Seconds (1.);

  PingHelper ping6;

  ping6.SetAttribute ("Count", UintegerValue (maxPacketCount));
  ping6.SetAttribute ("Interval", TimeValue (interPacketInterval));
  ping6.SetAttribute ("Size", UintegerValue (MaxPacketSize));

  ApplicationContainer apps;

  if (position == "Grid")
    {
      // 6LBR addresses: "2001::ff:fe00:5" - "fe80::ff:fe00:5"
      ping6.SetAttribute("InterfaceAddress", AddressValue(Ipv6Address("fe80::ff:fe00:5")));
      ping6.SetAttribute("Destination", AddressValue(Ipv6Address("fe80::ff:fe00:1")));
      // ping6.SetLocal ("fe80::ff:fe00:5");
      // ping6.SetRemote ("fe80::ff:fe00:1");

      apps.Add (ping6.Install (lo_nodes.Get (4)));
      apps.Start (Seconds (1.5));
      apps.Stop (Seconds (stopTime - 1));
    }

  else if (position == "Circle")
    {
      // 6LBR addresses: "2001::ff:fe00:1" - "fe80::ff:fe00:1"
      ping6.SetAttribute("InterfaceAddress", AddressValue(Ipv6Address("fe80::ff:fe00:1")));
      ping6.SetAttribute("Destination", AddressValue(Ipv6Address("fe80::ff:fe00:5")));
      // ping6.SetLocal ("fe80::ff:fe00:1");
      // ping6.SetRemote ("fe80::ff:fe00:5");
      apps.Add (ping6.Install (lo_nodes.Get (0)));
      apps.Start (Seconds (1.5));
      apps.Stop (Seconds (stopTime - 1));
    }
  else
    {
      std::cout << "PING: invalid Position\n";
      exit (0);
    }


  AsciiTraceHelper ascii;
  lrWpanHelper.EnableAsciiAll (ascii.CreateFileStream ("ping-test.tr"));
  lrWpanHelper.EnablePcapAll (std::string ("ping-test"), true);

//	if (printNeighborCache)
//	{
//		Ptr<OutputStreamWrapper> neighborStream = Create<OutputStreamWrapper> (&std::cout);
//		for (int x = 0; x < stopTime; ++x)
//		{
//			Ipv6RoutingHelper::PrintNeighborCacheAllEvery (Seconds (x), neighborStream);
//			Ipv6RoutingHelper::PrintRoutingTableAllAt(Seconds (x), neighborStream);
//		}
//	}

  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::lrwpan::LrWpanNetDevice/Phy/PhyTxBegin", MakeCallback (&PhyCallback));
  Config::Connect ("/NodeList/*/DeviceList/*/$ns3::SixLowPanNetDevice/TxPre", MakeCallback (&SixLowCallback));

  Simulator::Schedule (interval, &PrintResults, interval);

  Simulator::Stop (Seconds (stopTime));
  Simulator::Run ();

  Simulator::Destroy ();
}

