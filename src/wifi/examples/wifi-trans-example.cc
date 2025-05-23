/*
 * Copyright (c) 2017 Orange Labs
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Rediet <getachew.redieteab@orange.com>
 */

#include "ns3/command-line.h"
#include "ns3/gnuplot.h"
#include "ns3/mobility-helper.h"
#include "ns3/spectrum-analyzer-helper.h"
#include "ns3/spectrum-channel.h"
#include "ns3/spectrum-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/wifi-utils.h"

using namespace ns3;

/**
 * This example (inspired from tv-trans-example) enables to generate the transmitted spectra of
 * Wi-Fi stations, so as to model transmit mask imperfections of OFDM-based Wi-Fi standards.
 * Only one data packet is sent from access point to station (once association has been performed)
 * so as to reduce execution time.
 *
 * A spectrum analyzer is used to measure the transmitted spectra from Wi-Fi stations.
 * The file "spectrum-analyzer-wifi-[standard]-[bandwidth]MHz-sim-2-0.tr" contains its
 * output post simulation and use it to plot transmitted spectra with Gnuplot.
 *
 * The wifi-trans-example.sh script runs this example for all combinations, plots transmitted
 * spectra, and puts resulting png images in wifi-trans-results folder.
 */

void
SendPacket(Ptr<NetDevice> sourceDevice, Address& destination)
{
    Ptr<Packet> pkt = Create<Packet>(100); // dummy bytes of data
    sourceDevice->Send(pkt, destination, 0);
}

int
main(int argc, char** argv)
{
    std::string standardStr = "11a";
    MHz_u bw{20};
    dBm_u pow{23};
    bool verbose = false;
    CommandLine cmd(__FILE__);
    cmd.AddValue("standard",
                 "OFDM-based Wi-Fi standard [11a, 11p_10MHZ, 11p_5MHZ, 11n_2_4GHZ, 11n_5GHZ, 11ac, "
                 "11ax_2_4GHZ, 11ax_5GHZ]",
                 standardStr);
    cmd.AddValue("bw", "Bandwidth (consistent with standard, in MHz)", bw);
    cmd.AddValue("txPower", "Transmit power (dBm)", pow);
    cmd.AddValue("verbose",
                 "Display log messages for WifiSpectrumValueHelper and SpectrumWifiPhy",
                 verbose);
    cmd.Parse(argc, argv);

    WifiHelper wifi;
    Ssid ssid;
    std::string dataRate;
    MHz_u freq;
    Time dataStartTime =
        MicroSeconds(800); // leaving enough time for beacon and association procedure
    Time dataDuration =
        MicroSeconds(300); // leaving enough time for data transfer (+ acknowledgment)
    WifiStandard standard{WifiStandard::WIFI_STANDARD_UNSPECIFIED};
    WifiPhyBand phyBand{WIFI_PHY_BAND_UNSPECIFIED};
    std::string channelBand;
    if (standardStr == "11a")
    {
        standard = WIFI_STANDARD_80211a;
        phyBand = WIFI_PHY_BAND_5GHZ;
        channelBand = "BAND_5GHZ";
        ssid = Ssid("ns380211a");
        dataRate = "OfdmRate6Mbps";
        freq = MHz_u{5180};
        if (bw != MHz_u{20})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11p_10MHZ")
    {
        standard = WIFI_STANDARD_80211p;
        phyBand = WIFI_PHY_BAND_5GHZ;
        channelBand = "BAND_5GHZ";
        ssid = Ssid("ns380211p_10MHZ");
        dataRate = "OfdmRate3MbpsBW10MHz";
        freq = MHz_u{5860};
        dataStartTime = MicroSeconds(1400);
        dataDuration = MicroSeconds(600);
        if (bw != MHz_u{10})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11p_5MHZ")
    {
        standard = WIFI_STANDARD_80211p;
        phyBand = WIFI_PHY_BAND_5GHZ;
        channelBand = "BAND_5GHZ";
        ssid = Ssid("ns380211p_5MHZ");
        dataRate = "OfdmRate1_5MbpsBW5MHz";
        freq = MHz_u{5860};
        dataStartTime = MicroSeconds(2500);
        dataDuration = MicroSeconds(1200);
        if (bw != MHz_u{5})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11n_2_4GHZ")
    {
        standard = WIFI_STANDARD_80211n;
        phyBand = WIFI_PHY_BAND_2_4GHZ;
        channelBand = "BAND_2_4GHZ";
        ssid = Ssid("ns380211n_2_4GHZ");
        dataRate = "HtMcs0";
        freq = MHz_u{2402} + (bw / 2); // so as to have 2412/2422 for 20/40
        dataStartTime = MicroSeconds(4700);
        dataDuration = MicroSeconds(400);
        if (bw != MHz_u{20} && bw != MHz_u{40})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11n_5GHZ")
    {
        standard = WIFI_STANDARD_80211n;
        phyBand = WIFI_PHY_BAND_5GHZ;
        channelBand = "BAND_5GHZ";
        ssid = Ssid("ns380211n_5GHZ");
        dataRate = "HtMcs0";
        freq = MHz_u{5170} + (bw / 2); // so as to have 5180/5190 for 20/40
        dataStartTime = MicroSeconds(1000);
        if (bw != MHz_u{20} && bw != MHz_u{40})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11ac")
    {
        standard = WIFI_STANDARD_80211ac;
        phyBand = WIFI_PHY_BAND_5GHZ;
        channelBand = "BAND_5GHZ";
        ssid = Ssid("ns380211ac");
        dataRate = "VhtMcs0";
        freq = MHz_u{5170} + (bw / 2); // so as to have 5180/5190/5210/5250 for 20/40/80/160
        dataStartTime = MicroSeconds(1100);
        dataDuration += MicroSeconds(400); // account for ADDBA procedure
        if (bw != MHz_u{20} && bw != MHz_u{40} && bw != MHz_u{80} && bw != MHz_u{160})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11ax_2_4GHZ")
    {
        standard = WIFI_STANDARD_80211ax;
        phyBand = WIFI_PHY_BAND_2_4GHZ;
        channelBand = "BAND_2_4GHZ";
        ssid = Ssid("ns380211ax_2_4GHZ");
        dataRate = "HeMcs0";
        freq = MHz_u{2402} + (bw / 2); // so as to have 2412/2422/2442 for 20/40/80
        dataStartTime = MicroSeconds(5500);
        dataDuration += MicroSeconds(2000); // account for ADDBA procedure
        if (bw != MHz_u{20} && bw != MHz_u{40} && bw != MHz_u{80})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else if (standardStr == "11ax_5GHZ")
    {
        standard = WIFI_STANDARD_80211ax;
        phyBand = WIFI_PHY_BAND_5GHZ;
        channelBand = "BAND_5GHZ";
        ssid = Ssid("ns380211ax_5GHZ");
        dataRate = "HeMcs0";
        freq = MHz_u{5170} + (bw / 2); // so as to have 5180/5190/5210/5250 for 20/40/80/160
        dataStartTime = MicroSeconds(1200);
        dataDuration += MicroSeconds(500); // account for ADDBA procedure
        if (bw != MHz_u{20} && bw != MHz_u{40} && bw != MHz_u{80} && bw != MHz_u{160})
        {
            std::cout << "Bandwidth is not compatible with standard" << std::endl;
            return 1;
        }
    }
    else
    {
        std::cout << "Unknown OFDM standard (please refer to the listed possible values)"
                  << std::endl;
        return 1;
    }

    if (verbose)
    {
        LogComponentEnableAll(LOG_PREFIX_ALL);
        LogComponentEnable("WifiSpectrumValueHelper", LOG_LEVEL_ALL);
        LogComponentEnable("SpectrumWifiPhy", LOG_LEVEL_ALL);
    }

    /* nodes and positions */
    NodeContainer wifiNodes;
    NodeContainer spectrumAnalyzerNodes;
    NodeContainer allNodes;
    wifiNodes.Create(2);
    spectrumAnalyzerNodes.Create(1);
    allNodes.Add(wifiNodes);
    allNodes.Add(spectrumAnalyzerNodes);
    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    wifiApNode.Add(wifiNodes.Get(0));
    wifiStaNode.Add(wifiNodes.Get(1));

    /* channel and propagation */
    SpectrumChannelHelper channelHelper = SpectrumChannelHelper::Default();
    channelHelper.SetChannel("ns3::MultiModelSpectrumChannel");
    // constant path loss added just to show capability to set different propagation loss models
    // FriisSpectrumPropagationLossModel already added by default in SpectrumChannelHelper
    channelHelper.AddSpectrumPropagationLoss("ns3::ConstantSpectrumPropagationLossModel");
    Ptr<SpectrumChannel> channel = channelHelper.Create();

    /* Wi-Fi transmitter setup */

    SpectrumWifiPhyHelper spectrumPhy;
    spectrumPhy.SetChannel(channel);
    spectrumPhy.SetErrorRateModel("ns3::NistErrorRateModel");
    spectrumPhy.Set("TxPowerStart", DoubleValue(pow)); // dBm
    spectrumPhy.Set("TxPowerEnd", DoubleValue(pow));
    auto channelNumber = WifiPhyOperatingChannel::FindFirst(0, freq, bw, standard, phyBand)->number;
    std::ostringstream channelSettings;
    channelSettings << "{" << +channelNumber << ", " << bw << ", " << channelBand << ", 0}";
    spectrumPhy.Set("ChannelSettings", StringValue(channelSettings.str()));

    WifiMacHelper mac;
    wifi.SetStandard(standard);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue(dataRate),
                                 "ControlMode",
                                 StringValue(dataRate));

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevice = wifi.Install(spectrumPhy, mac, wifiStaNode);
    mac.SetType("ns3::ApWifiMac",
                "Ssid",
                SsidValue(ssid),
                "EnableBeaconJitter",
                BooleanValue(false)); // so as to be sure that first beacon arrives quickly
    NetDeviceContainer apDevice = wifi.Install(spectrumPhy, mac, wifiApNode);

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> nodePositionList = CreateObject<ListPositionAllocator>();
    nodePositionList->Add(Vector(0.0, 1.0, 0.0)); // AP
    nodePositionList->Add(Vector(1.0, 0.0, 0.0)); // STA
    nodePositionList->Add(Vector(0.0, 0.0, 0.0)); // Spectrum Analyzer
    mobility.SetPositionAllocator(nodePositionList);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    /* Need to send data packet because beacon and association frames shall be sent using lowest
     * rate */
    // Send one data packet (this packet is sent using data rate / MCS defined above) once
    // association is done (otherwise dropped)
    Simulator::Schedule(dataStartTime,
                        &SendPacket,
                        apDevice.Get(0),
                        staDevice.Get(0)->GetAddress());

    /* frequency range for spectrum analyzer */
    std::vector<double> freqs;
    const MHz_u margin{2}; // 1MHz margin on each side
    const auto band = (bw + margin);
    freqs.reserve(4 * 10 * band);
    const MHz_u scale{0.1};
    for (int i = 0; i < (4 * 10 * band); ++i) // conversion to 100kHz scale
    {
        freqs.push_back(MHzToHz((i * scale) + (freq - 2 * band)));
    }
    Ptr<SpectrumModel> spectrumAnalyzerFreqModel = Create<SpectrumModel>(freqs);

    /* spectrum analyzer setup */
    SpectrumAnalyzerHelper spectrumAnalyzerHelper;
    spectrumAnalyzerHelper.SetChannel(channel);
    spectrumAnalyzerHelper.SetRxSpectrumModel(spectrumAnalyzerFreqModel);
    spectrumAnalyzerHelper.SetPhyAttribute(
        "Resolution",
        TimeValue(MicroSeconds(4))); // enough resolution to distinguish OFDM symbols (default 1ms
                                     // too long even for PPDUs)
    std::ostringstream ossFileName;
    ossFileName << "spectrum-analyzer-wifi-" << standardStr << "-" << bw << "MHz";
    spectrumAnalyzerHelper.EnableAsciiAll(ossFileName.str());
    NetDeviceContainer spectrumAnalyzerDevices =
        spectrumAnalyzerHelper.Install(spectrumAnalyzerNodes);

    /* Let enough time for first beacon, association procedure, and first data (+acknowledgment and
     * eventually preceding ADDBA procedure) */
    Simulator::Stop(dataStartTime + dataDuration);

    Simulator::Run();

    /* Plot transmitted spectra with Gnuplot */
    ossFileName << "-2-0"; // append node-interface info
    std::ostringstream ossPlt;
    ossPlt << ossFileName.str() << ".plt";
    std::ofstream plotFile(ossPlt.str());
    std::ostringstream ossPng;
    ossPng << ossFileName.str() << ".png";
    Gnuplot plot = Gnuplot(ossPng.str());
    // Prepare 3D plot (reset previous values)
    std::ostringstream ossExtra;
    ossExtra << "file = '" << ossFileName.str() << "'";
    plot.SetExtra(ossExtra.str());
    plot.AppendExtra("unset surface");
    plot.AppendExtra("set key off");
    // Configure output file as png
    plot.AppendExtra("set term png");
    plot.AppendExtra("set output file . '.png'");
    // Switch to 3D plot
    plot.AppendExtra("set pm3d at s");
    plot.AppendExtra("set palette");
    // Orient view
    plot.AppendExtra("set view 50,50");
    // Add legends
    plot.AppendExtra("set xlabel \"time (ms)\"");
    plot.AppendExtra("set ylabel \"freq (MHz)\" offset 15,0,0");
    plot.AppendExtra("set zlabel \"PSD (dBW/Hz)\" offset 15,0,0");
    // Define grid
    plot.AppendExtra("set ytics");
    plot.AppendExtra("set mytics 2");
    plot.AppendExtra("set ztics");
    plot.AppendExtra("set mztics 5");
    plot.AppendExtra("set grid ytics mytics ztics mztics");
    // tr file name
    plot.AppendExtra("filename = file . '.tr'");
    // Extract max power using stats (so as to normalize during display)
    plot.AppendExtra("stats filename using 3");
    plot.AppendExtra("refW = STATS_max");
    // Plot graph (file being defined upon gnuplot call)
    plot.AppendExtra("splot filename using ($1*1000.0):($2/1e6):(10*log10($3/refW))");
    // Generate output and close file
    plot.GenerateOutput(plotFile);
    plotFile.close();

    Simulator::Destroy();

    std::cout << "Simulation done!" << std::endl;
    std::cout << "See spectrum analyzer output file: " << ossFileName.str() << ".tr" << std::endl;
    std::cout << "To generate plot simply execute the following command: gnuplot "
              << ossFileName.str() << ".plt" << std::endl;

    return 0;
}
