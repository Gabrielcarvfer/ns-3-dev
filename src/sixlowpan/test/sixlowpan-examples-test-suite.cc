#include "ns3/example-as-test.h"

static ns3::ExampleAsTestSuite g_sixlowpanExample ("example-ping-lr-wpan", "example-ping-lr-wpan", std::string ("src/sixlowpan/test"), "--disable-pcap --disable-asciitrace --enable-sixlowpan-loginfo");
