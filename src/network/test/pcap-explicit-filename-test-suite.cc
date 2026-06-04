/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/pcap-file-wrapper.h"
#include "ns3/pcap-file.h"
#include "ns3/ptr.h"
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/trace-helper.h"

#include <cstdint>
#include <cstdio>
#include <ios>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("pcap-explicit-filename-test-suite");

/**
 * @ingroup network-test
 * @ingroup tests
 *
 * @brief KNOWN-FAILING regression test for issue #1150.
 *
 * Issue #1150 reports that a malformed pcap file is produced when the same
 * explicit pcap filename is reused for more than one interface, e.g. when
 * EnablePcapIpv4(prefix, ..., explicitFilename = true) is called for two
 * different interfaces with the same @c prefix.
 *
 * Root cause (confirmed): PcapHelper::CreateFile()
 * (src/network/helper/trace-helper.cc) opens the file in @c std::ios::out
 * mode (which truncates) and then calls Init() to write a fresh pcap global
 * header, on every invocation. When two interfaces share the same explicit
 * filename, CreateFile() is invoked twice on the same path. Each invocation
 * truncates the file and writes a new global header, and the two returned
 * PcapFileWrapper objects keep independent file handles (independent write
 * offsets) pointing at the same file. The trace sinks of the two interfaces
 * therefore write records that overlap and clobber one another in the file,
 * producing a structurally malformed pcap. See
 * src/internet/helper/internet-stack-helper.cc, EnablePcapIpv4Internal()
 * (around line 436), which is the higher-level entry point that triggers
 * this for the IPv4 case.
 *
 * This test exercises the network-module root cause directly (it deliberately
 * avoids a dependency on the internet module) by calling
 * PcapHelper::CreateFile() twice with the same explicit filename, exactly as
 * the buggy helper path does, and then writing a couple of records of
 * different sizes through each returned wrapper, interleaved, mirroring the
 * trace sinks of two interfaces. It then reopens the resulting file and
 * verifies that it is a well-formed pcap: after the single 24-byte global
 * header, the record stream must parse self-consistently all the way to a
 * clean end-of-file on a record boundary, with every record header plausible.
 *
 * Concretely, two "interfaces" each write two records (four records total).
 * On the current (unfixed) code base the two wrappers share one truncated
 * file with independent write offsets, so one stream's records overwrite the
 * other's: the file ends up containing at most two records, never the four
 * that were written, and may additionally contain a record header whose
 * inclLen is implausible or whose payload runs past the physical end of file.
 * The test therefore asserts that all four written records are present and
 * self-consistent; this FAILS by design pre-fix (records are missing /
 * clobbered). Once issue #1150 is fixed (so that a shared explicit filename
 * yields a single coherent pcap stream containing every written packet), the
 * file parses cleanly to EOF with all four records and the assertion passes.
 *
 * The "fewer than four records" outcome holds regardless of build type and of
 * stream-flush timing, which makes this assertion robust; the extra
 * structural checks (plausible inclLen, no run past EOF, no trailing bytes)
 * can only ever add failure detail, never cause a false pass.
 *
 * The scenario is fully deterministic: fixed packet sizes, fixed contents,
 * fixed write ordering, no randomness and no simulator scheduling involved.
 */
class PcapExplicitFilenameTestCase : public TestCase
{
  public:
    PcapExplicitFilenameTestCase();
    ~PcapExplicitFilenameTestCase() override;

  private:
    void DoSetup() override;
    void DoRun() override;
    void DoTeardown() override;

    /// Magic number of a standard (microsecond) little-endian pcap file.
    static constexpr uint32_t PCAP_MAGIC = 0xa1b2c3d4;
    /// Size, in bytes, of the pcap global header.
    static constexpr uint32_t PCAP_GLOBAL_HEADER_LEN = 24;
    /// Size, in bytes, of a per-packet pcap record header.
    static constexpr uint32_t PCAP_RECORD_HEADER_LEN = 16;
    /// Snap length used when initializing the pcap files.
    static constexpr uint32_t SNAP_LEN = 65535;

    std::string m_testFilename; //!< Explicit pcap file name shared by both "interfaces"
};

PcapExplicitFilenameTestCase::PcapExplicitFilenameTestCase()
    : TestCase("Reusing one explicit pcap filename for two interfaces yields a valid pcap (#1150)")
{
}

PcapExplicitFilenameTestCase::~PcapExplicitFilenameTestCase()
{
}

void
PcapExplicitFilenameTestCase::DoSetup()
{
    m_testFilename = CreateTempDirFilename("pcap-explicit-filename-1150.pcap");
    std::remove(m_testFilename.c_str());
}

void
PcapExplicitFilenameTestCase::DoTeardown()
{
    if (std::remove(m_testFilename.c_str()))
    {
        NS_LOG_ERROR("Failed to delete file " << m_testFilename);
    }
}

void
PcapExplicitFilenameTestCase::DoRun()
{
    PcapHelper pcapHelper;

    //
    // Reproduce the exact buggy helper path: two interfaces share one explicit
    // filename, so CreateFile() is called twice on the same path. Each call
    // opens the file in std::ios::out mode (truncating) and writes a fresh
    // global header.
    //
    Ptr<PcapFileWrapper> fileA =
        pcapHelper.CreateFile(m_testFilename, std::ios::out, PcapHelper::DLT_RAW, SNAP_LEN);
    NS_TEST_ASSERT_MSG_EQ(fileA->Fail(),
                          false,
                          "CreateFile() for the first interface should succeed");

    Ptr<PcapFileWrapper> fileB =
        pcapHelper.CreateFile(m_testFilename, std::ios::out, PcapHelper::DLT_RAW, SNAP_LEN);
    NS_TEST_ASSERT_MSG_EQ(fileB->Fail(),
                          false,
                          "CreateFile() for the second interface should succeed");

    //
    // Build two packets of clearly different sizes, one per "interface", so
    // that records written at overlapping offsets are detectably inconsistent.
    //
    std::vector<uint8_t> payloadA(40, 0xa1);
    std::vector<uint8_t> payloadB(80, 0xb2);
    Ptr<Packet> pktA = Create<Packet>(payloadA.data(), payloadA.size());
    Ptr<Packet> pktB = Create<Packet>(payloadB.data(), payloadB.size());

    //
    // Interleave the writes, mirroring the trace sinks of the two interfaces.
    //
    fileA->Write(Seconds(1), pktA);
    fileB->Write(Seconds(2), pktB);
    fileA->Write(Seconds(3), pktA);
    fileB->Write(Seconds(4), pktB);

    //
    // Release the wrappers so their file handles are flushed and closed.
    //
    fileA = nullptr;
    fileB = nullptr;

    //
    // Reopen the file as a reader and verify it is a well-formed pcap.
    //
    PcapFile reader;
    reader.Open(m_testFilename, std::ios::in);
    NS_TEST_ASSERT_MSG_EQ(reader.Fail(),
                          false,
                          "Resulting pcap file should have a valid global header");
    NS_TEST_ASSERT_MSG_EQ(reader.GetMagic(),
                          PCAP_MAGIC,
                          "Resulting pcap file should carry the standard pcap magic number");

    //
    // Determine the physical size of the file so we can verify that the record
    // stream parses self-consistently all the way to EOF on a record boundary.
    //
    FILE* raw = std::fopen(m_testFilename.c_str(), "rb");
    NS_TEST_ASSERT_MSG_NE(raw, nullptr, "Should be able to open the resulting pcap file");
    std::fseek(raw, 0, SEEK_END);
    auto fileSize = static_cast<uint64_t>(std::ftell(raw));
    std::fclose(raw);

    //
    // Walk every record. PcapFile::Read() positions the reader just past the
    // global header on Open(), so we account for that header up front and then
    // add each record's on-disk footprint (record header + included length).
    //
    uint64_t consumed = PCAP_GLOBAL_HEADER_LEN;
    uint32_t recordCount = 0;
    std::vector<uint8_t> buffer(SNAP_LEN);

    while (consumed + PCAP_RECORD_HEADER_LEN <= fileSize)
    {
        uint32_t tsSec = 0;
        uint32_t tsUsec = 0;
        uint32_t inclLen = 0;
        uint32_t origLen = 0;
        uint32_t readLen = 0;

        reader.Read(buffer.data(), buffer.size(), tsSec, tsUsec, inclLen, origLen, readLen);

        if (reader.Eof())
        {
            // Clean end-of-file reached exactly on a record boundary.
            break;
        }
        NS_TEST_ASSERT_MSG_EQ(reader.Fail(),
                              false,
                              "Record " << recordCount << " in a well-formed pcap should read "
                                        << "without error (issue #1150: shared explicit filename "
                                        << "corrupts the file)");

        // The included length must never exceed the configured snap length.
        NS_TEST_ASSERT_MSG_LT_OR_EQ(inclLen,
                                    SNAP_LEN,
                                    "Record " << recordCount << " has an implausible included "
                                              << "length, indicating a malformed pcap (issue #1150)");

        // The record (header + payload) must fit entirely within the file.
        NS_TEST_ASSERT_MSG_LT_OR_EQ(consumed + PCAP_RECORD_HEADER_LEN + inclLen,
                                    fileSize,
                                    "Record " << recordCount << " runs past the end of the file, "
                                              << "indicating a malformed pcap (issue #1150)");

        consumed += PCAP_RECORD_HEADER_LEN + inclLen;
        recordCount++;
    }

    //
    // After parsing every record, the consumed byte count must match the
    // physical file size exactly: a well-formed pcap has no leftover bytes
    // that do not belong to a complete record.
    //
    NS_TEST_ASSERT_MSG_EQ(consumed,
                          fileSize,
                          "Parsed pcap record stream does not end exactly at the end of the "
                              << "file; the file contains overlapping/garbled records "
                              << "(issue #1150: shared explicit filename for two interfaces)");

    //
    // Primary, flush-order-independent invariant: every packet written for the
    // shared explicit filename must be present. Two interfaces wrote two
    // records each, so a well-formed file must contain exactly four records.
    // Pre-fix, the two wrappers clobber each other and the file contains at
    // most two records, so this assertion fails by design.
    //
    NS_TEST_ASSERT_MSG_EQ(recordCount,
                          4,
                          "Resulting pcap should contain all four written records, one per "
                              << "Write() call across the two interfaces; fewer records means "
                              << "packets were overwritten (issue #1150: shared explicit "
                              << "filename for two interfaces)");

    reader.Close();
}

/**
 * @ingroup network-test
 * @ingroup tests
 *
 * @brief Test suite for the issue #1150 explicit-pcap-filename regression test.
 */
class PcapExplicitFilenameTestSuite : public TestSuite
{
  public:
    PcapExplicitFilenameTestSuite();
};

PcapExplicitFilenameTestSuite::PcapExplicitFilenameTestSuite()
    : TestSuite("pcap-explicit-filename", Type::UNIT)
{
    AddTestCase(new PcapExplicitFilenameTestCase, TestCase::Duration::QUICK);
}

static PcapExplicitFilenameTestSuite
    g_pcapExplicitFilenameTestSuite; //!< Static variable for test initialization
