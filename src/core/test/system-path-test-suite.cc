/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/system-path.h"
#include "ns3/test.h"

#include <filesystem>
#include <fstream>

using namespace ns3;
using namespace ns3::SystemPath;

/**
 * @ingroup core-tests
 * @defgroup system-path-tests SystemPath test suite
 */

/**
 * @ingroup system-path-tests
 *
 * @brief Tests for SystemPath::Exists().
 *
 * Regression test for issue #781: Exists() used to derive the directory part
 * of the path and search it for the final component, which returned a wrong
 * result for a bare relative filename (the derived directory was the empty
 * string) and mishandled trailing separators. It now delegates to
 * std::filesystem and must agree with it.
 */
class SystemPathExistsTestCase : public TestCase
{
  public:
    SystemPathExistsTestCase()
        : TestCase("SystemPath::Exists matches std::filesystem::exists")
    {
    }

  private:
    void DoRun() override;
};

void
SystemPathExistsTestCase::DoRun()
{
    namespace fs = std::filesystem;

    // Create a unique temporary directory and a file inside it.
    std::string tempDir = MakeTemporaryDirectoryName();
    MakeDirectories(tempDir);
    NS_TEST_ASSERT_MSG_EQ(Exists(tempDir), true, "Created directory should exist");
    // A trailing separator must still resolve to the existing directory.
    NS_TEST_ASSERT_MSG_EQ(Exists(tempDir + "/"), true, "Directory with trailing slash should exist");

    std::string filePath = Append(tempDir, "a-file.txt");
    {
        std::ofstream f(filePath);
        f << "ns-3";
    }
    NS_TEST_ASSERT_MSG_EQ(Exists(filePath), true, "Created file should exist");
    NS_TEST_ASSERT_MSG_EQ(Exists(Append(tempDir, "missing.txt")),
                          false,
                          "Non-existent file must not exist");
    NS_TEST_ASSERT_MSG_EQ(Exists(Append(tempDir, "no/such/dir")),
                          false,
                          "Non-existent nested path must not exist");

    // Regression for issue #781: a bare relative filename (no separator) that
    // exists in the current working directory must be reported as existing.
    fs::path original = fs::current_path();
    fs::current_path(tempDir);
    NS_TEST_ASSERT_MSG_EQ(Exists("a-file.txt"),
                          true,
                          "Bare filename present in the current directory must exist");
    NS_TEST_ASSERT_MSG_EQ(Exists("definitely-not-here.txt"),
                          false,
                          "Bare filename absent from the current directory must not exist");
    fs::current_path(original);

    // Cleanup.
    std::error_code ec;
    fs::remove_all(tempDir, ec);
}

/**
 * @ingroup system-path-tests
 *
 * @brief SystemPath test suite.
 */
class SystemPathTestSuite : public TestSuite
{
  public:
    SystemPathTestSuite()
        : TestSuite("system-path", Type::UNIT)
    {
        AddTestCase(new SystemPathExistsTestCase, TestCase::Duration::QUICK);
    }
};

static SystemPathTestSuite g_systemPathTestSuite; //!< Static variable for test initialization
