/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/names.h"
#include "ns3/test.h"

/**
 * @file
 * @ingroup core-tests
 * @ingroup config
 * @ingroup names-tests
 * Object Names test suite.
 */

/**
 * @ingroup core-tests
 * @defgroup names-tests Object Names test suite
 */

namespace ns3
{

namespace tests
{

/**
 * @ingroup names-tests
 * Simple test object to exercise the Name service.
 */
class TestObject : public Object
{
  public:
    /**
     * Register this type.
     * @return The TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("TestObject")
                                .SetParent<Object>()
                                .SetGroupName("Core")
                                .HideFromDocumentation()
                                .AddConstructor<TestObject>();
        return tid;
    }

    /** Constructor. */
    TestObject()
    {
    }
};

/**
 * @ingroup names-tests
 * Alternate test object for the Name service.
 */
class AlternateTestObject : public Object
{
  public:
    /**
     * Register this type.
     * @return The TypeId.
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("AlternateTestObject")
                                .SetParent<Object>()
                                .SetGroupName("Core")
                                .HideFromDocumentation()
                                .AddConstructor<AlternateTestObject>();
        return tid;
    }

    /** Constructor. */
    AlternateTestObject()
    {
    }
};

/**
 * @ingroup names-tests
 * Test the Object Name Service can do its most basic job.
 *
 * Add associations between Objects using the lowest level add
 * function, which is:
 *
 *     Add (Ptr<Object> context, std::string name, Ptr<Object> object);
 *
 * All other add functions will just translate into this form, so this is the
 * most basic Add functionality.
 */
class BasicAddTestCase : public TestCase
{
  public:
    /** Constructor. */
    BasicAddTestCase();
    /** Destructor. */
    ~BasicAddTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

BasicAddTestCase::BasicAddTestCase()
    : TestCase("Check low level Names::Add and Names::FindName functionality")
{
}

BasicAddTestCase::~BasicAddTestCase()
{
}

void
BasicAddTestCase::DoTeardown()
{
    Names::Clear();
}

void
BasicAddTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add(Ptr<Object>(nullptr, false), "Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add(Ptr<Object>(nullptr, false), "Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add(objectOne, "Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add(objectTwo, "Child", childOfObjectTwo);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name One", "Could not Names::Add and Names::FindName an Object");

    found = Names::FindName(objectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Name Two",
                          "Could not Names::Add and Names::FindName a second Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    found = Names::FindName(childOfObjectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can correctly use a string context.
 *
 *     Add (std::string context, std::string name, Ptr<Object> object);
 *
 * High level path-based functions will translate into this form, so this is
 * the second most basic Add functionality.
 */
class StringContextAddTestCase : public TestCase
{
  public:
    /** Constructor. */
    StringContextAddTestCase();
    /** Destructor. */
    ~StringContextAddTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

StringContextAddTestCase::StringContextAddTestCase()
    : TestCase("Check string context Names::Add and Names::FindName functionality")

{
}

StringContextAddTestCase::~StringContextAddTestCase()
{
}

void
StringContextAddTestCase::DoTeardown()
{
    Names::Clear();
}

void
StringContextAddTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("/Names", "Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("/Names", "Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name One", "Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("/Names/Name Two", "Child", childOfObjectTwo);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name One", "Could not Names::Add and Names::FindName an Object");

    found = Names::FindName(objectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Name Two",
                          "Could not Names::Add and Names::FindName a second Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    found = Names::FindName(childOfObjectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can correctly use a
 * fully qualified path to add associations.
 *
 *     Add (std::string name, Ptr<Object> object);
 *
 */
class FullyQualifiedAddTestCase : public TestCase
{
  public:
    /** Constructor. */
    FullyQualifiedAddTestCase();
    /** Destructor. */
    ~FullyQualifiedAddTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

FullyQualifiedAddTestCase::FullyQualifiedAddTestCase()
    : TestCase("Check fully qualified path Names::Add and Names::FindName functionality")

{
}

FullyQualifiedAddTestCase::~FullyQualifiedAddTestCase()
{
}

void
FullyQualifiedAddTestCase::DoTeardown()
{
    Names::Clear();
}

void
FullyQualifiedAddTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("/Names/Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name One/Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("/Names/Name Two/Child", childOfObjectTwo);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name One", "Could not Names::Add and Names::FindName an Object");

    found = Names::FindName(objectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Name Two",
                          "Could not Names::Add and Names::FindName a second Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    found = Names::FindName(childOfObjectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can correctly use a
 * relative path to add associations.
 *
 * This functionality is provided as a convenience so clients
 * don't always have to provide the name service namespace name
 * in all of their strings.
 *
 *     Add (std::string name, Ptr<Object> object);
 *
 */
class RelativeAddTestCase : public TestCase
{
  public:
    /** Constructor. */
    RelativeAddTestCase();
    /** Destructor. */
    ~RelativeAddTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

RelativeAddTestCase::RelativeAddTestCase()
    : TestCase("Check relative path Names::Add and Names::FindName functionality")

{
}

RelativeAddTestCase::~RelativeAddTestCase()
{
}

void
RelativeAddTestCase::DoTeardown()
{
    Names::Clear();
}

void
RelativeAddTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("Name One/Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("Name Two/Child", childOfObjectTwo);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name One", "Could not Names::Add and Names::FindName an Object");

    found = Names::FindName(objectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Name Two",
                          "Could not Names::Add and Names::FindName a second Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    found = Names::FindName(childOfObjectTwo);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can rename objects.
 *
 *     Rename (Ptr<Object> context, std::string oldname, std::string newname);
 *
 * All other rename functions will just translate into this form, so this is the
 * most basic rename functionality.
 */
class BasicRenameTestCase : public TestCase
{
  public:
    /** Constructor. */
    BasicRenameTestCase();
    /** Destructor. */
    ~BasicRenameTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

BasicRenameTestCase::BasicRenameTestCase()
    : TestCase("Check low level Names::Rename functionality")
{
}

BasicRenameTestCase::~BasicRenameTestCase()
{
}

void
BasicRenameTestCase::DoTeardown()
{
    Names::Clear();
}

void
BasicRenameTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add(Ptr<Object>(nullptr, false), "Name", objectOne);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add(objectOne, "Child", childOfObjectOne);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name", "Could not Names::Add and Names::FindName an Object");

    Names::Rename(Ptr<Object>(nullptr, false), "Name", "New Name");

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Name", "Could not Names::Rename an Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    Names::Rename(objectOne, "Child", "New Child");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Child", "Could not Names::Rename a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can rename objects
 * using a string context.
 *
 *     Rename (std::string context, std::string oldname, std::string newname);
 *
 */
class StringContextRenameTestCase : public TestCase
{
  public:
    /** Constructor. */
    StringContextRenameTestCase();
    /** Destructor. */
    ~StringContextRenameTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

StringContextRenameTestCase::StringContextRenameTestCase()
    : TestCase("Check string context-based Names::Rename functionality")
{
}

StringContextRenameTestCase::~StringContextRenameTestCase()
{
}

void
StringContextRenameTestCase::DoTeardown()
{
    Names::Clear();
}

void
StringContextRenameTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("/Names", "Name", objectOne);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name", "Child", childOfObjectOne);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name", "Could not Names::Add and Names::FindName an Object");

    Names::Rename("/Names", "Name", "New Name");

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Name", "Could not Names::Rename an Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    Names::Rename("/Names/New Name", "Child", "New Child");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Child", "Could not Names::Rename a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can rename objects
 * using a fully qualified path name.
 *
 *     Rename (std::string oldpath, std::string newname);
 *
 */
class FullyQualifiedRenameTestCase : public TestCase
{
  public:
    /** Constructor. */
    FullyQualifiedRenameTestCase();
    /** Destructor. */
    ~FullyQualifiedRenameTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

FullyQualifiedRenameTestCase::FullyQualifiedRenameTestCase()
    : TestCase("Check fully qualified path Names::Rename functionality")
{
}

FullyQualifiedRenameTestCase::~FullyQualifiedRenameTestCase()
{
}

void
FullyQualifiedRenameTestCase::DoTeardown()
{
    Names::Clear();
}

void
FullyQualifiedRenameTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name", objectOne);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name/Child", childOfObjectOne);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name", "Could not Names::Add and Names::FindName an Object");

    Names::Rename("/Names/Name", "New Name");

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Name", "Could not Names::Rename an Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    Names::Rename("/Names/New Name/Child", "New Child");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Child", "Could not Names::Rename a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can rename objects
 * using a relative path name.
 *
 *     Rename (std::string oldpath, std::string newname);
 *
 */
class RelativeRenameTestCase : public TestCase
{
  public:
    /** Constructor. */
    RelativeRenameTestCase();
    /** Destructor. */
    ~RelativeRenameTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

RelativeRenameTestCase::RelativeRenameTestCase()
    : TestCase("Check relative path Names::Rename functionality")
{
}

RelativeRenameTestCase::~RelativeRenameTestCase()
{
}

void
RelativeRenameTestCase::DoTeardown()
{
    Names::Clear();
}

void
RelativeRenameTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("Name", objectOne);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("Name/Child", childOfObjectOne);

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "Name", "Could not Names::Add and Names::FindName an Object");

    Names::Rename("Name", "New Name");

    found = Names::FindName(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Name", "Could not Names::Rename an Object");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "Child",
                          "Could not Names::Add and Names::FindName a child Object");

    Names::Rename("New Name/Child", "New Child");

    found = Names::FindName(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found, "New Child", "Could not Names::Rename a child Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can look up an object
 * and return its fully qualified path name.
 *
 *     FindPath (Ptr<Object> object);
 *
 */
class FindPathTestCase : public TestCase
{
  public:
    /** Constructor. */
    FindPathTestCase();
    /** Destructor. */
    ~FindPathTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

FindPathTestCase::FindPathTestCase()
    : TestCase("Check Names::FindPath functionality")
{
}

FindPathTestCase::~FindPathTestCase()
{
}

void
FindPathTestCase::DoTeardown()
{
    Names::Clear();
}

void
FindPathTestCase::DoRun()
{
    std::string found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("Name", objectOne);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name/Child", childOfObjectOne);

    found = Names::FindPath(objectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "/Names/Name",
                          "Could not Names::Add and Names::FindPath an Object");

    found = Names::FindPath(childOfObjectOne);
    NS_TEST_ASSERT_MSG_EQ(found,
                          "/Names/Name/Child",
                          "Could not Names::Add and Names::FindPath a child Object");

    Ptr<TestObject> objectNotThere = CreateObject<TestObject>();
    found = Names::FindPath(objectNotThere);
    NS_TEST_ASSERT_MSG_EQ(found.empty(), true, "Unexpectedly found a non-existent Object");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can find Objects.
 *
 *     Find (Ptr<Object> context, std::string name);
 *
 */
class BasicFindTestCase : public TestCase
{
  public:
    /** Constructor. */
    BasicFindTestCase();
    /** Destructor. */
    ~BasicFindTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

BasicFindTestCase::BasicFindTestCase()
    : TestCase("Check low level Names::Find functionality")
{
}

BasicFindTestCase::~BasicFindTestCase()
{
}

void
BasicFindTestCase::DoTeardown()
{
    Names::Clear();
}

void
BasicFindTestCase::DoRun()
{
    Ptr<TestObject> found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("Name One/Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("Name Two/Child", childOfObjectTwo);

    found = Names::Find<TestObject>(Ptr<Object>(nullptr, false), "Name One");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectOne,
                          "Could not find a previously named Object via object context");

    found = Names::Find<TestObject>(Ptr<Object>(nullptr, false), "Name Two");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectTwo,
                          "Could not find a previously named Object via object context");

    found = Names::Find<TestObject>(objectOne, "Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectOne,
                          "Could not find a previously named child Object via object context");

    found = Names::Find<TestObject>(objectTwo, "Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectTwo,
                          "Could not find a previously named child Object via object context");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can find Objects using
 * a string context.
 *
 *     Find (std::string context, std::string name);
 *
 */
class StringContextFindTestCase : public TestCase
{
  public:
    /** Constructor. */
    StringContextFindTestCase();
    /** Destructor. */
    ~StringContextFindTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

StringContextFindTestCase::StringContextFindTestCase()
    : TestCase("Check string context-based Names::Find functionality")
{
}

StringContextFindTestCase::~StringContextFindTestCase()
{
}

void
StringContextFindTestCase::DoTeardown()
{
    Names::Clear();
}

void
StringContextFindTestCase::DoRun()
{
    Ptr<TestObject> found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("Name One/Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("Name Two/Child", childOfObjectTwo);

    found = Names::Find<TestObject>("/Names", "Name One");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectOne,
                          "Could not find a previously named Object via string context");

    found = Names::Find<TestObject>("/Names", "Name Two");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectTwo,
                          "Could not find a previously named Object via stribng context");

    found = Names::Find<TestObject>("/Names/Name One", "Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectOne,
                          "Could not find a previously named child Object via string context");

    found = Names::Find<TestObject>("/Names/Name Two", "Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectTwo,
                          "Could not find a previously named child Object via string context");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can find Objects using
 * a fully qualified path name.
 *
 *     Find (std::string name);
 *
 */
class FullyQualifiedFindTestCase : public TestCase
{
  public:
    /** Constructor. */
    FullyQualifiedFindTestCase();
    /** Destructor. */
    ~FullyQualifiedFindTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

FullyQualifiedFindTestCase::FullyQualifiedFindTestCase()
    : TestCase("Check fully qualified path Names::Find functionality")
{
}

FullyQualifiedFindTestCase::~FullyQualifiedFindTestCase()
{
}

void
FullyQualifiedFindTestCase::DoTeardown()
{
    Names::Clear();
}

void
FullyQualifiedFindTestCase::DoRun()
{
    Ptr<TestObject> found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("/Names/Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("/Names/Name One/Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("/Names/Name Two/Child", childOfObjectTwo);

    found = Names::Find<TestObject>("/Names/Name One");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectOne,
                          "Could not find a previously named Object via string context");

    found = Names::Find<TestObject>("/Names/Name Two");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectTwo,
                          "Could not find a previously named Object via stribng context");

    found = Names::Find<TestObject>("/Names/Name One/Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectOne,
                          "Could not find a previously named child Object via string context");

    found = Names::Find<TestObject>("/Names/Name Two/Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectTwo,
                          "Could not find a previously named child Object via string context");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can find Objects using
 * a relative path name.
 *
 *     Find (std::string name);
 *
 */
class RelativeFindTestCase : public TestCase
{
  public:
    /** Constructor. */
    RelativeFindTestCase();
    /** Destructor. */
    ~RelativeFindTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

RelativeFindTestCase::RelativeFindTestCase()
    : TestCase("Check relative path Names::Find functionality")
{
}

RelativeFindTestCase::~RelativeFindTestCase()
{
}

void
RelativeFindTestCase::DoTeardown()
{
    Names::Clear();
}

void
RelativeFindTestCase::DoRun()
{
    Ptr<TestObject> found;

    Ptr<TestObject> objectOne = CreateObject<TestObject>();
    Names::Add("Name One", objectOne);

    Ptr<TestObject> objectTwo = CreateObject<TestObject>();
    Names::Add("Name Two", objectTwo);

    Ptr<TestObject> childOfObjectOne = CreateObject<TestObject>();
    Names::Add("Name One/Child", childOfObjectOne);

    Ptr<TestObject> childOfObjectTwo = CreateObject<TestObject>();
    Names::Add("Name Two/Child", childOfObjectTwo);

    found = Names::Find<TestObject>("Name One");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectOne,
                          "Could not find a previously named Object via string context");

    found = Names::Find<TestObject>("Name Two");
    NS_TEST_ASSERT_MSG_EQ(found,
                          objectTwo,
                          "Could not find a previously named Object via stribng context");

    found = Names::Find<TestObject>("Name One/Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectOne,
                          "Could not find a previously named child Object via string context");

    found = Names::Find<TestObject>("Name Two/Child");
    NS_TEST_ASSERT_MSG_EQ(found,
                          childOfObjectTwo,
                          "Could not find a previously named child Object via string context");
}

/**
 * @ingroup names-tests
 * Test the Object Name Service can find Objects using
 * a second type.
 */
class AlternateFindTestCase : public TestCase
{
  public:
    /** Constructor. */
    AlternateFindTestCase();
    /** Destructor. */
    ~AlternateFindTestCase() override;

  private:
    void DoRun() override;
    void DoTeardown() override;
};

AlternateFindTestCase::AlternateFindTestCase()
    : TestCase("Check GetObject operation in Names::Find")
{
}

AlternateFindTestCase::~AlternateFindTestCase()
{
}

void
AlternateFindTestCase::DoTeardown()
{
    Names::Clear();
}

void
AlternateFindTestCase::DoRun()
{
    Ptr<TestObject> testObject = CreateObject<TestObject>();
    Names::Add("Test Object", testObject);

    Ptr<AlternateTestObject> alternateTestObject = CreateObject<AlternateTestObject>();
    Names::Add("Alternate Test Object", alternateTestObject);

    Ptr<TestObject> foundTestObject;
    Ptr<AlternateTestObject> foundAlternateTestObject;

    foundTestObject = Names::Find<TestObject>("Test Object");
    NS_TEST_ASSERT_MSG_EQ(foundTestObject,
                          testObject,
                          "Could not find a previously named TestObject via GetObject");

    foundAlternateTestObject = Names::Find<AlternateTestObject>("Alternate Test Object");
    NS_TEST_ASSERT_MSG_EQ(foundAlternateTestObject,
                          alternateTestObject,
                          "Could not find a previously named AlternateTestObject via GetObject");

    foundAlternateTestObject = Names::Find<AlternateTestObject>("Test Object");
    NS_TEST_ASSERT_MSG_EQ(foundAlternateTestObject,
                          nullptr,
                          "Unexpectedly able to GetObject<AlternateTestObject> on a TestObject");

    foundTestObject = Names::Find<TestObject>("Alternate Test Object");
    NS_TEST_ASSERT_MSG_EQ(foundTestObject,
                          nullptr,
                          "Unexpectedly able to GetObject<TestObject> on an AlternateTestObject");
}

/**
 * @ingroup names-tests
 * Names Test Suite
 */
class NamesTestSuite : public TestSuite
{
  public:
    /** Constructor. */
    NamesTestSuite();
};

NamesTestSuite::NamesTestSuite()
    : TestSuite("object-name-service")
{
    AddTestCase(new BasicAddTestCase);
    AddTestCase(new StringContextAddTestCase);
    AddTestCase(new FullyQualifiedAddTestCase);
    AddTestCase(new RelativeAddTestCase);
    AddTestCase(new BasicRenameTestCase);
    AddTestCase(new StringContextRenameTestCase);
    AddTestCase(new FullyQualifiedRenameTestCase);
    AddTestCase(new RelativeRenameTestCase);
    AddTestCase(new FindPathTestCase);
    AddTestCase(new BasicFindTestCase);
    AddTestCase(new StringContextFindTestCase);
    AddTestCase(new FullyQualifiedFindTestCase);
    AddTestCase(new RelativeFindTestCase);
    AddTestCase(new AlternateFindTestCase);
}

/**
 * @ingroup names-tests
 *  NamesTestSuite instance variable.
 */
static NamesTestSuite g_namesTestSuite;

} // namespace tests

} // namespace ns3
