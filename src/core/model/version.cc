/*
 * Copyright (c) 2018 Lawrence Livermore National Laboratory
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Mathew Bielejeski <bielejeski1@llnl.gov>
 */

#include "version.h"

#include "ns3/version-defines.h"

#include <sstream>

/**
 * @file
 * @ingroup buildversion
 * ns3::Version implementation
 */

namespace ns3
{

std::string
Version::VersionTag()
{
    return NS3_VERSION_TAG;
}

std::string
Version::ClosestAncestorTag()
{
    return NS3_VERSION_CLOSEST_TAG;
}

uint32_t
Version::Major()
{
    return NS3_VERSION_MAJOR;
}

uint32_t
Version::Minor()
{
    return NS3_VERSION_MINOR;
}

uint32_t
Version::Patch()
{
    return NS3_VERSION_PATCH;
}

std::string
Version::ReleaseCandidate()
{
    return std::string{NS3_VERSION_RELEASE_CANDIDATE};
}

uint32_t
Version::TagDistance()
{
    return NS3_VERSION_TAG_DISTANCE;
}

bool
Version::DirtyWorkingTree()
{
    return static_cast<bool>(NS3_VERSION_DIRTY_FLAG);
}

std::string
Version::CommitHash()
{
    return std::string{NS3_VERSION_COMMIT_HASH};
}

std::string
Version::BuildProfile()
{
    return std::string{NS3_VERSION_BUILD_PROFILE};
}

std::string
Version::ShortVersion()
{
    std::ostringstream ostream;
    ostream << VersionTag();

    auto ancestorTag = ClosestAncestorTag();
    if ((!ancestorTag.empty() && (ancestorTag != VersionTag())) || TagDistance() > 0)
    {
        ostream << "+";
    }
    if (DirtyWorkingTree())
    {
        ostream << "*";
    }

    return ostream.str();
}

std::string
Version::BuildSummary()
{
    std::ostringstream ostream;
    ostream << ClosestAncestorTag();

    if (TagDistance() > 0)
    {
        ostream << "+";
    }
    if (DirtyWorkingTree())
    {
        ostream << "*";
    }

    return ostream.str();
}

std::string
Version::LongVersion()
{
    std::ostringstream ostream;
    ostream << VersionTag();

    auto ancestorTag = ClosestAncestorTag();
    if (!ancestorTag.empty() && (ancestorTag != VersionTag()))
    {
        ostream << '+' << ancestorTag;
    }

    auto tagDistance = TagDistance();

    if (tagDistance > 0)
    {
        ostream << '+' << tagDistance;
    }

    ostream << '@' << CommitHash();

    if (DirtyWorkingTree())
    {
        ostream << "-dirty";
    }

    ostream << '-' << BuildProfile();

    return ostream.str();
}

} // namespace ns3
