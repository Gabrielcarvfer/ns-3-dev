.. include:: replace.txt
.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

.. _AI-tool:

AI tool policy
--------------

This policy is mostly copied from a proposal being discussed for
`LLVM <https://github.com/llvm/llvm-project/pull/154441>`_.

Policy
******

ns-3's policy is that contributors can use whatever tools they would like to
craft their contributions, but there must be a *human in the loop*.
*Contributors must read and review all LLM-generated code or text before they
ask other project members to review it.* The contributor is always the author
and is fully accountable for their contributions. Contributors should be
sufficiently confident that the contribution is high enough quality that asking
for a review is a good use of scarce maintainer time, and they should be *able
to answer questions about their work* during review.

We expect that new contributors will be less confident in their contributions,
and our guidance to them is to *start with small contributions* that they can
fully understand to build confidence. We aspire to be a welcoming community
that helps new contributors grow their expertise, but learning involves taking
small steps, getting feedback, and iterating. Passing maintainer feedback to an
LLM doesn't help anyone grow, and does not sustain our community.

Contributors are expected to *be transparent and label contributions that
contain substantial amounts of tool-generated content*. Our policy on
labelling is intended to facilitate reviews, and not to track which parts of
|ns3| are generated. Contributors should note tool usage in their merge request
description, commit message, or wherever authorship is normally indicated for
the work. For instance, use a commit message trailer like ``Assisted-by: (name
and version of code assistant)``. This transparency helps the community develop
best practices and understand the role of these new tools.

This policy includes, but is not limited to, the following kinds of
contributions:

- Code, usually in the form of a merge request
- RFCs or design proposals
- Issues or security vulnerabilities
- Comments and feedback on merge requests

Details
*******

To ensure sufficient self review and understanding of the work, it is strongly
recommended that contributors write MR descriptions themselves (if needed,
using tools for translation or copy-editing). The description should explain
the motivation, implementation approach, expected impact, and any open
questions or uncertainties to the same extent as a contribution made without
tool assistance.

An important implication of this policy is that it bans agents that take action
in our digital spaces without human approval, such as the `GitHub @claude
agent <https://github.com/claude/>`_. Similarly, automated review tools that
publish comments without human review are not allowed. However, an opt-in
review tool that *keeps a human in the loop* is acceptable under this policy.
As another example, using an LLM to generate documentation, which a contributor
manually reviews for correctness, edits, and then posts as a MR, is an approved
use of tools under this policy.

AI tools must not be used to fix GitLab issues labelled ``good first
issue``. These issues are generally not urgent, and are
intended to be learning opportunities for new contributors to get familiar with
the codebase. Whether you are a newcomer or not, fully automating the process
of fixing this issue squanders the learning opportunity and doesn't add much
value to the project. *Using AI tools to fix issues labelled as "good first
issues" is forbidden*.

Extractive Contributions
************************

The reason for our *human in the loop* contribution policy is that processing
patches, MRs, RFCs, and comments to |ns3| is not free -- it takes a lot of
maintainer time and energy to review those contributions! Sending the
unreviewed output of an LLM to open source project maintainers *extracts* work
from them in the form of design and code review, so we call this kind of
contribution an *extractive contribution*.

Prior to the advent of LLMs, open source project maintainers would often review
any and all changes sent to the project simply because posting a change for
review was a sign of interest from a potential long-term contributor. While new
tools enable more development, it shifts effort from the implementer to the
reviewer, and our policy exists to ensure that we value and do not squander
maintainer time.

Reviewing changes from new contributors is part of growing the next generation
of contributors and sustaining the project. We want the |ns3| project to be
welcoming and open to aspiring engineers who are willing to invest
time and effort to learn and grow, because growing our contributor base and
recruiting new maintainers helps sustain the project over the long term.

Handling Violations
*******************

If a maintainer judges that a contribution doesn't comply with this policy,
they should paste the following response to request changes:

.. code-block:: text

    This MR doesn't appear to comply with our policy on tool-generated content,
    and requires additional justification for why it is valuable enough to the
    project for us to review it. Please see our developer policy on
    AI-generated contributions: (URL TBD)

The best ways to make a change less extractive and more valuable are to reduce
its size or complexity or to increase its usefulness to the community. These
factors are impossible to weigh objectively, and our project policy leaves this
determination up to the maintainers of the project, i.e. those who are doing
the work of sustaining the project.

If or when it becomes clear that a GitLab issue or MR is off-track and not
moving in the right direction, maintainers should apply the ``extractive`` label
to help other reviewers prioritize their review time. If the contributor
doesn't take steps to fix the reported issues within a short time (days to few
weeks), the MR may be closed.

Copyright
*********

Artificial intelligence systems raise many questions around copyright that have
yet to be answered. Our policy on AI tools is similar to our copyright policy:
Contributors are responsible for ensuring that they have the right to
contribute code under the terms of our license, typically meaning that either
they, their employer, or their collaborators hold the copyright. Using AI tools
to regenerate copyrighted material does not remove the copyright, and
contributors are responsible for ensuring that such material does not appear in
their contributions. Contributions found to violate this policy will be removed
just like any other offending contribution.

Examples of when to mention tool use
************************************

We expect that LLM-based tools will increasingly be integrated into IDEs and
other developer tools, and it will be difficult to determine when to mention
tool use in a MR description, header file of a contribution, or commit message.
We suggest the following broad guidelines:

- if the LLM provided inputs along the lines of cleanup, linting, catching of
  small mistakes, debugging of compiler error messages, etc., use does not
  need to be mentioned or disclosed. Use of LLMs to auto-generate Doxygen
  of class methods or member variables need not be mentioned explicitly.
- if the LLM generated a substantial chunk of original code (implementation or
  tests), or documentation (such as Sphinx), use should be disclosed
- if the LLM was used to port another implementation to |ns3|, use should be
  disclosed

When in doubt, ask yourself if you asked the LLM to generate something new for
you, or if you instead asked it to review and tidy up something that you
generated largely by yourself. If the former, disclose the tool use, but
otherwise, use need not be mentioned.

Here are some examples of contributions that demonstrate how to apply
the principles of this policy:

.. code-block:: text

  Examples to be provided


Use of LLMs to review merge requests
************************************

LLMs may be used to review merge requests by other contributors, but reviewers should not blindly
copy comments generated by the LLM without human review, and the submitter must have high
confidence that the LLM review comment is technically correct. Again, the principle of
*human in the loop* should apply.
