# Contributing to The BrainGrid Project

Thank you for your interest in the BrainGrid Project, which includes both the BrainGrid simulator and Workbench software and data provenance system. This project operates with an [Apache 2.0 license](LICENSE.md) which provides wide reusability and adaptability, under the proviso of citing the originators and maintaining provenance information.

For people outside of the [UW Bothell Biocomputing laboratory](http://depts.washington.edu/biocomp/) (BCL), we use a [fork and pull development model](https://help.github.com/articles/about-collaborative-development-models/). If you're interested in adapting this project for your own use, then please feel free to make your own copy of this repository and adapt it to your work. We would be greatly interested to learn about what you do, potentially incorporating your work back into this main repo. *Please cite us in your work*; the repo [README](README.md) has a DOI for that purpose.

For UW Bothell students interested in working in the BCL, we use a [shared repository development model](https://help.github.com/articles/about-collaborative-development-models/). If you're interested in contributing directly to this project, then please contact [Prof. Michael Stiber](mailto:stiber@uw.edu) and read the information below.

## Code of Conduct

This project and everyone participating in it is governed by the [Biocomputing Laboratory Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Workflow

Please read up on Github basics (including [Managing your work on GitHub](https://help.github.com/categories/managing-your-work-on-github/)); seek the guidance of more senior lab members regarding how to get started. Of especial importance is that you *don't work directly on the master branch* (yes, there are exceptions, but they are few and far between). Instead, create a branch, do what you intend, *check that your haven't broken anything*, and then merge your branch into master. If you're unsure about doing such a merge, then discuss what you've done at a lab meeting or open a pull request (read more about [pull requests](http://help.github.com/pull-requests/)).

If you're creating a branch that is in response to an issue, then name the branch accordingly, i.e., "issue-3141". This implies a one-to-one correspondence between issues and branches. If you want to work on an issue and it seems pretty clear that it's a big undertaking, then talk with the group. Possibly, it will be a branch that exists for a while, and you may need to merge the master branch back into it multiple times as you work on it. But, it's also possible that the issue in question should really be broken into sub-issues that can be worked on separately. You can use [the GitHub syntax](https://help.github.com/articles/closing-issues-using-keywords/) to close issues directly from commits or pull requests upon merge into the master branch.

We use [HuBoard](https://huboard.com/UWB-Biocomputing/BrainGrid#/) to integrate GitHub issue tracking into a kanban system. *Please avoid using issue labels that start with a numeral*; those labels are used by HuBoard.

We are working on developing a Jenkins server to help validate changes, so that you'll more easily know whether what you've done passes all of our tests for correctly working (or, more pedantically, behaving in a manner consistent with the current release version). More on this to come later.

*Please document what you've done*, not only in your commit messages but also with useful comments in code and via changes to the github pages content in the docs directory.

## Coding Conventions

We are anal about some things. You have been warned.

  * We use `.cpp` and `.h` for our C++ code. We name files with exactly the same name (including capitalization) as the primary classes they define.
  * We indent using *three spaces*. *Not tabs*. Spaces.
  * We use [cC]amelCase naming, rather than underscores. Classes start with capital letters; functions and variables start with lower-case letters.
  * We put spaces after list items and method parameters (`f(a, b, c)`, not `f(a,b,c)`) and around operators (`x += 1`, not `x+=1`). We don't put spaces after or before parentheses (`f(a)`, not `f( a )`).
  * We like to [cuddle our braces](http://blog.gskinner.com/archives/2008/11/curly_braces_to.html), avoiding isolating curly braces on their own lines for loops and conditionals (except for right braces closing a code block, so there's a limit to how cuddly we are); we do put isolated braces on their own lines for functions. Examples:
  ```c++
  if (x > m) {
     x--;
  } else {
     x++;
  }
  ```
  ```c++
  int f(a)
  {
     return a;
  }
  ```
  * We use braces even when a code block is a single line, to prevent bugs when it (inevitably) later expands to multiple lines.
  * We limit code to 80 character line lengths. You never know when someone will want to print something out on an [ASR-33 teletype](https://en.wikipedia.org/wiki/Teletype_Model_33).
  * We help keep code clear to human readers. So `if (aPointerVar == nullptr)`, not `if (aPointerVar == 0)`; `if (!aBoolFlag)`, not `if (aBoolFlag == false)`; `if (aCharVar == '\0')`, not `if (aCharVar == 0)`.
  * We use `#pragma once` instead of `#define` guards.
  * We check our spelling. Just because it's code doesn't mean spelling isn't important.
  * We use an empty line between methods.
  * We use empty lines around multi-line blocks.
  * We use Unix end-of-line characters (`\n`).

