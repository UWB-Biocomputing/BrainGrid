# Contributing to The BrainGrid Project

Thank you for your interest in the BrainGrid Project, which includes both the BrainGrid simulator and Workbench software and data provenance system. This project operates with an [Apache 2.0 license](LICENSE.md) which provides wide reusability and adaptability, under the proviso of citing the originators and maintaining provenance information.

If you're interested in adapting this project for your own use, then please feel free to make your own copy of this repository and adapt it to your work. We would be greatly interested to learn about what you do, potentially incorporating your work back into this main repo. *Please cite us in your work*; the repo [README](README.md) has a DOI for that purpose.

If you're interested in contributing directly to this project, then please contact [Prof. Michael Stiber](mailto:stiber@uw.edu) and read the information below.

## Code of Conduct

This project and everyone participating in it is governed by the [Biocomputing Laboratory Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Workflow

Please read up on Github basics; seek the guidance of more senior lab members regarding how to get started. Of especial importance is that you *don't work directly on the master branch* (yes, there are exceptions, but they are few and far between). Instead, create a branch, do what you intend, *check that your haven't broken anything*, and then merge your branch into master. If you're unsure about doing such a merge, then discuss what you've done at a lab meeting or open a pull request (read more about [pull requests](http://help.github.com/pull-requests/)).

We are working on developing a Jenkins server to help validate changes, so that you'll more easily know whether what you've done passes all of our tests for correctly working (or, more pedantically, behaving in a manner consistent with the current release version). More on this to come later.

*Please document what you've done*, not only in your commit messages but also with useful comments in code and via changes to the github pages content in the docs directory.

## Coding Conventions

We are anal about some things. You have been warned.

  * We indent using *three spaces*. *Not tabs*. Spaces.
  * We use CamelCase naming, rather than underscores.
  * We put spaces after list items and method parameters (`f(a, b, c)`, not `f(a,b,c)`) and around operators (`x += 1`, not `x+=1`). We don't put spaces after or before parentheses (`f(a)`, not `f( a )`).
  * We avoid isolating curly braces on their own lines for loops and conditionals (except for right braces closing a code block); we do put isolated braces on their own lines for functions. Examples:
  ```C++
  int f(a)
  {
     return a;
  }
  ```
  ```C++
  if (x > m) {
     x--;
  } else {
     x++;
  }
  ```
  * We limit code to 80 character line lengths. You never know when someone will want to print something out on an [ASR-33 teletype](https://en.wikipedia.org/wiki/Teletype_Model_33).
  
  
