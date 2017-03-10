# BrainGrid Installation

BrainGrid is designed to be as easy to use as possible, but given its scope and flexibility, there are some tradeoffs. So, first things first:

1. **NVIDIA GPU**: If you want your simulator to run on GPUs, you MUST use an NVIDIA GPU that is CUDA capable. Check this [list](https://developer.nvidia.com/cuda-gpus). Of course, BrainGrid is totally open source, so if you wanted, you could make an OpenCL version and use that. But for the speedups that we desire, we found that CUDA was the most reasonable way to go.
2. **LINUX**: Currently, BrainGrid only works on Linux. Any distro that supports GNU-Make and your chosen NVIDIA graphics card (if going the GPU route) should work.

## Part 0 - Necessary Software

In order to run BrainGrid, you will need the following software:

- Linux (probably any distro is fine)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (only if you intend to use the GPU functionaility)
- Matlab or Octave (only if you want to view the output files using our scripts)

## Part 1 - Download 

In order to get started with BrainGrid, you will need to build it from scratch, which means getting its source from the [BrainGrid GitHub repository](https://github.com/UWB-Biocomputing/BrainGrid).

### Fork and clone BrainGrid

1. Have your [Github](https://github.com/) account ready (go over our [Git Crash Course](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Git-Crash-Course) if you are not familiar with Github)

2. To fork a repo, navigate to the [BrianGrid Github](https://github.com/UWB-Biocomputing/BrainGrid) page and click on the **Fork** button

3. Open terminal and go to the directory you plan to have BrainGrid placed. For example, under your home directory `$ cd ~`

   ```
   $ cd YOUR_PREFERED_PATH
   ```

4. To clone the forked repo to your local machine: [How](https://help.github.com/articles/fork-a-repo/)

   ```
   $ git clone https://github.com/YOUR_USERNAME/BrainGrid.git
   ```

5. Set up origin & upstream to keep your forked repo in sync: [Why](https://help.github.com/articles/configuring-a-remote-for-a-fork/)

- List the current configured remote repository for your fork. 

  When a repo is cloned, the default remote **origin** is your fork on Github, not BrainGrid repo it was forked from.
  ```
  $ git remote -v 
  origin  https://github.com/YOUR_USERNAME/public_html.git (fetch)
  origin  https://github.com/YOUR_USERNAME/public_html.git (push)
  ```

- Set BrainGrid as your new remote **upstream** in order to keep your local copy in sync with the BrainGrid.
  ```
  $ git remote add upstream https://github.com/UWB-Biocomputing/BrainGrid.git
  ```

- Verify the new remote repository you've specified for your fork. 
  ```
  $ git remote -v
  origin  https://github.com/YOUR_USERNAME/BrainGrid.git (fetch)
  origin  https://github.com/YOUR_USERNAME/BrainGrid.git (push)
  upstream        https://github.com/UWB-Biocomputing/BrainGrid.git (fetch)
  upstream        https://github.com/UWB-Biocomputing/BrainGrid.git (push)
  ```

  if you want to see more detail, do:

  ```
  $ git remote show origin
  $ git remote show upstream
  ```

- Syncing your fork by fetching from upstream BrainGrid repo: [How](https://help.github.com/articles/syncing-a-fork/) 

  ```
  $ git fetch upstream
  ```

   Merge the changes from `upstream/master` into your local `master` branch

  ```
  $ git checkout master
  $ git merge upstream/master
  ```

  Now your fork's `master` branch is in sync with the latest BrainGrid repo without losing your local changes. You are now all set to use BrainGrid!

### Contribute to BrainGrid

If you want to make changes and contribute to BrainGrid, we strongly recommend creating new branches on your end. This would make it easier for us to track changes and collaborate with other people.

1. List all branches in the repo 
   ```
   $ git branch
   ```
2. Create a new branch to work on [Why](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
   ```
   $ git checkout -b YOUR_BRANCH_NAME [BASE_BRANCH_NAME]
   ```

  (Note: BASE_BRANCH_NAME is optional and defaults to master)
3. Commit and push the changes you made locally to your own Github repository

   ```
   $ git status
   $ git add FILE_NAME
   $ git commit -m “MESSAGE”
   $ git push -u origin YOUR_BRANCH_NAME
   ```
4. Open a pull request when you are ready to let us know about the changes you've pushed to your repo on Github: [How](https://help.github.com/articles/about-pull-requests/)

- Go to your forked BrainGrid repo page on GitHub. 
- Click on **Pull Request** button in the repo header
- Click on the **Head Branch** dropdown and pick the branch you wish to merge with.
- Enter the **title** and **description** for your pull request. 
- Lastly, click on the green **Send pull request** button.
- Once a pull request is sent, collaborators can review and discuss the changes you made.

## Part 3 - Quick Sanity Test

As a quick start and sanity test, let's run a small, prepackaged simulation. Follow these steps:

1. At your terminal, go to BrainGrid directory
   ```
   $ cd ~/BrainGrid
   ```
2. Unless you have the necessary **HDF5** stuff installed, go into the Makefile and set **CUSEHDF5** to "no" to use default XML output. 
   - HDF5 is useful for making the data analysis easier for Matlab, which has native HDF5 support, after a simulation - especially a very long one; but it is fine to use the default XML output.
   - If you like to use HDF5 or have issues with using HDF5, see [Getting BrainGrid working with HDF5](https://github.com/UWB-Biocomputing/BrainGrid/wiki/Getting-BrainGrid-working-with-HDF5)
3. Compile the single threaded version
   ```
   $ make growth
   ```
4. Run it with one of our numerous test files 
   ```
   $ ./growth -t ./validation/test-small.xml
   ```
   **Note:** The run time of this test is small-ish on a fast computer (maybe a couple minutes), but this particular test also doesn't do much. The output will be mostly nothing - but it shouldn't crash or give you anything weird. If you want to run a real test, you could use test-small-connected.xml, but be warned: using the single threaded version of this (or any larger test) will result in hours of waiting, but the output will be much more interesting. You can take a look at the next section on Screen for how to deal with these wait times.

### Screen

When you run a simulation with BrainGrid, you may not want to wait around for it to run to completion. This is especially true if running it remotely. To help with this, you can use the built in 'screen' command in Linux.

The `screen` command will essentially allow you to start a simulation and then detach it so that it runs in the background.  This has the huge advantage of allowing you to log out of the server you are remotely connected to.  

#### Here is how you might go about doing this:

1. Log into the server of your choice 
   `$ ssh bobjoe@foo.bar.edu`

2. Change into the BrainGrid repository
   `$ cd ~/BrainGrid`

3. Compile the (single threaded) simulation
   `$ make growth`

4. Start a screen
   `$ screen`

5. Start the Simulation
   `$ ./growth -t ./validation/test-small-connected.xml`

6. Detach the screen by pressing the following key combinations:
   `"Ctrl+A"`  then `"Ctrl+D"`

7. Allow the simulation to run as long as you want. During this time, you can log out without any problem.

8. Reattach the screen whenever you want to check in on it.
   `$ screen -r`
   If it isn't done yet, detach again and come back later!
