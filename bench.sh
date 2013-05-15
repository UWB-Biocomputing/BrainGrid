#!/bin/bash

# This script will run through the following operations: 
#		- Create unique benchmark directory
#		- Store system information
#		- Run all three test files on master branch and store outputs 
#		- Run all three test files on refactor branch and store outputs
#		- Diff the historyDump files between master and refactor branch
#		- Tar and Zip the files benchmark directory
# 
# 
# TO USE:
# 
# Don't forget to pull in the latest commits and don't forget to commit your 
# latest changes.  This is a stupid script and won't fail if git checkout
# fails.  The easiest method would be to do a fresh clone of the repository
# and then run the bench marking script.
# 
# Move this file to the BrainGrid Parent Directory (mv bench.sh ../../) before
# execution. This script assumes that the respository has been cloned into a
# directory named BrainGrid; if this is not the case, please edit the
# BRAINGRID variable.
# 
# Also, don't forget to make the scrip executable!
#
# If you are running this via ssh then use screen to detach the process so it 
# will continue to run in the background, and you can log out of the ssh 
# session.  
# 	$ screen
#	$ ./bench.sh > bench.out
#	Ctrl-A then Ctrl-D to detach the screen
#	Log out of ssh session
# 	Log in to new ssh session
#	$ screen -r 
# 
# For now this file is only in the root directory of the refactor branch.  It 
# will probably be moved to the 'bin' folder at some point.


# Create Unique Directory
###############################################################################
echo ""
echo ""
echo "Setting up Directories"
echo "---------------------------------------------------------------------------------"


HOST=`hostname`
TIME=`date +%m-%d-%H-%M`
BENCH_DIR=$HOST-bench-$TIME
BRAINGRID=BrainGrid

mkdir $BENCH_DIR
echo "$BENCH_DIR/"
mkdir $BENCH_DIR/master
echo "$BENCH_DIR/master"
mkdir $BENCH_DIR/refactor
echo "$BENCH_DIR/refactor/"

# Save System Info
cat /proc/cpuinfo > ./$BENCH_DIR/cpuinfo.out
cat /proc/meminfo > ./$BENCH_DIR/meminfo.out
echo "System info saved"

cd $BRAINGRID
git pull

# Bench Master Branch
###############################################################################
echo ""
echo ""
echo "Testing Master Branch"
echo "---------------------------------------------------------------------------------"
git checkout master

echo "compiling..."
make -s clean
make -s growth > /dev/null 2> /dev/null


echo "testing 'test.xml'..."
# test.xml
./growth -t test.xml > test.out
mv test.out ../$BENCH_DIR/master/test.out
mv tR_1.9--fE_0.98_historyDump.xml ../$BENCH_DIR/master/test-historyDump.xml


echo "testing 'test-medium.xml'..."
# test-medium.xml
./growth -t test-medium.xml > test-medium.out
mv test-medium.out ../$BENCH_DIR/master/test-medium.out
mv test-medium-historyDump.xml ../$BENCH_DIR/master/test-medium-historyDump.xml


echo "testing 'test-medium-2.xml'..."
# test-medium-2.xml
./growth -t test-medium-2.xml > test-medium-2.out
mv test-medium-2.out ../$BENCH_DIR/master/test-medium-2.out
mv test-medium-2-historyDump.xml ../$BENCH_DIR/master/test-medium-2-historyDump.xml



# Bench Refactor Branch
###############################################################################
echo ""
echo ""
echo "Testing Refactor Branch..."
echo "---------------------------------------------------------------------------------"
git checkout refactor

echo "compiling..."
make -s clean
make -s growth > /dev/null 2> /dev/null


echo "testing 'test.xml'..."
# test.xml
./growth -t ./config/test.xml > test.out
mv test.out ../$BENCH_DIR/refactor/test.out
mv tR_1.9--fE_0.98_historyDump.xml ../$BENCH_DIR/refactor/test-historyDump.xml

echo "testing 'test-medium.xml'..."
# test-medium.xml
./growth -t ./config/test-medium.xml > test-medium.out
mv test-medium.out ../$BENCH_DIR/refactor/test-medium.out
mv test-medium-historyDump.xml ../$BENCH_DIR/refactor/test-medium-historyDump.xml

echo "testing 'test-medium-2.xml'..."
# test-medium-2.xml
./growth -t ./config/test-medium-2.xml > test-medium-2.out
mv test-medium-2.out ../$BENCH_DIR/refactor/test-medium-2.out
mv test-medium-2-historyDump.xml ../$BENCH_DIR/refactor/test-medium-2-historyDump.xml


# Verify History files
###############################################################################
echo ""
echo ""
echo "Verifying History Dump Files..."
echo "---------------------------------------------------------------------------------"

cd ../$BENCH_DIR

DIFF=`diff ./master/test-historyDump.xml ./refactor/test-historyDump.xml`
if [ "$DIFF" != '' ]; then
	echo "ERROR: test-historyDump.xml are different"
else
	echo "test-historyDump.xml are identical"
fi


DIFF=`diff ./master/test-medium-historyDump.xml ./refactor/test-medium-historyDump.xml`
if [ "$DIFF" != '' ]; then
	echo "ERROR: test-medium-historyDump.xml are different"
else 
	echo "test-medium-historyDump.xml are identical"
fi


DIFF=`diff ./master/test-medium-2-historyDump.xml ./refactor/test-medium-2-historyDump.xml`
if [ "$DIFF" != '' ]; then
	echo "ERROR: test-medium-2-historyDump.xml are different"
else
	echo "test-medium-2-historyDump.xml are identical"
fi


# Compress Directory
###############################################################################

echo ""
echo ""
echo "Compressing Directory..."
echo "---------------------------------------------------------------------------------"

cd ../
tar -czvf $BENCH_DIR.tar.gz $BENCH_DIR

