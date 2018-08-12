#!/bin/bash
# script created on: Fri Aug 03 16:03:03 PDT 2018


if [ "$#" -ne 0 ]; then
	echo "wrong number of arguments. expected 0"
	echo "usage:  ${0##*/}"
exit 1
fi

printf "version: 1\n" > ~/provis_v1_scriptStatus.txt 2> ~/provis_v1_scriptStatus.txt
printf "simExecutable: growth\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "simInputs: provis.xml\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "simOutputs: output.xml\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "endSimSpec: \n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: mkdir -p C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
mkdir "-p" "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: git clone C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
git "clone" "C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD" "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: cd C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
cd "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: git pull\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
git "pull" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: git log --pretty=format:'%%H' -n 1\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
git log --pretty=format:'%H' -n 1 > C:\Users\Max\provis_v1_SHA1Key.txt 2>> C:\Users\Max\provis_v1_SHA1Key.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: make -s clean\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
make "-s" "clean" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: make -s growth CUSEHDF5='no'\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
make "-s" "growth" "CUSEHDF5='no'" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: mkdir C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\\\results\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
mkdir "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\results" >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: mkdir -p C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\workbenchconfigfiles\NList\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
mkdir -p C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\workbenchconfigfiles\NList >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: mv -f C:/Users/Max/provis.xml C:/Users/Max/C:/Users/Max/Documents/DOCUMENTS/Braingrid-WD/workbenchconfigfiles/provis.xml\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
mv -f C:/Users/Max/provis.xml C:/Users/Max/C:/Users/Max/Documents/DOCUMENTS/Braingrid-WD/workbenchconfigfiles/provis.xml >> ~/provis_v1_output.txt 2>> ~/provis_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
printf "command: ./growth -t workbenchconfigfiles/provis.xml\ntime started: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
./growth -t workbenchconfigfiles/provis.xml >> C:\Users\Max\provis_v1_simStatus.txt 2>> C:\Users\Max\provis_v1_simStatus.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/provis_v1_scriptStatus.txt 2>> ~/provis_v1_scriptStatus.txt
