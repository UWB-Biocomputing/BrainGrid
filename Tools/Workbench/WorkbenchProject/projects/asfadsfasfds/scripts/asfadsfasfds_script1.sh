#!/bin/bash
# script created on: Mon Jun 11 14:16:07 PDT 2018


if [ "$#" -ne 0 ]; then
	echo "wrong number of arguments. expected 0"
	echo "usage:  ${0##*/}"
exit 1
fi

printf "version: 1\n" > ~/asfadsfasfds_v1_scriptStatus.txt 2> ~/asfadsfasfds_v1_scriptStatus.txt
printf "simExecutable: growth\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "simInputs: asfadsfasfds.xml\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "simOutputs: output.xml\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "endSimSpec: \n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: mkdir -p C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
mkdir "-p" "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder" >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: cd C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
cd "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder" >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: git log --pretty=format:'%%H' -n 1\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
git log --pretty=format:'%H' -n 1 > C:\Users\Max\asfadsfasfds_v1_SHA1Key.txt 2>> C:\Users\Max\asfadsfasfds_v1_SHA1Key.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: make -s clean\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
make "-s" "clean" >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: make -s growth CUSEHDF5='no'\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
make "-s" "growth" "CUSEHDF5='no'" >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: mkdir C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder\\\results\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
mkdir "C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder\results" >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: mkdir -p C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder\workbenchconfigfiles\NList\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
mkdir -p C:\Users\Max\C:\Users\Max\Documents\DOCUMENTS\Braingrid-WD\simfolder\workbenchconfigfiles\NList >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: mv -f C:/Users/Max/asfadsfasfds.xml C:/Users/Max/C:/Users/Max/Documents/DOCUMENTS/Braingrid-WD/simfolder/workbenchconfigfiles/asfadsfasfds.xml\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
mv -f C:/Users/Max/asfadsfasfds.xml C:/Users/Max/C:/Users/Max/Documents/DOCUMENTS/Braingrid-WD/simfolder/workbenchconfigfiles/asfadsfasfds.xml >> ~/asfadsfasfds_v1_output.txt 2>> ~/asfadsfasfds_v1_output.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
printf "command: ./growth -t workbenchconfigfiles/asfadsfasfds.xml\ntime started: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
./growth -t workbenchconfigfiles/asfadsfasfds.xml >> C:\Users\Max\asfadsfasfds_v1_simStatus.txt 2>> C:\Users\Max\asfadsfasfds_v1_simStatus.txt
printf "exit status: $?\ntime completed: `date +%s`\n" >> ~/asfadsfasfds_v1_scriptStatus.txt 2>> ~/asfadsfasfds_v1_scriptStatus.txt
