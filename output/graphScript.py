#!/usr/bin/python

#import everything needed
import xml.etree.ElementTree as ET
import subprocess
import sys
import os

if(len(sys.argv) != 2):
    print "\nError: not enough arguments!"
    sys.exit(0)

tree = ET.parse(sys.argv[1])
root = tree.getroot()

directory = 'graphs'

if not os.path.exists(directory):
    os.makedirs(directory)

for child in root:
    with open(directory+'/'+child.get('name'), 'w') as output:
        arr = child.text.split()
        rows = int(child.get('rows'))
        columns = int(child.get('columns'))
        for x in range(0, len(arr)):
            temp = str(x/columns) + ' ' + arr[x] + '\n'
            output.write(temp)
        output.close()

    command = 'gnuplot -e \"filename=\''+directory+'/'+child.get('name')+'\'; outputname=\''+directory+'/'+child.get('name')+'.png\'\" gnuplotScript.cfg'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    tempOutput = process.communicate()[0]
    os.remove(directory+'/'+child.get('name'))
