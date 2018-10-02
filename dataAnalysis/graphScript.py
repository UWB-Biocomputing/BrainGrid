#!/usr/bin/python

#import everything needed
from optparse import OptionParser
import xml.etree.ElementTree as ET
import subprocess
import sys
import os


#parse args
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                    help = "xml file to read in", type="string", action="store")
parser.add_option("-g", "--gnuplot", action="store_false", dest="R")
parser.add_option("-r", "--R", action="store_true", dest="R")
(option, args) = parser.parse_args()

if(len(option.filename) <= 0):
    sys.exit(0)

R = False;

if(option.R):
    R = True;

#set up xml tree
tree = ET.parse(option.filename)
root = tree.getroot()

#create directory to dump images and files into
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

    if(R):
        command = 'r '+directory+'/'+child.get('name')+' '+directory+'/'+child.get('name')+'.png < rScript.R'
    else:
        command = 'gnuplot -e \"filename=\''+directory+'/'+child.get('name')+'\'; outputname=\''+directory+'/'+child.get('name')+'.png\'\" gnuplotScript.cfg'
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    tempOutput = process.communicate()[0]    
    os.remove(directory+'/'+child.get('name'))
