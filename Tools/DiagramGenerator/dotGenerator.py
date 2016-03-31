# Python script for generating dot diagram for BrainGrid
# Pass in the file you want to "center" on and the script will create
# a dot file to generate a graph of all the files that the center recursively
# includes.


#TODO : Color! If an item is only included by one file, it should be the same color as that file. All files in a hierarchy should be the same color.
#TODO : Also make sure to add an ignore list (a list of files or subsystems to ignore).

import os, sys, re, fnmatch

##########GLOBALS##########

# List of classes and structs - populated by crawling the include web
classes = []

# List of tuples - first item is the principal item, second is the pointed to item
inheritance = []

# List of tuples - first item is the principal item, second is the one that is included by the first
includes = []


#########FUNCTIONS#########


def find_file(name, open_mode):
    print "Locating file: " + name
    matches = []
    for root, dirnames, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, name):
            matches.append(os.path.join(root, filename))
    if len(matches) is 0:
        print "No such file. Potentially not a problem."
        raise IOError
    else:
        print "Found it."
        return open(matches[0], open_mode)


def print_opening(dot_file):
    opening_blurb = "//BrainGrid Overview" + os.linesep + "//Written in the Dot language (See Graphviz" + os.linesep
    opening_digraph = "" + os.linesep + "digraph {" + os.linesep
    shape = "node [" + os.linesep + "shape = \"record\"" + os.linesep + "]" + os.linesep + os.linesep
    dot_file.write(opening_blurb)
    dot_file.write(opening_digraph)
    dot_file.write(shape)


def print_classes(dot_file):
    for c in classes:
        dot_file.write(c + "[" + os.linesep)
        dot_file.write("label = \"{\" + \"" + c + "\" + \"}\"" + os.linesep)
        dot_file.write("];" + os.linesep)


def print_layout(dot_file):
    header = "//###########################" + os.linesep + "//Layout" + os.linesep + "//###########################" + os.linesep
    rankdir = "rankdir = TB; // Rank Direction Top to Bottom" + os.linesep
    nodesep = "nodesep = 0.25; // Node Separation" + os.linesep
    ranksep = "ranksep = 0.25; // Rank Separation" + os.linesep

    dot_file.write(header)
    dot_file.write(rankdir)
    dot_file.write(nodesep)
    dot_file.write(ranksep)
    for c in inheritance:
        line = c[0] + " -> " + c[1] + " [arrowhead=empty];" + os.linesep
        dot_file.write(line)

    for c in includes:
        line = c[0] + " -> " + c[1] + " [arrowhead=ediamond];" + os.linesep
        dot_file.write(line)


def print_end(dot_file):
    dot_file.write("}//End digraph declaration" + os.linesep)


def print_file(dot_file):
    print "Generating dot file..."
    print_opening(dot_file)
    print_classes(dot_file)
    print_layout(dot_file)
    print_end(dot_file)


def is_inheritance(derived, base):
    """
    This function determines if the argument "derived" is inherited from the argument "base".
    """
    try:
        derived_file = find_file(derived + ".h", 'rb')
    except IOError as ex:
        return False
    # decide if the "derived" class actually derives from base
    # read file in line by line into a string.
    lines = [line for line in derived_file]
    derived_file.close()
    contents = ""
    for line in lines:
        contents += line

    # match the string's first occurrence of this: .*"class" <whitespace>+ <derived's name> <whitespace>+ ":" .* <base's name> .* "{"
    regex = '(class)(\s)+(' + derived + ')(\s)+(:)(.)*(' + base + ')(.)*\{'
    pattern = re.compile(regex, re.DOTALL)
    match_obj = pattern.search(contents)
    if match_obj:
        return True
    else:
        return False


def find_includes(includer):
    found_list = []
    for line in includer:
        pattern = re.compile('(\s)*#(\s)*include(\s)*"_*(\w)+\.h"')
        match_obj = pattern.search(line)
        if match_obj:
            matched_string = match_obj.group()  # # include   "BookStore.h"
            name_pattern = re.compile('"_*(\w)+\.h"')
            name = name_pattern.search(matched_string).group()  # "BookStore.h"
            name_split = name.split('"')
            name = name_split[1].split('.')[0]  # BookStore
            found_list.append(name)
    return found_list


def combine_cpp_and_h_files(file_name):
    """
    Finds the .cpp and .h files that correspond to the given name (name
    is given without a file extension) and combines their contents into a list
    of lines.
    """
    lines = []
    try:
        h = find_file(file_name + ".h", 'rb')
        h_lines = [line for line in h]
        h.close()

        for line in h_lines:
            lines.append(line)
    except IOError as ex:
        pass

    try:
        cpp = find_file(file_name + ".cpp", 'rb')
        cpp_lines = [line for line in cpp]
        cpp.close()

        for line in cpp_lines:
            lines.append(line)
    except IOError as ex:
        pass

    return lines


def crawl_web(center_file_as_list, center_file_name):
    """
    This function returns a list of lists. Each list being
    all the files that the first item in that list includes, not including .h files (which are
    essentially combined into their like-named .cpp).
    E.g.:
    [   [main, BookStore] , [BookStore, HashTable, Book, Customer, Transaction], [HashTable, etc.], etc.  ]
    """
    print "-----------Crawling the include paths.-------------"
    center_file_name = center_file_name.split("\\")[-1].split('.')[0]
    included = find_includes(center_file_as_list)
    if center_file_name in included:
        included.remove(center_file_name)
    center_as_list = [center_file_name]
    print "--------File " + center_file_name + " has the following dependencies: --------"
    for el in included:
        print el
        center_as_list.append(el)  # Put the center_file into the front of the list of included items
    found_list = []
    found_list.append(center_as_list)  # Put the list of included items into the found_list
    already_done = [center_file_name]
    countdown = 0
    while len(included) is not 0:
        for inc in included:
            if countdown >= len(included):
                break
            if inc in already_done:
                countdown += 1
            else:
                countdown = 0
                already_done.append(inc)
                inc_as_lines = combine_cpp_and_h_files(inc)
                next_includes = find_includes(inc_as_lines)
                print "----------File " + inc + " has the following dependencies: -----------"
                # remove duplicates
                next_includes = list(set(next_includes))
                if inc in next_includes:
                    next_includes.remove(inc)
                center_as_list = [inc]
                for el in next_includes:
                    print el
                    center_as_list.append(el)
                found_list.append(center_as_list)
        if countdown >= len(included):
            break
        else:
            included = next_includes

    return found_list


def map_inheritance_and_composition(files_to_check):
    print "Mapping the relationships as either inheritance or composition..."
    for layer in files_to_check:
        if len(layer) > 1:
            parent_name = layer[0]
            index = 0
            rest_of_layer = layer[1:]
            print "Mapping relationships for " + parent_name
            for item in rest_of_layer:
                relationship = (parent_name, item)
                if is_inheritance(parent_name, item) and not relationship in inheritance:
                    print parent_name + " INHERITS from " + item
                    inheritance.append(relationship)
                elif not relationship in includes and not relationship in inheritance:      # Don't include if already in inheritance
                    print parent_name + " DEPENDS on " + item
                    includes.append(relationship)


#############MAIN SCRIPT#############

# Get the input arg as the "center" file
if len(sys.argv) is not 3:
    print "USAGE: dotGenerator.py <FILE_NAME> <OUTPUT_NAME_WO_EXTENSION>"
    exit(0)
else:
    center_file_name = sys.argv[1]
    dot_file_name = sys.argv[2]

try:
    center_file = find_file(center_file_name, 'rb')
    extension = center_file.name.split('.')[-1]
    if extension != "cpp" and extension != "h":
        print "This only works for .cpp and .h files. Please pass me one of those."
        exit(-1)
except IOError as ex:
    print "File not found or some other IO error."
    exit(-1)

# Crawl through the web and find all the includes and inheritance
center_as_lines = combine_cpp_and_h_files(center_file_name.split("\\")[-1].split('.')[0])
files_to_check = crawl_web(center_as_lines, center_file.name)
center_file.close()
map_inheritance_and_composition(files_to_check)

# Print the dot file
dot_file = open(dot_file_name + ".dot", 'wb')
print_file(dot_file)
dot_file.close()
