#USAGE: python dotGenerator.py input_file_name.cpp output_file_name
#
#Put this script into the same directory as the file you want to pass it
#and run it by passing in the file you want to start with and a name for the
#output. It will then start with that file and recursively look through
#directories from that location down for files that that file includes.
#
#For example, if you pass in BGDriver.cpp, it will look for all the files that
#BGDriver.cpp includes (including BGDriver.h, if it existed). It will map the
#relationships between those included files and BGDriver as either composition
#or inheritance, and then it will look through all the files that are included
#for their includes recursively. It will then generate several dot diagrams.
#
#It generates three dot diagrams based on the name you passed it: a top-level
#overview file which shows all the connections between all the files not
#included in the ignore list; an overview broken down into subsystems that it
#has attempted to identify, where the connections inside of the subsystems
#have been removed; and a subsystem diagram that shows only connections between
#subsystems at the system level (so no connections between individual files).
#
#It also generates a folder and places all the subsystems that it has identified
#into it. It has no good way of naming those, so it currently just goes through
#the alphabet and numbers, so the outputs will be A.dot, B.dot, etc.
#
#To ignore certain files - simply modify the ignore list near the top of this
#script.


import os
import sys
import re
import fnmatch
import shutil


##########GLOBALS##########

# List of allowable file types.
allowable_file_types = [".cpp", ".cc", ".cu", ".h", ".c"]

# List of classes and structs - populated by crawling the include web -> a list of dictionaries
# of the form: {"name":<name>, "style":"filled", "color":"0.1 0.3 0.8" etc.}
classes = []

# List of tuples - first item is the principal item, second is the pointed to item
inheritance = []

# List of tuples - first item is the principal item, second is the one that is included by the first
includes = []

# List of items to ignore, including directories
# NOTE : Adding directories to this list will make it so that ANY directories with that name are skipped. So
# ./Blah/Foo/old will be skipped AND ./old will be skipped if "old" is added to this list.
ignores = ["old"]

# List of file extensions to ignore - modified by the script itself
extension_ignores = []

# Names for subsystem - defaults to these characters, one at a time. Change to a real list of names if you want.
# But probably you just want to manually edit the DOT file to use reasonable subsystem names. The script has no real
# way of figuring out reasonable names.
sub_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# List of tuples from each subgraph. Used for making sure that if a relationship is printed to the DOT file inside of
# a subgraph declaration, it isn't also printed later on in the subgraph connection layout area.
subgraph_totals = []

# List of strings. Each string is the text that is printed to a DOT file to generate a subgraph. This list is used
# to make all the subsystem files if requested by the user.
subgraph_texts = []

# There are lots of allowable colors in dot - you can also specify them by their RGB values, e.g. (#0FA23C)
SUBSYSTEM_COLORS = [
    "aquamarine",
    # "blue",  # Too dark
    # "chartreuse", # Too bright
    "coral",
    "cyan",
    "darkgoldenrod",
    "darkorange",
    "darksalmon",
    "darkturquoise",
    # "deeppink",  # Way too bright
    # "firebrick",  # A little too dark
    # "forestgreen", # Too dark
    "gold",
    "hotpink",
    "indianred",
    "khaki",
    # "lightcyan",  # A little too light
    "lightgoldenrod",
    "limegreen",
    "mediumseagreen",
    "mediumturquoise"
    # "navyblue",  # Way too dark
    "olivedrab",
    "orange",
    "orangered",
    "palegreen",
    "paleturquoise",
    # "red",  # A little too intense
    "seagreen",
    "springgreen",
    "steelblue",
    "skyblue",
    "tomato",
    "violetred",
    "wheat",
    "yellowgreen"
]

# A dictionary of subsystem_names to subsystem_colors; populated during the execution of the script
subsystem_color_map = {}

# A reverse dictionary for subsystem_color_map
subsystem_color_map_inverse = {}

# A dictionary of subsystem directory names to their actual systems
# (that is, {'common' : [file_A, file_B], 'etc.' : etc }
global_dict_subsystems = {}

# A dictionary (used as a hash table) for storing files we have already looked up.
__file_hash = {}

#########FUNCTIONS#########


# Printing functions

def print_all_subsystems():
    dir_name = "dot_subsystems"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for graph_text in subgraph_texts:
        f_name = graph_text.strip().split(os.linesep)[0].split(' ')[1][7:]      # This is a bit of a hack...
        f_name_with_dir = os.path.join(dir_name, f_name + ".dot")
        f = open(f_name_with_dir, 'wb')
        f.write("digraph{")
        f.write(graph_text)
        f.write("}//end digraph")
        f.close()


def print_block_layout(block_file):
    # Form all the inter-system relationships
    all_relationships = inheritance + includes
    inter_system_relationships = []
    for relationship in all_relationships:
        tup0_system = get_color_from_name(relationship[0])
        tup1_system = get_color_from_name(relationship[1])
        if tup0_system != tup1_system:
#            inter_system_relationships.append((relationship[0], relationship[1], tup0_system))
            inter_system_relationships.append(relationship)

    temp = []
    # For each inter-system relationship,
    for relationship in inter_system_relationships:
        inter_system_relationships.remove(relationship)

        # Replace the first item in that relationship with the master node for the subsystem it belongs to
        r0 = relationship[0]
        color = get_color_from_name(r0)
        if color is None:
            node_name0 = r0
        else:
            node_name0 = subsystem_color_map_inverse.get(color)

        # Replace the second item in that relationship with the master node for the subsystem it belongs to
        r1 = relationship[1]
        color = get_color_from_name(r1)
        if color is None:
            node_name1 = r1
        else:
            node_name1 = subsystem_color_map_inverse.get(color)

        inter_relationship = (node_name0, node_name1, get_color_from_name(r0))
        temp.append(inter_relationship)

    # Remove any duplicate tuples
    inter_system_relationships = list(set(temp))

    # Now Formulate the string for each subgraph and print it to the file
    print_layout_boilerplate(block_file, 'b')

    current_color = None
    for r in inter_system_relationships:
        color = r[2]
        if not color and current_color != "black":
            current_color = "black"
            block_file.write(os.linesep + "edge [color=black];" + os.linesep)
        elif color and current_color != color:
            current_color = color
            block_file.write(os.linesep + "edge [color=" + color + "];" + os.linesep)
        line = r[0] + " -> " + r[1] + " [arrowhead=ediamond];" + os.linesep
        block_file.write(line)


def print_classes(dot_file, sys_overview_file, block_file, use_old_style_systems=False):
    subgraphs = get_subgraphs()

    # Now actually write the information to the file
    sub_name_index = 0
    for sub in subgraphs:
        sub_name = sub_names[sub_name_index] if use_old_style_systems else get_sub_name_new_style([item['name'] for item in sub])

        print_subgraph(dot_file, sub_name, sub, 'd')
        print_subgraph(sys_overview_file, sub_name, sub, 'o')
        print_subgraph(block_file, sub_name, sub, 'b')

        sub_name_index = 0 if sub_name_index >= (len(sub_names) - 1) else sub_name_index + 1


def print_end(dot_file, sys_overview_file, block_file):
    line = "}//End digraph declaration" + os.linesep
    dot_file.write(line)
    sys_overview_file.write(line)
    block_file.write(line)


def print_file(dot_file, sys_overview_file, block_file, use_old_style_systems=False):
    print "Generating dot files..."
    print_opening(dot_file, sys_overview_file, block_file)
    print_classes(dot_file, sys_overview_file, block_file, use_old_style_systems)
    print_layout(dot_file, sys_overview_file, block_file)
    print_end(dot_file, sys_overview_file, block_file)


def print_layout(dot_file, sys_overview_file, block_file):
    print_layout_boilerplate(dot_file, 'd')
    print_layout_boilerplate(sys_overview_file, 'o')

    current_color = None

    for c in inheritance:
        if c not in subgraph_totals:
            # Print the edge color if it isn't already that color
            color = get_color_from_name(c[0])
            if not color and current_color != "black":
                current_color = "black"
                dot_file.write(os.linesep + "edge [color=black];" + os.linesep)
                sys_overview_file.write(os.linesep + "edge [color=black];" + os.linesep)
            elif color and current_color != color:
                current_color = color
                color_line = os.linesep + "edge [color=" + color + "];" + os.linesep
                dot_file.write(color_line)
                sys_overview_file.write(color_line)

            line = c[0] + " -> " + c[1] + " [arrowhead=empty];" + os.linesep
            dot_file.write(line)
            sys_overview_file.write(line)

    for c in includes:
        if c not in subgraph_totals:
            # Print the edge color if it isn't already that color
            color = get_color_from_name(c[0])
            if not color and current_color != "black":
                current_color = "black"
                dot_file.write(os.linesep + "edge [color=black];" + os.linesep)
                sys_overview_file.write(os.linesep + "edge [color=black];" + os.linesep)
            elif color and current_color != color:
                current_color = color
                color_line = os.linesep + "edge [color=" + color + "];" + os.linesep
                dot_file.write(color_line)
                sys_overview_file.write(color_line)

            line = c[0] + " -> " + c[1] + " [arrowhead=ediamond];" + os.linesep
            dot_file.write(line)
            sys_overview_file.write(line)

    print_block_layout(block_file)


def print_layout_boilerplate(dot_file, behavior='d'):
    header = "//-------LAYOUT OF RELATIONSHIPS BETWEEN SUBGRAPHS------//" + os.linesep
    rankdir = "rankdir = BT; // Rank Direction Top to Bottom" + os.linesep
    if behavior == 'd':
        nodesep = ranksep = 0.02 * len(classes)
    elif behavior == 'b':
        nodesep = ranksep = 0.005 * len(classes)
    else:
        nodesep = 0.005 * len(classes)
        ranksep = 0.02 * len(classes)
    nodesep = "nodesep = " + str(nodesep) + "; // Node Separation" + os.linesep
    ranksep = "ranksep = " + str(ranksep) + "; // Rank Separation" + os.linesep

    concentrate = "concentrate = true;" + os.linesep

    dot_file.write(header)
    dot_file.write(rankdir)
    dot_file.write(nodesep)
    dot_file.write(ranksep)
    dot_file.write(concentrate)
    dot_file.write(os.linesep)


def print_opening(dot_file, sys_overview_file, block_file):
    opening_blurb = "//BrainGrid Overview" + os.linesep + "//Written in the Dot language (See Graphviz)" + os.linesep
    opening_digraph = "" + os.linesep + "digraph {" + os.linesep + os.linesep + os.linesep

    dot_file.write(opening_blurb)
    dot_file.write(opening_digraph)

    sys_overview_file.write(opening_blurb)
    sys_overview_file.write(opening_digraph)

    block_file.write(opening_blurb)
    block_file.write(opening_digraph)


def print_subgraph_layout(subgraph_inheritance, subgraph_includes, behavior='d'):
    header = \
        "" + os.linesep + os.linesep + "\t\t//------LAYOUT FOR SUBGRAPH------" + os.linesep + os.linesep + os.linesep
    rankdir = "\t\trankdir = BT; // Rank Direction Bottom to Top" + os.linesep
    nodesep = ranksep = 0.02 * len(classes)
    nodesep = "\t\tnodesep = " + str(nodesep) + "; // Node Separation" + os.linesep
    ranksep = "\t\tranksep = " + str(ranksep) + "; // Rank Separation" + os.linesep + os.linesep

    to_print = header + rankdir + nodesep + ranksep

    to_print += "" + os.linesep + "\t\t//INHERITANCE//" + os.linesep + os.linesep

    # Sort for human readability of the DOT file
    subgraph_inheritance = sorted(subgraph_inheritance)
    subgraph_includes = sorted(subgraph_includes)

    # Add each relationship tuple to the subgraph_totals so that we don't print them again later on.
    for s in (subgraph_inheritance + subgraph_includes):
        subgraph_totals.append(s)

    # Print each tuple in subgraph_inheritance, and add a line between groups of them
    last_c = None
    for c in subgraph_inheritance:
        if last_c is None or last_c[0] != c[0]:
            to_print += "" + os.linesep
        last_c = c
        if behavior == 'd':
            to_print += "\t\t" + c[0] + " -> " + c[1] + " [arrowhead=empty];" + os.linesep
        else:
            to_print += "\t\t" + c[0] + " -> " + c[1] + " [style=invis];" + os.linesep

    to_print += "" + os.linesep + os.linesep + "\t\t//COMPOSITION//" + os.linesep + os.linesep

    # Print each tuple in subgraph_includes, and add a line between groups of them
    last_c = None
    for c in subgraph_includes:
        if last_c is None or last_c[0] != c[0]:
            to_print += "" + os.linesep
        last_c = c
        if behavior == 'd':
            to_print += "\t\t" + c[0] + " -> " + c[1] + " [arrowhead=ediamond];" + os.linesep
        else:
            to_print += "\t\t" + c[0] + " -> " + c[1] + " [style=invis];" + os.linesep

    return to_print


def print_subgraph(dot_file, sub_name, sub, behavior='d'):
    """
    Writes the given subgraph to the given file.
    :param dot_file: The file to write to
    :param sub_name: The subgraph's name
    :param sub: The subgraph to print
    :param behavior: 'd' = dot_file; 'o' = sys_overview; 'b' = block_file
    :return: Nothing
    """

    to_print = "" + os.linesep + os.linesep

    if len(sub) is 1:
        to_print += "\tsubgraph " + sub_name + " {" + os.linesep
    else:
        to_print += "\tsubgraph cluster" + sub_name + " {" + os.linesep

    color = sub[0].get("color")

    # Update the dictionary
    subsystem_color_map[sub_name] = color
    subsystem_color_map_inverse[color] = sub_name

    if color is not None:
        to_print += "\t\tcolor = " + color + os.linesep
        to_print += "\t\tnode [shape = record, color = " + color + "];" + os.linesep + os.linesep
    else:
        to_print += "\t\tnode [shape = record];" + os.linesep + os.linesep + os.linesep

    # Alphabetize the subgraph's nodes for human-readability of the DOT file
    sub = sorted(sub)

    # Print the subgraph's nodes
    for c in sub:
        to_print += "\t\t" + c.get("name") + "[label = " + c.get("name") + ", style = filled"
        to_print += "];" + os.linesep

    # If block diagram, print the master node for the subgraph
    if behavior == 'b' and len(sub) > 1:
        if sub[0].get("color") is not None:
            to_print += "\t\t" + sub_name + "[label = \"" + sub_name + " (" + color + ")\"" + ", style = filled"
        else:
            to_print += "\t\t" + sub_name + "[label = " + sub_name + ", style = filled"
        to_print += "];" + os.linesep

    # Print the subgraph's layout
    names_in_this_subgraph = [d.get("name") for d in sub]
    subgraph_inheritance = \
        [tup for tup in inheritance if tup[0] in names_in_this_subgraph and tup[1] in names_in_this_subgraph]
    subgraph_includes = \
        [tup for tup in includes
         if tup[0] in names_in_this_subgraph and tup[1] in names_in_this_subgraph and tup not in subgraph_inheritance]

    layout = print_subgraph_layout(subgraph_inheritance, subgraph_includes, behavior)
    to_print += layout

    to_print += "\t}//end subgraph " + sub_name + os.linesep

    dot_file.write(to_print)

    if behavior == 'd':
        if len(sub) > 1:
            subgraph_texts.append(to_print)


# Other Functions


def color_subsystems(subsystems, use_old_discover_mode=False):
    """
    Colors the subsystems that are bigger than a single item by modifying the "classes" global.
    :param subsystems: A list of all the file names lumped into subsystems. The exact nature of this piece of data
    depends on use_old_discover_mode -> if true, this argument should be a list of the form:
    [[file_A, file_B, file_C], [file_D, file_E], etc.] where files A through C are one subsystem etc.
    If use_old_discover_mode is false, this argument should be a dictionary of the form:
    {(system_A_'s_name: [file A, file B, file C]), (system_B_'s_name: [file_D, file_E]), etc.}
    :param use_old_discover_mode: Whether or not subsystems was generated using the old-style generation method.
    :return: Nothing
    """
    sys_index = 0
    iterable = subsystems if use_old_discover_mode else subsystems.itervalues()
    for sys in iterable:
        if (len(sys) > 1 and use_old_discover_mode) or (len(sys) > 0 and not use_old_discover_mode):
            list_of_each_dictionary = [item for item in classes if item.get("name") in sys]
            # update each dictionary with the new color
            for dic in list_of_each_dictionary:
                dic["color"] = SUBSYSTEM_COLORS[sys_index]
            sys_index = 0 if sys_index >= (len(SUBSYSTEM_COLORS) - 1) else sys_index + 1


def combine_cpp_and_h_files(file_name, map_subsystems_too=False):
    """
    Finds the .cpp/.cu and .h files that correspond to the given name (name
    is given without a file extension) and combines their contents into a list
    of lines.
    """
    lines = []
    file_paths = []

    for extension in allowable_file_types:
        if extension in extension_ignores:
            continue

        try:
            f = find_file(file_name + extension, 'rb')
            f_lines = [line for line in f]
            file_paths.append(f.name)
            f.close()
            for line in f_lines:
                lines.append(line)
        except IOError as ex:
            pass

    if map_subsystems_too and len(file_paths) > 0:
        map_directories(file_paths)

    return lines


def crawl_web(center_file_as_list, center_file_name, map_systems_too=False):
    """
    This function returns a list of lists. Each list being
    all the files that the first item in that list includes, not including .h files (which are
    essentially combined into their like-named .cpp).
    E.g.:
    [   [main, BookStore] , [BookStore, HashTable, Book, Customer, Transaction], [HashTable, etc.], etc.  ] <- where main includes BookStore
    and BookStore includes HashTable, Book, Customer, and Transaction, etc.
    """

    center_file_name = center_file_name.split(os.sep)[-1].split('.')[0]  # Strip the extension from the name: BGDriver.cpp -> BGDriver

    to_ret = []
    stack = []

    # Create list of includes for center file and put center_file into the front of that list
    includes = find_includes(center_file_as_list)
    if center_file_name in includes:
        includes.remove(center_file_name)
    includes.insert(0, center_file_name)

    # Push that list onto stack
    stack.append(includes)

    # While stack is not empty
    while len(stack) is not 0:
        # l = stack.pop()
        el = stack.pop()

        # to_ret.append(l)
        to_ret.append(el)

        # for e in l:
        for e in el:
            # if e is not the first item in any list in to_ret (since we've already done it if it is):
            lists_that_e_is_first_item_in = \
                [group for group in to_ret if len(group) > 0 and group[0] == e]
            if len(lists_that_e_is_first_item_in) is 0:
                # create e's list (and maybe map e to its subsystem, depending on settings)
                e_as_lines = combine_cpp_and_h_files(e, map_systems_too)
                e_list = find_includes(e_as_lines)

                # put e at front of e's list
                if e in e_list:
                    e_list.remove(e)
                e_list.insert(0, e)

                # stack.push(e's list)
                stack.append(e_list)

    return to_ret


def create_subgraphs(subsystems, use_old_discover_mode=False):
    """
    Changes the layout of the graph to lump all the subsystems together by modifying the "classes" global.
    :param subsystems: A list of all the file names lumped into subsystems. The exact nature of this piece of data
    depends on use_old_discover_mode -> if true, this argument should be a list of the form:
    [[file_A, file_B, file_C], [file_D, file_E], etc.] where files A through C are one subsystem etc.
    If use_old_discover_mode is false, this argument should be a dictionary of the form:
    {(system_A_'s_name: [file A, file B, file C]), (system_B_'s_name: [file_D, file_E]), etc.}
    :param use_old_discover_mode: Whether or not subsystems was generated using the old-style generation method.
    :return: Nothing
    """
    iterable = subsystems if use_old_discover_mode else subsystems.itervalues()
    for j, sub in enumerate(iterable):
        list_of_each_dictionary = [item for item in classes if item.get("name") in sub]
        for dic in list_of_each_dictionary:
            dic["subgraph"] = j


def find_file(name, open_mode):
    """
    Searches from the script's current directory down recursively through all files under it until it finds
    the given file.
    :param name:
    :param open_mode:
    :return:
    """
    # All useful files have already been hashed. So just check the hash.
    if __file_hash.get(name):
        return open(__file_hash[name], open_mode)
    else:
        raise IOError


def find_includes(includer):
    found_list = []
    inside_multi_line_comment = False
    for line in includer:
        if re.compile('.*(/\*).*').search(line):
            inside_multi_line_comment = True
        if re.compile('.*(\*).*').search(line):
            inside_multi_line_comment = False

        if inside_multi_line_comment:
            continue

        pattern_exp = '(\s)*#(\s)*include(\s)*"_*(\w)+\.h"'
        pattern = re.compile(pattern_exp)
        exclude = re.compile('.*//' + pattern_exp)
        include = re.compile(pattern_exp + '.*//' + pattern_exp)
        match_obj = pattern.search(line)
        if (match_obj and not exclude.search(line)) or include.search(line):
            matched_string = match_obj.group()  # # include   "BookStore.h"
            name_pattern = re.compile('"_*(\w)+\.h"')
            name = name_pattern.search(matched_string).group()  # "BookStore.h"
            name_split = name.split('"')
            if name_split[1] not in ignores:
                name = name_split[1].split('.')[0]  # BookStore
                found_list.append(name)

    return found_list


def form_subsystems_from_inheritance(subsystems):
    """
    Forms a list of lists ('subsystems') from the input one such that inheritance hierarchies are each a separate
    subsystem.
    :param subsystems:
    :return:
    """

    for a_inherits_from_b in inheritance:
        A = a_inherits_from_b[0]
        B = a_inherits_from_b[1]
        to_merge = []

        for sub in subsystems:
            # Go through the subsystems and pull out all those that contain B
            if B in sub:
                subsystems.remove(sub)
                to_merge += sub

        # Merge all those subsystems together, removing any duplicates.
        to_merge = list(set(to_merge))

        # Now go through and find all the subsystems with A and merge B's new subsystem into those
        to_merge_including_A = []

        for sub in subsystems:
            if A in sub:
                subsystems.remove(sub)
                to_merge_including_A += sub

        to_merge += to_merge_including_A
        to_merge = list(set(to_merge))
        subsystems.append(to_merge)

    return subsystems


def form_subsystems_from_inclusions_in_other_subsystems(subsystems):
    """
    Check each subsystem A to see if any other subsystem B includes items from A,
    if B includes any items from A and it is the ONLY subsystem that includes any items from A,
    merge A into B.

    E.g. Model, in subsystem B, includes Coordinate, which is its own subsystem A. (Model in B, Coordinate = A).
    B is the ONLY subsystem that includes anything in A, so A should be merged INTO B.
    :param subsystems:
    :return:
    """
    for sys_A in subsystems:
        systems_that_include_from_A = []
        # Check each subsystem A
        # to see if any other subsystem B includes item(s) from A
        for sys_B in subsystems:
            # If B includes any items from A,
            if sys_B == sys_A:
                continue
            elif list_includes_any_items_from_other_list(sys_B, sys_A):
                systems_that_include_from_A.append(sys_B)

        # If the list of systems_that_include_from_A contains any systems that are a subset of another, remove those.
        for potential_subset in systems_that_include_from_A:
            for other in systems_that_include_from_A:
                if potential_subset == other:
                    continue
                elif set(potential_subset).issubset(set(other)) and potential_subset in systems_that_include_from_A:
                    systems_that_include_from_A.remove(potential_subset)

        # If B is the ONLY subsystem that includes any items from A,
        if len(systems_that_include_from_A) is 1 and len(systems_that_include_from_A[0]) is not 1:
            # Merge A and B.
            subsystems.remove(sys_A)
            subsystems.append(list(set(sys_A + systems_that_include_from_A[0])))
    return subsystems


def form_subsystems_from_single_includers(subsystems):
    """
    For each item, if that item does not include anything, check if it is only included by one file.
    :param subsystems:
    :return:
    """
    relationships = includes + inheritance

    for sys in subsystems:
        if len(sys) is not 1:
            continue
        else:
            item = sys[0]

        item_includes_anything = False
        for tup in relationships:
            if tup[0] is item:
                item_includes_anything = True

        if item_includes_anything:
            continue
        else:
            all_files_that_include_this_one = [r[0] for r in relationships if r[1] == item]
            if len(all_files_that_include_this_one) is 1:
                # There is only one file that includes item - lump item with that one file
                for system in subsystems:
                    if all_files_that_include_this_one[0] in system:
                        system.append(item)
    return subsystems


def get_color_from_name(name):
    sub_systems = get_subgraphs()
    for sub in sub_systems:
        for d in sub:
            if d.get("name") == name:
                return d.get("color")
    return None


def get_subgraphs():
    """
    Returns a list of lists. Each list is a subgraph (represented as a list of dictionaries).
    :return: A list of lists of dictionaries.
    """
    subgraph_list = [c.get("color") for c in classes if c.get("color") is not None]
    subgraphs = []

    # Add to subgraphs all the lists of actual subgraphs
    for c in subgraph_list:
        sub = [cl for cl in classes if cl.get("color") == c and cl]
        if sub not in subgraphs:
            subgraphs.append(sub)

    # Now add to subgraphs all the items (as lists) that don't belong to a subsystem
    for c in classes:
        if c.get("color") is None:
            sub = [c]
            subgraphs.append(sub)

    return subgraphs


def get_sub_name_new_style(sub):
    """
    Gets the name of the passed in subgraph. The subgraph that is passed in is a list of names.
    :param sub:
    :return: The name of the passed in subgraph.
    """
    for name in global_dict_subsystems.iterkeys():
        system = global_dict_subsystems[name]
        if set(system) == set(sub):
            return name
    return "NAME_ERROR"


def get_top_files():
    """
    Returns a list of lists. Each list is all the top files from a subsystem, where a top file is defined as one which
    is included by nothing else in its own subsystem.
    :return:
    """
    top_levels = []
    subsystems = list(set([d.get("color") for d in classes if d.get("color") is not None]))
    relationships = includes + inheritance
    f = lambda tup, sub: tup[0] in sub and tup[1] in sub    # lambda for filtering out relationships based on subsystem
    for s in subsystems:
        subsystem = [c.get("name") for c in classes if c.get("color") == s]

        # The top level files from a subsystem are what we are trying to find here. Those are files which are not
        # included by any other files in their subsystem.

        # so filter out all the relationships that include items from other subsystems
        subsystem_relationships = [t for t in relationships if f(t, subsystem)]

        # then filter out of the subsystem all the items that are included by other items in the subsystem
        # that is, for each item in the subsystem, if that item appears in the second position of a relationship
        # of items in the subsystem, it is out.
        top_lambda = lambda c, sub_rel: len([t for t in sub_rel if t[1] == c]) is 0
        top_files = [c for c in subsystem if top_lambda(c, subsystem_relationships)]

        # the remaining items are top level items for the subsystem.
        top_levels.append(top_files)
    return top_levels


def hash_all_files():
    """
    Walks through all the directories from working one down and hashes those files that exist.
    :return: Nothing
    """
    dir_tree = os.walk('.')
    for root, dirnames, filenames in dir_tree:
        d = os.path.basename(root)
        if d in ignores:
            continue
        else:
            # Hash all the useful files in this directory
            files_to_hash = [f_name for f_name in filenames if '.' + f_name.split('.')[-1] in allowable_file_types]

            for f_name in files_to_hash:
                __file_hash[f_name] = os.path.join(root, f_name)


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
    pattern = re.compile(regex)
    match_obj = pattern.search(contents)
    if match_obj:
        return True
    else:
        return False


def list_includes_any_items_from_other_list(list_a, list_b):
    """
    This method doesn't do at all what it sounds like.

    It checks list_a and returns True if any of the items in it include any of the items in list b.
    That is, if any item in list_a inherits from or includes any item in list_b, this method returns True.
    False otherwise.
    :param list_a:
    :param list_b:
    :return:
    """
    all_includes = includes + inheritance

    for a in list_a:
        for b in list_b:
            tup = (a, b)
            if tup in all_includes:
                return True
    return False


def map_directories(file_paths):
    """
    Maps the subsystems globally using the new method of system detection (directory structures).
    :param file_paths: The file paths of each file that corresponds to this class/module.
    :return:
    """
    # The module can only be a part of a single subsystem, so decide by majority vote which path makes the most sense
    votes = {}
    for path in file_paths:
        path_minus_name = os.sep.join(path.strip().split(os.sep)[0:-1])
        votes[path_minus_name] = votes[path_minus_name] + 1 if votes.has_key(path_minus_name) else 1

    # Get the path that is used the majority of the files in the module or the last one if tied
    item_path = ""
    total_votes = 0
    for key in votes.iterkeys():
        if votes[key] >= total_votes:
            total_votes = votes[key]
            item_path = key

    folder = item_path.split(os.sep)[-1]
    file_name = file_paths[0].split(os.sep)[-1].split('.')[0]
    if global_dict_subsystems.has_key(folder):
        global_dict_subsystems[folder].append(file_name)
    else:
        global_dict_subsystems[folder] = [file_name]


def map_inheritance_and_composition(list_of_include_groups, use_old_discovery_mode):
    """
    This function maps the relationships between the files which are related and fills the global
    "includes" and "inheritance" lists with tuples of the form: (includer, included).
    This function also populates the "classes" list.

    :param list_of_include_groups: A list of lists, each of the form [file_name_A, file_name_B, file_name_C, etc.] where
    file_name_B and file_name_C, etc. are all included BY file A.
    :param use_old_discovery_mode: Whether or not to use the old way of discovering subsystems (heuristics). The new
    way uses the directory structure to determine subsystems.
    """
    print "Mapping relationships and identifying subsystems..."
    for include_group in list_of_include_groups:
        # For each include_group, determine the type of relationship it is
        if len(include_group) > 1:
            parent_name = include_group[0]

            # Add this group's head to the list of classes already found
            if {"name": parent_name} not in classes:
                classes.append({"name": parent_name})

            rest_of_layer = include_group[1:]
            print "Mapping relationships for " + parent_name
            for item in rest_of_layer:
                # If the item is not in the list of classes, add it
                if {"name": item} not in classes:
                    classes.append({"name": item})

                # Determine the type of relationship between the parent and this item
                relationship = (parent_name, item)
                if is_inheritance(parent_name, item) and not relationship in inheritance:
                    print parent_name + " INHERITS from " + item
                    inheritance.append(relationship)
                elif relationship not in includes and not relationship in inheritance:  # Don't include if already in inheritance
                    print parent_name + " DEPENDS on " + item
                    includes.append(relationship)

    # At this point, the "classes" list is filled with all the files that this script has examined, and the
    # "inheritance" and "includes" lists are also filled with the correct relationships. Use them to determine
    # subsystems and provide them with the correct colors accordingly.
    map_subsystems(use_old_discovery_mode)


def map_subsystems(use_old_discovery_mode=False):
    """
    Walks through the three global lists (inheritance, includes, and classes) and determines what subsystem each
    item belongs to. Adds that information to the "classes" list.
    :param use_old_discovery_mode: Whether or not the subgraphs should be made by the old way of discovering them.
    """
    if use_old_discovery_mode:
        # For each item in "classes", make a group
        subsystems = [[item.get("name")] for item in classes]

        # Now merge some of those groups into others to form the various subsystems

        # All inheritance hierarchies form a subsystem
        subsystems = form_subsystems_from_inheritance(subsystems)

        # For each item, if that item does not include anything, check if it is only included by one file.
        # If so, it should be lumped with that file.
        subsystems = form_subsystems_from_single_includers(subsystems)

        # Check each subsystem A to see if any other subsystem B includes items from A,
        # if B includes any items from A and it is the ONLY subsystem that includes any items from A,
        # merge A and B.
        # Coordinate should therefore be included in that subsystem.
        # Do it several times.. Should figure out how to make this recursive or something so that it does it only as many
        # times as is necessary... but whatever.
        for i in range(0, 10):
            subsystems = form_subsystems_from_inclusions_in_other_subsystems(subsystems)

        color_subsystems(subsystems, use_old_discovery_mode)
        create_subgraphs(subsystems, use_old_discovery_mode)

    else:
        color_subsystems(global_dict_subsystems)
        create_subgraphs(global_dict_subsystems)


def remove_extensions():
    """
    Removes all the file extensions from allowables which don't actually exist for this project.
    :return: Nothing
    """
    for i in extension_ignores:
        if i in allowable_file_types:
            allowable_file_types.remove(i)


def trim_directory():
    """
    Searches from the working directory down recursively, adding any directories it finds which don't have any
    files with .cpp, .h, .cu, etc extensions in them to the ignore list.
    Also, if it never finds any .cc or .cu (or .c, .cpp, etc) files, it adds those extensions to the ignore list.
    :return: Nothing
    """
    # Set up a dictionary of extensions to keep track of whether or not you have seen each type
    exts = {}
    for t in allowable_file_types:
        exts[t] = False

    dir_tree = os.walk('.')
    for root, dirnames, filenames in dir_tree:
        d = os.path.basename(root)
        if d in ignores:
            continue
        else:
            ignore_dir = True
            for extension in allowable_file_types:
                if fnmatch.filter(filenames, '*' + extension):
                    exts[extension] = True
                    ignore_dir = False
            if ignore_dir:
                ignores.append(d)

    for extension in exts.iterkeys():
        if not exts[extension]:
            extension_ignores.append(extension)


def main(center_file_names, dot_file_name, old_style_sub_discovery=False):
    files_to_map = []
    for center_file_name in center_file_names:
        try:
            center_file = find_file(center_file_name, 'rb')

            # Crawl through the web and find all the includes and inheritance
            center_as_lines = combine_cpp_and_h_files(center_file_name.split(os.sep)[-1].split('.')[0])

            # files_to_check is a list of lists, each of the form [file, included by file, also included by file, etc.]
            map_subsystems_too = not old_style_sub_discovery
            files_to_check = crawl_web(center_as_lines, center_file.name, map_subsystems_too)

            files_to_map += files_to_check

            center_file.close()

        except IOError as ex:
            print "No file " + center_file_name

    # Walk through each list of includings and decide what relationship exists amongst them
    map_inheritance_and_composition(files_to_map, old_style_sub_discovery)

    # Print the dot files for the overview
    dot_file = open(dot_file_name + ".dot", 'wb')
    overview_file = open(dot_file_name + "_sys_overview.dot", 'wb')
    block_file = open(dot_file_name + "_block_overview.dot", 'wb')

    print_file(dot_file, overview_file, block_file, old_style_sub_discovery)

    dot_file.close()
    overview_file.close()
    block_file.close()

    # Print the subsystems
    print_all_subsystems()


#############MAIN SCRIPT#############


# Get the input arg as the "center" file
if len(sys.argv) is not 3 and len(sys.argv) is not 4:
    help_str = "USAGE: " + sys.argv[0] + "<FILE_NAME> <OUTPUT_NAME_WO_EXTENSION> <ARGS>"
    help_str += ", where <ARGS> is currently just -sub for the old-style sub-system generation. Use -sub if you want"
    help_str += " to have this script attempt to generate subsystems for you, otherwise it will do so based on"
    help_str += " directory structures."
    help_str += os.linesep + "----------------" + os.linesep
    help_str += "EXAMPLE USAGE: python " + sys.argv[0] + " ./Core/BGDriver.cpp output" + os.linesep
    help_str += "In this example, the script would look for BGDriver.cpp in ./Core/ then it would glean its includes,"
    help_str += " and look for them from the directory that the script is currently in on down."
    print help_str
    exit(0)
else:
    center_file_name = sys.argv[1]
    dot_file_name = sys.argv[2]
    old_style_subsystem_discovery = (len(sys.argv) is 4 and sys.argv[3].lower() == "-sub".lower())


# Before we do anything else, let's trim the directory to only those folders that actually have useful files
print "Optimizing directory search..."
trim_directory()

# Remove those file extensions from allowables that don't exist
remove_extensions()

# Now hash all of the files so that finding them is super fast later
print "Hashing files...."
hash_all_files()

# Find the given "center" file
try:
    extension = '.' + center_file_name.split('.')[-1]
    if extension not in allowable_file_types:
        print "This will only work in this directory on files of type:"
        for ext in allowable_file_types:
            print " " + ext
        print ".....Please give me one of those."
        exit(-1)
    else:
        center_file = find_file(center_file_name, 'rb')
except IOError as ex:
    print "File not found or some other IO error."
    exit(-1)

main([center_file_name], dot_file_name, old_style_subsystem_discovery)
