# Python script for generating dot diagram for BrainGrid
# Pass in the file you want to "center" on and the script will create
# a dot file to generate a graph of all the files that the center recursively
# includes.

# TODO : Make sure all the colors are nice and pretty

import os, sys, re, fnmatch, shutil

##########GLOBALS##########

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
ignores = ["Global.cpp", "Global.h", "old", "ParseParamError.cpp", "ParseParamError.h", "Util.cpp", "Util.h"]

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
    "coral",
    "darkgoldenrod",
    "chartreuse",
    "blueviolet",
    "aquamarine",
    "darkorange",
    "darkturquoise",
    "forestgreen",
    "mediumseagreen",
    "steelblue"
]

# A dictionary of subsystem_names to subsystem_colors; populated during the execution of the script
subsystem_color_map = {}

# A reverse dictionary for subsystem_color_map
subsystem_color_map_inverse = {}

#########FUNCTIONS#########


def print_all_subsystems():
    dir_name = "dot_subsystems"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    i = 0
    for graph_text in subgraph_texts:
        f_name = os.path.join(dir_name, "subsystem" + sub_names[i] + ".dot")
        f = open(f_name, 'wb')
        i += 1
        f.write("digraph{")
        f.write(graph_text)
        f.write("}//end digraph")
        f.close()


def find_file(name, open_mode):
    print "Locating file: " + name
    matches = []
    dir_tree = os.walk('.')
    for root, dirnames, filenames in dir_tree:
        d = os.path.basename(root)
        if d in ignores:
            continue
        else:
            for filename in fnmatch.filter(filenames, name):
                matches.append(os.path.join(root, filename))
    if len(matches) is 0:
        print "No such file. Potentially not a problem."
        raise IOError
    else:
        print "Found it."
        return open(matches[0], open_mode)


def print_opening(dot_file, sys_overview_file, block_file):
    opening_blurb = "//BrainGrid Overview" + os.linesep + "//Written in the Dot language (See Graphviz)" + os.linesep
    opening_digraph = "" + os.linesep + "digraph {" + os.linesep + os.linesep + os.linesep

    dot_file.write(opening_blurb)
    dot_file.write(opening_digraph)

    sys_overview_file.write(opening_blurb)
    sys_overview_file.write(opening_digraph)

    block_file.write(opening_blurb)
    block_file.write(opening_digraph)


def print_subgraph_layout(subgraph_inheritance, subraph_includes, behavior='d'):
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
    subraph_includes = sorted(subraph_includes)

    # Add each relationship tuple to the subgraph_totals so that we don't print them again later on.
    for s in (subgraph_inheritance + subraph_includes):
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
    for c in subraph_includes:
        if last_c is None or last_c[0] != c[0]:
            to_print += "" + os.linesep
        last_c = c
        if behavior == 'd':
            to_print += "\t\t" + c[0] + " -> " + c[1] + " [arrowhead=ediamond];" + os.linesep
        else:
            to_print += "\t\t" + c[0] + " -> " + c[1] + " [style=invis];" + os.linesep

    return to_print


def print_subgraph(dot_file, sub_name_index, sub, behavior='d'):
    """
    Writes the given subgraph to the given file.
    :param dot_file: The file to write to
    :param sub_name_index: The index into sub_names for this subgraph
    :param sub: The subgraph to print
    :param behavior: 'd' = dot_file; 'o' = sys_overview; 'b' = block_file
    :return: Nothing
    """

    to_print = "" + os.linesep + os.linesep

    if len(sub) is 1:
        to_print += "\tsubgraph " + sub_names[sub_name_index] + " {" + os.linesep
    else:
        to_print += "\tsubgraph cluster" + sub_names[sub_name_index] + " {" + os.linesep

    to_print += "\t\tnode [shape = record];" + os.linesep + os.linesep

    color = sub[0].get("color")

    # Update the dictionary
    subsystem_color_map[sub_names[sub_name_index]] = color
    subsystem_color_map_inverse[color] = sub_names[sub_name_index]

    if color is not None:
        to_print += "\t\tcolor = " + sub[0].get("color") + os.linesep

    # Alphabetize the subgraph's nodes for human-readability of the DOT file
    sub = sorted(sub)

    # Print the subgraph's nodes
    for c in sub:
        to_print += "\t\t" + c.get("name") + "[label = " + c.get("name") + ", style = filled"
        if c.get("color") is not None:
            color = c.get("color")
            to_print += ", color = " + color
        to_print += "];" + os.linesep

    # If block diagram, print the master node for the subgraph
    if behavior == 'b' and len(sub) > 1:
        to_print += "\t\t" + sub_names[sub_name_index] + "[label = " + sub_names[sub_name_index] + ", style = filled"
        if sub[0].get("color") is not None:
            to_print += ", color = " + sub[0].get("color")
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

    to_print += "\t}//end subgraph " + sub_names[sub_name_index] + os.linesep

    dot_file.write(to_print)

    if behavior == 'd':
        if len(sub) > 1:
            subgraph_texts.append(to_print)


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


def print_classes(dot_file, sys_overview_file, block_file):
    subgraphs = get_subgraphs()

    # Now actually write the information to the file
    sub_name_index = 0
    for sub in subgraphs:
        print_subgraph(dot_file, sub_name_index, sub, 'd')
        print_subgraph(sys_overview_file, sub_name_index, sub, 'o')
        print_subgraph(block_file, sub_name_index, sub, 'b')

        sub_name_index += 1
        if sub_name_index >= len(sub_names):
            sub_name_index = 0


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


def find_system_from_name(name):
    sub_systems = get_subgraphs()
    for sub in sub_systems:
        for d in sub:
            if d.get("name") == name:
                return d.get("color")
    return None


def print_block_layout(block_file):
    # Form all the inter-system relationships
    all_relationships = inheritance + includes
    inter_system_relationships = []
    for relationship in all_relationships:
        tup0_system = find_system_from_name(relationship[0])
        tup1_system = find_system_from_name(relationship[1])
        if tup0_system != tup1_system:
            inter_system_relationships.append(relationship)

    temp = []
    # For each inter-system relationship,
    for relationship in inter_system_relationships:
        inter_system_relationships.remove(relationship)

        # Replace the first item in that relationship with the master node for the subsystem it belongs to
        r0 = relationship[0]
        color = find_system_from_name(r0)
        if color is None:
            node_name0 = r0
        else:
            node_name0 = subsystem_color_map_inverse.get(color)

        # Replace the second item in that relationship with the master node for the subsystem it belongs to
        r1 = relationship[1]
        color = find_system_from_name(r1)
        if color is None:
            node_name1 = r1
        else:
            node_name1 = subsystem_color_map_inverse.get(color)

        inter_relationship = (node_name0, node_name1)
        temp.append(inter_relationship)

    # Remove any duplicate tuples
    inter_system_relationships = list(set(temp))

    # Now Formulate the string for each subgraph and print it to the file
    print_layout_boilerplate(block_file, 'b')

    for r in inter_system_relationships:
        line = r[0] + " -> " + r[1] + " [arrowhead=ediamond];" + os.linesep
        block_file.write(line)


def print_layout(dot_file, sys_overview_file, block_file):
    print_layout_boilerplate(dot_file, 'd')
    print_layout_boilerplate(sys_overview_file, 'o')

    for c in inheritance:
        if c not in subgraph_totals:
            line = c[0] + " -> " + c[1] + " [arrowhead=empty];" + os.linesep
            dot_file.write(line)
            sys_overview_file.write(line)

    for c in includes:
        if c not in subgraph_totals:
            line = c[0] + " -> " + c[1] + " [arrowhead=ediamond];" + os.linesep
            dot_file.write(line)
            sys_overview_file.write(line)

    print_block_layout(block_file)


def print_end(dot_file, sys_overview_file, block_file):
    line = "}//End digraph declaration" + os.linesep
    dot_file.write(line)
    sys_overview_file.write(line)
    block_file.write(line)


def print_file(dot_file, sys_overview_file, block_file):
    print "Generating dot files..."
    print_opening(dot_file, sys_overview_file, block_file)
    print_classes(dot_file, sys_overview_file, block_file)
    print_layout(dot_file, sys_overview_file, block_file)
    print_end(dot_file, sys_overview_file, block_file)


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
            if name_split[1] not in ignores:
                name = name_split[1].split('.')[0]  # BookStore
                found_list.append(name)
    return found_list


def combine_cpp_and_h_files(file_name):
    """
    Finds the .cpp/.cu and .h files that correspond to the given name (name
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

    try:
        cu = find_file(file_name + ".cu", 'rb')
        cu_lines = [line for line in cu]
        cu.close()

        for line in cu_lines:
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
    [   [main, BookStore] , [BookStore, HashTable, Book, Customer, Transaction], [HashTable, etc.], etc.  ] <- where main includes BookStore
    and BookStore includes HashTable, Book, Customer, and Transaction, etc.
    """

    center_file_name = center_file_name.split("\\")[-1].split('.')[0]  # Strip the extension from the name: BGDriver.cpp -> BGDriver

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
                # create e's list
                e_as_lines = combine_cpp_and_h_files(e)
                e_list = find_includes(e_as_lines)

                # put e at front of e's list
                if e in e_list:
                    e_list.remove(e)
                e_list.insert(0, e)

                # stack.push(e's list)
                stack.append(e_list)

    # return to_ret
    return to_ret


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


def color_subsystems(subsystems):
    """
    Colors the subsystems that are bigger than a single item by modifying the "classes" global.
    :param subsystems: A list of all the files lumped into lists ('subsystems')
    :return: Nothing
    """
    sys_index = 0
    for sys in subsystems:
        if len(sys) > 1:
            list_of_each_dictionary = [item for item in classes if item.get("name") in sys] # Each item in the subsystem as a dictionary
            # update each dictionary with the new color
            for dic in list_of_each_dictionary:
                dic["color"] = SUBSYSTEM_COLORS[sys_index]
        sys_index += 1
        if sys_index >= len(SUBSYSTEM_COLORS):
            sys_index = 0


def create_subgraphs(subsystems):
    """
    Changes the layout of the graph to lump all the subsystems together by modifying the "classes" global.
    :param subsystems:
    :return:
    """
    j = 0
    for sub in subsystems:
        list_of_each_dictionary = [item for item in classes if item.get("name" in sub)]
        for dic in list_of_each_dictionary:
            dic["subgraph"] = j
        j += 1


def map_subsystems():
    """
    Walks through the three global lists (inheritance, includes, and classes) and determines what subsystem each
    item belongs to. Adds that information to the "classes" list.
    """
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

    color_subsystems(subsystems)
    create_subgraphs(subsystems)


def map_inheritance_and_composition(list_of_include_groups):
    """
    This function maps the relationships between the files which are related and fills the global
    "includes" and "inheritance" lists with tuples of the form: (includer, included).
    This function also populates the "classes" list.

    :param list_of_include_groups A list of lists, each of the form [file_name_A, file_name_B, file_name_C, etc.] where
    file_name_B and file_name_C, etc. are all included BY file A.
    """
    print "Mapping relationships and identifying subsystems..."
    for include_group in list_of_include_groups:
        # For each include_group, determine the type of relationship it is
        if len(include_group) > 1:
            parent_name = include_group[0]
            if {"name": parent_name} not in classes:
                classes.append({"name": parent_name})
            index = 0
            rest_of_layer = include_group[1:]
            print "Mapping relationships for " + parent_name
            for item in rest_of_layer:
                if {"name": item} not in classes:
                    classes.append({"name": item})
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
    map_subsystems()


def main(center_file_names, dot_file_name, print_sub_systems=False):
    files_to_map = []
    for center_file_name in center_file_names:
        try:
            center_file = find_file(center_file_name, 'rb')

            # Crawl through the web and find all the includes and inheritance
            center_as_lines = combine_cpp_and_h_files(center_file_name.split("\\")[-1].split('.')[0])

            # files_to_check is a list of lists, each of the form [file, included by file, also included by file, etc.]
            files_to_check = crawl_web(center_as_lines, center_file.name)

            files_to_map += files_to_check

            center_file.close()

        except IOError as ex:
            print "No file " + center_file_name

    # Walk through each list of includings and decide what relationship exists amongst them
    map_inheritance_and_composition(files_to_map)

    # Print the dot files for the overview
    dot_file = open(dot_file_name + ".dot", 'wb')
    overview_file = open(dot_file_name + "_sys_overview.dot", 'wb')
    block_file = open(dot_file_name + "_block_overview.dot", 'wb')
    print_file(dot_file, overview_file, block_file)
    dot_file.close()

    if print_sub_systems:
        print_all_subsystems()


#############MAIN SCRIPT#############

# Get the input arg as the "center" file
if len(sys.argv) is not 3:
    print "USAGE: dotGenerator.py <FILE_NAME> <OUTPUT_NAME_WO_EXTENSION>"
    exit(0)
else:
    center_file_name = sys.argv[1]
    dot_file_name = sys.argv[2]

#Find the given "center" file
try:
    center_file = find_file(center_file_name, 'rb')
    extension = center_file.name.split('.')[-1]
    if extension != "cpp" and extension != "h" and extension != "cu":
        print "This only works for .cpp/.cu and .h files. Please pass me one of those."
        exit(-1)
except IOError as ex:
    print "File not found or some other IO error."
    exit(-1)

main([center_file_name], dot_file_name, True)
