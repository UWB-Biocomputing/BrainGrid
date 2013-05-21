#include "Util.h"

#include <sstream>

/**
 *  Helper function that helps with parsing integers in a fixed layout
 *  @param  val_string  list to read from
 *  @param  values_list list to fill
 */
void getValueList(const char *val_string, vector<int> *value_list) {
    std::istringstream val_stream(val_string);
    int i;

    // Parse integers out of the string and add them to a list
    while (val_stream.good()) {
        val_stream >> i;
        value_list->push_back(i);
    }
}
