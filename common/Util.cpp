
#include <sstream>
#include "BGTypes.h"
#include "Util.h"

/**
 * Helper function that helps with parsing integers in a fixed layout
 */
void getValueList(const char *val_string, vector<uint32_t> *value_list) {
    std::istringstream val_stream(val_string);
    uint32_t i;

    // Parse integers out of the string and add them to a list
    while (val_stream.good()) {
        val_stream >> i;
        value_list->push_back(i);
    }
}
