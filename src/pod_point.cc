#include "pod_point.h"
namespace mpm {
namespace pod {
namespace point {
const size_t dst_offset[NFIELDS] = {
    HOFFSET(PODPoint, id),
    HOFFSET(PODPoint, area),
    HOFFSET(PODPoint, coord_x),
    HOFFSET(PODPoint, coord_y),
    HOFFSET(PODPoint, coord_z),
    HOFFSET(PODPoint, displacement_x),
    HOFFSET(PODPoint, displacement_y),
    HOFFSET(PODPoint, displacement_z),
    HOFFSET(PODPoint, status),
    HOFFSET(PODPoint, cell_id),
};

// Get size of point
PODPoint point;
const size_t dst_sizes[NFIELDS] = {
    sizeof(point.id),
    sizeof(point.area),
    sizeof(point.coord_x),
    sizeof(point.coord_y),
    sizeof(point.coord_z),
    sizeof(point.displacement_x),
    sizeof(point.displacement_y),
    sizeof(point.displacement_z),
    sizeof(point.status),
    sizeof(point.cell_id),
};

// Define point field information
const char* field_names[NFIELDS] = {
    "id",      "area",           "coord_x",        "coord_y",
    "coord_z", "displacement_x", "displacement_y", "displacement_z",
    "status",  "cell_id",
};

// Initialize field types
const hid_t field_type[NFIELDS] = {
    H5T_NATIVE_LLONG,  H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE,
    H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE,
    H5T_NATIVE_HBOOL,  H5T_NATIVE_LLONG};
}  // namespace point
}  // namespace pod
}  // namespace mpm
