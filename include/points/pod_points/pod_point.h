#ifndef MPM_POD_POINT_H_
#define MPM_POD_POINT_H_

// HDF5
#include "hdf5.h"
#include "hdf5_hl.h"

#include "data_types.h"

namespace mpm {
// Define a struct of point
typedef struct PODPoint {
  // Index
  mpm::Index id;
  // Area
  double area;
  // Coordinates
  double coord_x, coord_y, coord_z;
  // Displacement
  double displacement_x, displacement_y, displacement_z;
  // Status
  bool status;
  // Index
  mpm::Index cell_id;
  // Destructor
  virtual ~PODPoint() = default;
} PODPoint;

namespace pod {
namespace point {
const hsize_t NFIELDS = 10;

const size_t dst_size = sizeof(PODPoint);

// Destination offset
extern const size_t dst_offset[NFIELDS];

// Destination size
extern const size_t dst_sizes[NFIELDS];

// Define point field information
extern const char* field_names[NFIELDS];

// Initialize field types
extern const hid_t field_type[NFIELDS];

}  // namespace point
}  // namespace pod

}  // namespace mpm

#endif  // MPM_POD_POINT_H_
