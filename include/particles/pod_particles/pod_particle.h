#ifndef MPM_POD_H_
#define MPM_POD_H_

// HDF5
#include "hdf5.h"
#include "hdf5_hl.h"

#include "data_types.h"

namespace mpm {
// Define a struct of particle
typedef struct PODParticle {
  // Index
  mpm::Index id;
  // Mass
  double mass;
  // Volume
  double volume;
  // Pressure
  double pressure;
  // Coordinates
  double coord_x, coord_y, coord_z;
  // Displacement
  double displacement_x, displacement_y, displacement_z;
  // Natural particle size
  double nsize_x, nsize_y, nsize_z;
  // Velocity
  double velocity_x, velocity_y, velocity_z;
  // Acceleration
  double acceleration_x, acceleration_y, acceleration_z;
  // Stresses
  double stress_xx, stress_yy, stress_zz;
  double tau_xy, tau_yz, tau_xz;
  // Strains
  double strain_xx, strain_yy, strain_zz;
  double gamma_xy, gamma_yz, gamma_xz;
  // Deformation gradient
  double defgrad_00, defgrad_01, defgrad_02;
  double defgrad_10, defgrad_11, defgrad_12;
  double defgrad_20, defgrad_21, defgrad_22;
  // Mapping matrix
  bool initialise_mapping_matrix;
  double mapping_matrix_00, mapping_matrix_01, mapping_matrix_02;
  double mapping_matrix_10, mapping_matrix_11, mapping_matrix_12;
  double mapping_matrix_20, mapping_matrix_21, mapping_matrix_22;
  // Status
  bool status;
  // Index
  mpm::Index cell_id;
  // Material id
  unsigned material_id;
  // Number of state variables
  unsigned nstate_vars;
  // State variables (init to zero)
  double svars[20] = {0};
  // Destructor
  virtual ~PODParticle() = default;
} PODParticle;

namespace pod {
namespace particle {
const hsize_t NFIELDS = 74;

const size_t dst_size = sizeof(PODParticle);

// Destination offset
extern const size_t dst_offset[NFIELDS];

// Destination size
extern const size_t dst_sizes[NFIELDS];

// Define particle field information
extern const char* field_names[NFIELDS];

// Initialize field types
extern const hid_t field_type[NFIELDS];

}  // namespace particle
}  // namespace pod

}  // namespace mpm

#endif  // MPM_POD_H_
