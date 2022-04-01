#ifndef MPM_POD_FINITE_STRAIN_H_
#define MPM_POD_FINITE_STRAIN_H_

// POD Particle
#include "pod_particle.h"

namespace mpm {
// Define a struct of particle
typedef struct PODParticleFiniteStrain : PODParticle {
  // Deformation gradient
  double defgrad_11, defgrad_12, defgrad_13;
  double defgrad_21, defgrad_22, defgrad_23;
  double defgrad_31, defgrad_32, defgrad_33;
  // Destructor
  virtual ~PODParticleFiniteStrain() = default;
} PODParticleFiniteStrain;

namespace pod {
namespace particlefinitestrain {
const hsize_t NFIELDS = 65;

const size_t dst_size = sizeof(PODParticleFiniteStrain);

// Destination offset
extern const size_t dst_offset[NFIELDS];

// Destination size
extern const size_t dst_sizes[NFIELDS];

// Define particle field information
extern const char* field_names[NFIELDS];

// Initialize field types
extern const hid_t field_type[NFIELDS];

}  // namespace particlefinitestrain
}  // namespace pod

}  // namespace mpm

#endif  // MPM_POD_FINITE_STRAIN_H_