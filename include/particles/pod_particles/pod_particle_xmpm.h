#ifndef MPM_POD_XMPM_H_
#define MPM_POD_XMPM_H_

// POD Particle
#include "pod_particle.h"

namespace mpm {
// Define a struct of particle
typedef struct PODParticleXMPM : PODParticle {
  // Level set values
  double levelset_phi;

  // Destructor
  virtual ~PODParticleXMPM() = default;
} PODParticleXMPM;

namespace pod {
namespace particlexmpm {
const hsize_t NFIELDS = 65;

const size_t dst_size = sizeof(PODParticleXMPM);

// Destination offset
extern const size_t dst_offset[NFIELDS];

// Destination size
extern const size_t dst_sizes[NFIELDS];

// Define particle field information
extern const char* field_names[NFIELDS];

// Initialize field types
extern const hid_t field_type[NFIELDS];

}  // namespace particlexmpm
}  // namespace pod

}  // namespace mpm

#endif  // MPM_POD_XMPM_H_
