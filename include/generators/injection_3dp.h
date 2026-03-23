#ifndef MPM_INJECTION_3DP_H_
#define MPM_INJECTION_3DP_H_
namespace mpm {
template <unsigned Tdim>
struct Injection3DP {
  // Particle type
  std::string particle_type;
  // Material id
  std::vector<unsigned> material_ids;
  // Start
  double start_time{0.};
  // End
  double end_time{std::numeric_limits<double>::max()};
  // Injection interval
  double injection_interval{std::numeric_limits<double>::max()};
  // Last injection time
  double last_injection_time{std::numeric_limits<double>::max()};
  // Cell height for calculating particle position
  double cell_height{std::numeric_limits<double>::max()};
  // IO type for reading coordinates
  std::string io_type;
  // Check for duplicates
  bool check_duplicates{false};
  //! Number of particles to copy at each injection
  unsigned n_copies;
  //! Extrusion velocity
  double extrusion_velocity{0.0};
  //! Number of particles per cell in height direction
  unsigned particles_per_cell{1};
};
}  // namespace mpm

#endif  // MPM_INJECTION_3DP_H_
