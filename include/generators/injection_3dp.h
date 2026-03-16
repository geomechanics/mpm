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
  // Particle volume
  double particle_volume{std::numeric_limits<double>::max()};
  // Cell height for calculating particle position
  double h{std::numeric_limits<double>::max()};
  // Particle velocity
  std::vector<double> velocity_p;
  // coordinate file for initial particle coordinates
  std::string coordinate_file;
  // Initial coordinates for particle injection
  std::vector<Eigen::Matrix<double, Tdim, 1>> initial_coordinates;
  // IO type for reading coordinates
  std::string io_type;
  // Check for duplicates
  bool check_duplicates{false};
};
}  // namespace mpm

#endif  // MPM_INJECTION_3DP_H_
