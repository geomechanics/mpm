#include <fstream>

#include "mpm.h"

namespace mpm_test {

// Write JSON Configuration file
bool write_json(unsigned dim, bool resume, const std::string& analysis,
                const std::string& mpm_scheme, const std::string& file_name);

// Write JSON Configuration file for xmpm
bool write_json_xmpm(unsigned dim, bool resume, const std::string& analysis,
                     const std::string& mpm_scheme,
                     const std::string& file_name);
// Write JSON Configuration file for implicit linear
bool write_json_implicit_linear(unsigned dim, bool resume,
                                const std::string& analysis,
                                const std::string& mpm_scheme,
                                const std::string& file_name,
                                const std::string& linear_solver_type = "none");

// Write JSON Configuration file for navierstokes
bool write_json_navierstokes(unsigned dim, bool resume,
                             const std::string& analysis,
                             const std::string& mpm_scheme,
                             const std::string& file_name,
                             const std::string& free_surface_type = "none",
                             const std::string& linear_solver_type = "none");

// Write JSON Configuration file for twophase
bool write_json_twophase(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& mpm_scheme,
                         const std::string& file_name,
                         const std::string& free_surface_type = "none",
                         const std::string& linear_solver_type = "none");
// Write JSON Entity Set
bool write_entity_set();

// Write Mesh file in 2D
bool write_mesh_2d();
// Write particles file in 2D
bool write_particles_2d();

// Write mesh file in 3D
bool write_mesh_3d();
// Write particles file in 3D
bool write_particles_3d();
// Write discontinuity file in 3D
bool write_discontinuity_3d();

}  // namespace mpm_test
