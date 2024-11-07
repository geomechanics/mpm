#include <fstream>

#include "mpm.h"

namespace mpm_test {

// Write JSON Configuration file
bool write_json(unsigned dim, bool resume, const std::string& analysis,
                const std::string& mpm_scheme, const std::string& file_name);

// Write JSON Configuration file for mpm_base warnings
bool write_json_warnings(unsigned dim, bool material_sets, bool math_functions,
                         std::string math_functions_type, int grav_dim,
                         std::string file_name);

// Write JSON Configuration file for absorbing boundary
bool write_json_absorbing(unsigned dim, bool resume,
                          const std::string& analysis,
                          const std::string& file_name,
                          const std::string& position,
                          const double delta = 100.0);

// Write JSON Configuration file for absorbing boundary
bool write_json_acceleration(unsigned dim, bool resume,
                             const std::string& analysis,
                             const std::string& file_name, const unsigned dir);

// Write JSON Configuration file for friction boundary
bool write_json_friction(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& file_name, const unsigned dir);

// Write JSON Configuration file for adhesion boundary
bool write_json_adhesion(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& file_name, const unsigned dir);

// Write JSON Configuration file for velocity boundary
bool write_json_velocity(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& file_name, const unsigned dir);

// Write JSON Configuration file for finite strain
bool write_json_finite_strain(unsigned dim, bool resume,
                              const std::string& analysis,
                              const std::string& mpm_scheme,
                              const std::string& file_name);

// Write JSON Configuration file for implicit
bool write_json_implicit(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& mpm_scheme, bool nonlinear,
                         bool quasi_static, const std::string& file_name,
                         const std::string& linear_solver_type = "none");

// Write JSON Configuration file for implicit finite strain
bool write_json_implicit_finite_strain(
    unsigned dim, bool resume, const std::string& analysis,
    const std::string& mpm_scheme, bool nonlinear, bool quasi_static,
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
                         const std::string& linear_solver_type = "none",
                         const std::string& vel_update = "flip");

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

// Write math function csv
bool write_math_function();

}  // namespace mpm_test
