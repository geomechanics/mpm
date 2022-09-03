#include <fstream>

#include "mpm.h"

namespace mpm_test {

// Write JSON Configuration file
bool write_json_unitcell(unsigned dim, const std::string& analysis,
                         const std::string& mpm_scheme,
                         const std::string& file_name);

// Write JSON Configuration file for finite strain
bool write_json_unitcell_finite_strain(unsigned dim,
                                       const std::string& analysis,
                                       const std::string& mpm_scheme,
                                       const std::string& file_name);

// Write JSON Configuration file for implicit
bool write_json_unitcell_implicit(
    unsigned dim, const std::string& analysis, const std::string& mpm_scheme,
    bool nonlinear, bool quasi_static, const std::string& file_name,
    const std::string& linear_solver_type = "none");

// Write JSON Configuration file for implicit finite strain
bool write_json_unitcell_implicit_finite_strain(
    unsigned dim, const std::string& analysis, const std::string& mpm_scheme,
    bool nonlinear, bool quasi_static, const std::string& file_name,
    const std::string& linear_solver_type = "none");

// Write JSON Configuration file for navier stokes
bool write_json_unitcell_navierstokes(
    unsigned dim, const std::string& analysis, const std::string& mpm_scheme,
    const std::string& file_name, const std::string& free_surface_type = "none",
    const std::string& linear_solver_type = "none");

// Write JSON Configuration file for two-phase
bool write_json_unitcell_twophase(
    unsigned dim, const std::string& analysis, const std::string& mpm_scheme,
    const std::string& file_name, const std::string& free_surface_type = "none",
    const std::string& linear_solver_type = "none");

// Write Mesh file in 2D
bool write_mesh_2d_unitcell();
// Write particles file in 2D
bool write_particles_2d_unitcell();

// Write mesh file in 3D
bool write_mesh_3d_unitcell();
// Write particles file in 3D
bool write_particles_3d_unitcell();

}  // namespace mpm_test
