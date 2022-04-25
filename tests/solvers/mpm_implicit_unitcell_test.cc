#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "mpm_implicit.h"
#include "write_mesh_particles_unitcell.h"

// Check MPM Implicit
TEST_CASE("MPM 2D Implicit implementation is checked in unitcells",
          "[MPM][2D][Implicit][1Phase][unitcell]") {
  // Dimension
  const unsigned Dim = 2;

  // Write JSON file
  const std::string fname = "mpm-implicit";
  const std::string analysis = "MPMImplicit2D";
  const std::string mpm_scheme = "newmark";
  const std::string lin_solver_type = "IterativeEigen";
  bool nonlinear = true;
  bool quasi_static = false;
  REQUIRE(mpm_test::write_json_unitcell_implicit(2, analysis, mpm_scheme,
                                                 nonlinear, quasi_static, fname,
                                                 lin_solver_type) == true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_2d_unitcell() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_2d_unitcell() == true);

  // Assign argc and argv to input arguments of MPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
                  (char*)"-f",  (char*)"./",
                  (char*)"-i",  (char*)"mpm-implicit-2d-unitcell.json"};
  // clang-format on

  SECTION("Check initialisation") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run Implicit MPM
    auto mpm = std::make_unique<mpm::MPMImplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());

    // Initialise mesh and particles
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());

    // Initialise external loading
    REQUIRE_NOTHROW(mpm->initialise_loads());

    // Renitialise materials
    REQUIRE_THROWS(mpm->initialise_materials());
  }

  SECTION("Check solver") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run Implicit MPM
    auto mpm = std::make_unique<mpm::MPMImplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
  }
}

// Check MPM Implicit
TEST_CASE("MPM 3D Implicit implementation is checked in unitcells",
          "[MPM][3D][Implicit][1Phase][unitcell]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "mpm-implicit";
  const std::string analysis = "MPMImplicit3D";
  const std::string mpm_scheme = "newmark";
  const std::string lin_solver_type = "IterativeEigen";
  bool nonlinear = true;
  bool quasi_static = false;
  REQUIRE(mpm_test::write_json_unitcell_implicit(3, analysis, mpm_scheme,
                                                 nonlinear, quasi_static, fname,
                                                 lin_solver_type) == true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_3d_unitcell() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_3d_unitcell() == true);

  // Assign argc and argv to input arguments of MPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
                  (char*)"-f",  (char*)"./",
                  (char*)"-i",  (char*)"mpm-implicit-3d-unitcell.json"};
  // clang-format on

  SECTION("Check initialisation") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run Implicit MPM
    auto mpm = std::make_unique<mpm::MPMImplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());

    // Initialise mesh and particles
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());

    // Renitialise materials
    REQUIRE_THROWS(mpm->initialise_materials());
  }

  SECTION("Check solver") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run Implicit MPM
    auto mpm = std::make_unique<mpm::MPMImplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
  }
}
