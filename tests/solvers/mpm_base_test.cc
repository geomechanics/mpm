#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "mpm_explicit.h"
#include "write_mesh_particles.h"

// Check MPM Base warnings/errors
TEST_CASE("MPM base warnings are checked (2D Explicit)", "[MPM][2D][Base]") {
  // Dimension
  const unsigned Dim = 2;

  // Write JSON Entity Sets file
  REQUIRE(mpm_test::write_entity_set() == true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_2d() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_2d() == true);

  // Assign argc and argv to input arguments of MPM
  int argc = 5;
  // clang-format off
    char* argv[] = {(char*)"./mpm",
                    (char*)"-f",  (char*)"./",
                    (char*)"-i",  (char*)"mpm-base-warnings-2d.json"};
  // clang-format on

  // Write bad JSON file
  std::string fname = "mpm-base-warnings";
  REQUIRE(mpm_test::write_json_warnings(2, false, false, "Linear", 2, fname) ==
          true);

  // Create an IO object and run explicit MPM
  auto io = std::make_unique<mpm::IO>(argc, argv);
  auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

  // Check bad initialisation
  REQUIRE_NOTHROW(mpm->initialise_materials());
  REQUIRE_NOTHROW(mpm->initialise_mesh());
  REQUIRE_NOTHROW(mpm->initialise_particles());
  REQUIRE_NOTHROW(mpm->initialise_loads());

  // Solve with bad JSON
  REQUIRE_THROWS(mpm->solve());
}

// Check MPM Base warnings/errors
TEST_CASE("MPM base warnings are checked (3D Explicit)", "[MPM][3D][Base]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON Entity Sets file
  REQUIRE(mpm_test::write_entity_set() == true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_3d() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_3d() == true);

  // Assign argc and argv to input arguments of MPM
  int argc = 5;
  // clang-format off
    char* argv[] = {(char*)"./mpm",
                    (char*)"-f",  (char*)"./",
                    (char*)"-i",  (char*)"mpm-base-warnings-3d.json"};
  // clang-format on

  // Write bad JSON file
  std::string fname = "mpm-base-warnings";
  REQUIRE(mpm_test::write_json_warnings(3, true, true, "Lin", 2, fname) ==
          true);

  // Create an IO object and run explicit MPM
  auto io = std::make_unique<mpm::IO>(argc, argv);
  auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

  // Check bad initialisation
  REQUIRE_NOTHROW(mpm->initialise_materials());
  REQUIRE_NOTHROW(mpm->initialise_mesh());
  REQUIRE_NOTHROW(mpm->initialise_particles());
  REQUIRE_THROWS(mpm->initialise_loads());

  // Solve with bad JSON
  REQUIRE_THROWS(mpm->solve());
}