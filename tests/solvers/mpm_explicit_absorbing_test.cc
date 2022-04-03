#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "mpm_explicit.h"
#include "write_mesh_particles.h"

// Check MPM Explicit Absorbing Boundary
TEST_CASE("MPM 2D Explicit Absorbing Boundary is checked",
          "[MPM][2D][Explicit][Absorbing]") {
  // Dimension
  const unsigned Dim = 2;

  // Write JSON file
  const std::string fname = "mpm-explicit-absorbing";
  const std::string analysis = "MPMExplicit2D";
  const std::string mpm_scheme = "usf";

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
                  (char*)"-i",  (char*)"mpm-explicit-absorbing-2d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    const bool resume = false;
    REQUIRE(mpm_test::write_json_absorbing(2, resume, analysis, mpm_scheme,
                                           fname, "corner") == true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());

    // Initialise mesh and particles
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());
  }
}

// Check MPM Explicit USL
TEST_CASE("MPM 3D Explicit Absorbing Boundary is checked",
          "[MPM][3D][Explicit][Absorbing]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "mpm-explicit-absorbing";
  const std::string analysis = "MPMExplicit3D";
  const std::string mpm_scheme = "usf";

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
                  (char*)"-i",  (char*)"mpm-explicit-absorbing-3d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    const bool resume = false;
    REQUIRE(mpm_test::write_json_absorbing(3, resume, analysis, mpm_scheme,
                                           fname, "face") == true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());

    // Initialise mesh and particles
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());
  }
}