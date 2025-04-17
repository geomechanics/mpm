#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "contact_levelset.h"
#include "mpm_explicit.h"
#include "particle_levelset.h"
#include "write_mesh_particles.h"

//! \brief Check MPM Explicit Levelset Interface (2D)
TEST_CASE("MPM 2D Explicit Levelset Interface is checked",
          "[MPM][2D][Explicit][Levelset]") {
  // Dimension
  const unsigned Dim = 2;

  // Write JSON file
  const std::string fname = "mpm-levelset-test";
  const std::string analysis = "MPMExplicit2D";

  // Write Mesh with levelset boundaries
  REQUIRE(mpm_test::write_mesh_2d() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_2d() == true);

  // Assign argc and argv to input arguments of MPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
                  (char*)"-f",  (char*)"./",
                  (char*)"-i",  (char*)"mpm-levelset-test-2d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    // Write custom JSON for levelset test with interface_type: "levelset"
    REQUIRE(mpm_test::write_json_levelset(Dim, false, analysis, fname, false,
                                          false) == true);
    REQUIRE(mpm_test::write_json_levelset(Dim, false, analysis, fname, true,
                                          true) == true);

    // Create an IO object and run explicit MPM
    auto io = std::make_unique<mpm::IO>(argc, argv);
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise and check that the solver runs with levelset boundary
    REQUIRE_NOTHROW(mpm->initialise_materials());
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());
    REQUIRE_NOTHROW(mpm->initialise_loads());
  }

  SECTION("Check solver") {
    // Create an IO object and run explicit MPM
    auto io = std::make_unique<mpm::IO>(argc, argv);
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Check that the solver runs without throwing
    REQUIRE(mpm->solve() == true);
  }
}

//! \brief Check MPM Explicit Levelset Interface (3D)
TEST_CASE("MPM 3D Explicit Levelset Interface is checked",
          "[MPM][3D][Explicit][Levelset]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "mpm-levelset-test";
  const std::string analysis = "MPMExplicit3D";

  // Write Mesh with levelset boundaries
  REQUIRE(mpm_test::write_mesh_3d() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_3d() == true);

  // Assign argc and argv to input arguments of MPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
            (char*)"-f",  (char*)"./",
            (char*)"-i",  (char*)"mpm-levelset-test-3d.json"};
  // clang-format on

  SECTION("Check initialisation and solver") {
    // Write custom JSON for levelset test with interface_type: "levelset"
    REQUIRE(mpm_test::write_json_levelset(Dim, false, analysis, fname, false,
                                          false) == true);
    REQUIRE(mpm_test::write_json_levelset(Dim, false, analysis, fname, true,
                                          true) == true);

    // Create an IO object and run explicit MPM
    auto io = std::make_unique<mpm::IO>(argc, argv);
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise and check that the solver runs with levelset boundary
    REQUIRE_NOTHROW(mpm->initialise_materials());
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());
    REQUIRE_NOTHROW(mpm->initialise_loads());
  }

  SECTION("Check solver") {
    // Create an IO object and run explicit MPM
    auto io = std::make_unique<mpm::IO>(argc, argv);
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Check that the solver runs without throwing
    REQUIRE(mpm->solve() == true);
  }
}
