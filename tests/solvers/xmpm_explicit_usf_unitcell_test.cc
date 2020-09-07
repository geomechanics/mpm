#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "write_mesh_particles_unitcell.h"
#include "xmpm_explicit.h"

// Check XMPM Explicit USF
TEST_CASE("XMPM 2D Explicit USF implementation is checked in unitcells",
          "[XMPM][2D][USF][Explicit][1Phase][unitcell]") {
  // Dimension
  const unsigned Dim = 2;

  // Write JSON file
  const std::string fname = "Xmpm-explicit-usf";
  // need to be done
  const std::string analysis = "XMPMExplicit3D";
  const std::string mpm_scheme = "usf";
  REQUIRE(mpm_test::write_json_unitcell(2, analysis, mpm_scheme, fname) ==
          true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_2d_unitcell() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_2d_unitcell() == true);

  // Assign argc and argv to input arguments of XMPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
                  (char*)"-f",  (char*)"./",
                  (char*)"-i",  (char*)"xmpm-explicit-usf-2d-unitcell.json"};
  // clang-format on

  SECTION("Check initialisation") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));

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
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
  }
}

// Check XMPM Explicit
TEST_CASE("XMPM 3D Explicit USF implementation is checked in unitcells",
          "[XMPM][3D][Explicit][USF][1Phase][unitcell]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "xmpm-explicit-usf";
  const std::string analysis = "XMPMExplicit3D";
  const std::string mpm_scheme = "usf";
  REQUIRE(mpm_test::write_json_unitcell(3, analysis, mpm_scheme, fname) ==
          true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_3d_unitcell() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_3d_unitcell() == true);

  // Assign argc and argv to input arguments of XMPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
                  (char*)"-f",  (char*)"./",
                  (char*)"-i",  (char*)"xmpm-explicit-usf-3d-unitcell.json"};
  // clang-format on

  SECTION("Check initialisation") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));

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
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
  }
}
