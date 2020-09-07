#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "write_mesh_particles.h"
#include "xmpm_explicit.h"

// Check XMPM Explicit
TEST_CASE("XMPM 3D Explicit implementation is checked",
          "[XMPM][3D][Explicit][USF][1Phase]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "xmpm-explicit-usf";
  const std::string analysis = "XMPMExplicit3D";
  const std::string mpm_scheme = "usf";
  const bool resume = false;
  REQUIRE(mpm_test::write_json_xmpm(3, resume, analysis, mpm_scheme, fname) ==
          true);

  // Write JSON Entity Sets file
  REQUIRE(mpm_test::write_entity_set() == true);

  // Write Mesh
  REQUIRE(mpm_test::write_mesh_3d() == true);

  // Write Particles
  REQUIRE(mpm_test::write_particles_3d() == true);

  // Write discontinuity
  REQUIRE(mpm_test::write_discontinuity_3d() == true);

  // Assign argc and argv to input arguments of XMPM
  int argc = 5;
  // clang-format off
  char* argv[] = {(char*)"./mpm",
                  (char*)"-f",  (char*)"./",
                  (char*)"-i",  (char*)"xmpm-explicit-usf-3d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());
    // Initialise mesh
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    // Initialise particles
    REQUIRE_NOTHROW(mpm->initialise_particles());
    // Initialise discontinuities
    REQUIRE_NOTHROW(mpm->initialise_discontinuities());

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
    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == false);
  }

  SECTION("Check resume") {
    // Write JSON file
    const std::string fname = "xmpm-explicit-usf";
    const std::string analysis = "XMPMExplicit3D";
    const std::string mpm_scheme = "usf";
    bool resume = true;
    REQUIRE(mpm_test::write_json_xmpm(3, resume, analysis, mpm_scheme, fname) ==
            true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));

    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == true);
    // Solve
    REQUIRE(mpm->solve() == true);
  }
}
