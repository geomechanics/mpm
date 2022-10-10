#include "catch.hpp"

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "write_mesh_particles.h"
#include "xmpm_explicit.h"

// Check MPM Explicit MUSL
TEST_CASE("XMPM 3D Explicit MUSL implementation is checked",
          "[XMPM][3D][Explicit][MUSL][1Phase]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "xmpm-explicit-musl";
  const std::string analysis = "XMPMExplicit3D";
  const std::string mpm_scheme = "musl";

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
                  (char*)"-i",  (char*)"xmpm-explicit-musl-3d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    const bool resume = false;
    REQUIRE(mpm_test::write_json_xmpm(3, resume, analysis, mpm_scheme, fname) ==
            true);
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());

    // Initialise mesh and particles
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    REQUIRE_NOTHROW(mpm->initialise_particles());

    // Initialise discontinuities
    REQUIRE_NOTHROW(mpm->initialise_discontinuity());
  }

  SECTION("Check solver") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
  }

  SECTION("Check resume") {
    // Write JSON file
    const std::string fname = "xmpm-explicit-musl";
    const std::string analysis = "XMPMExplicit3D";
    const std::string mpm_scheme = "musl";
    const bool resume = true;
    REQUIRE(mpm_test::write_json_xmpm(3, resume, analysis, mpm_scheme, fname) ==
            true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit XMPM
    auto mpm = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());
    // Initialise mesh
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == true);
    {
      // Solve
      auto io = std::make_unique<mpm::IO>(argc, argv);
      // Run explicit XMPM
      auto mpm_resume = std::make_unique<mpm::XMPMExplicit<Dim>>(std::move(io));
      REQUIRE(mpm_resume->solve() == true);
    }
  }
}