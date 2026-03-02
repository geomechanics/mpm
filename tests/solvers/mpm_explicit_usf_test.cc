#include "catch.hpp"

#include "particle.h"  

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "mpm_explicit.h"
#include "write_mesh_particles.h"

// Check MPM Explicit
TEST_CASE("MPM 2D Explicit implementation is checked",
          "[MPM][2D][Explicit][USF][1Phase]") {
  // Dimension
  const unsigned Dim = 2;

  // Write JSON file
  const std::string fname = "mpm-explicit-usf";
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
                  (char*)"-i",  (char*)"mpm-explicit-usf-2d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    const bool resume = false;
    REQUIRE(mpm_test::write_json(2, resume, analysis, mpm_scheme, fname) ==
            true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());
    // Initialise mesh
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    // Initialise particles
    REQUIRE_NOTHROW(mpm->initialise_particles());

    // Initialise external loading
    REQUIRE_NOTHROW(mpm->initialise_loads());

    // Renitialise materials
    REQUIRE_THROWS(mpm->initialise_materials());
  }

  SECTION("Check solver") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == false);
  }

  SECTION("Check resume") {
    // Write JSON file
    const std::string fname = "mpm-explicit-usf";
    const std::string analysis = "MPMExplicit2D";
    const std::string mpm_scheme = "usf";
    REQUIRE(mpm_test::write_json(2, true, analysis, mpm_scheme, fname) == true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());
    // Initialise mesh
    REQUIRE_NOTHROW(mpm->initialise_mesh());

    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == true);
    {
      // Solve
      auto io = std::make_unique<mpm::IO>(argc, argv);
      // Run explicit MPM
      auto mpm_resume = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));
      REQUIRE(mpm_resume->solve() == true);
    }
  }

  SECTION("Check rotation force physics execution") {
    mpm::Index id = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords.setZero();
    auto p = std::make_shared<mpm::Particle<Dim>>(id, coords);


    Eigen::Matrix<double, Dim, 1> p_vel;
    p_vel.fill(1.0);
    p->assign_velocity(p_vel);


    REQUIRE_NOTHROW(p->map_rotation_force(Eigen::Matrix<double, Dim, 1>::Zero(),
                                          10.0, false));
    REQUIRE_NOTHROW(p->map_rotation_force(Eigen::Matrix<double, Dim, 1>::Zero(),
                                          10.0, true));
  }

  SECTION("Check rotation JSON parsing error handling") {
    std::string err_fname = "mpm-bad-rotation";

    REQUIRE(mpm_test::write_json(2, false, analysis, mpm_scheme, err_fname) ==
            true);

    std::ifstream file_in("mpm-bad-rotation-2d.json");
    nlohmann::json j;
    file_in >> j;
    file_in.close();

    j["external_loading_conditions"]["rotation_forces"]["origin"] = {0.0, 0.0,
                                                                     0.0};

    std::ofstream file_out("mpm-bad-rotation-2d.json");
    file_out << j.dump(2);
    file_out.close();

    int argc_err = 5;
    char* argv_err[] = {(char*)"./mpm", (char*)"-f", (char*)"./", (char*)"-i",
                        (char*)"mpm-bad-rotation-2d.json"};

    auto io_err = std::make_unique<mpm::IO>(argc_err, argv_err);
    auto mpm_err = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io_err));

    REQUIRE_THROWS_AS(mpm_err->initialise_loads(), std::runtime_error);
  }

  SECTION("Check pressure smoothing") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));
    // Pressure smoothing
    REQUIRE_NOTHROW(mpm->pressure_smoothing(0));
  }
}

// Check MPM Explicit
TEST_CASE("MPM 3D Explicit implementation is checked",
          "[MPM][3D][Explicit][USF][1Phase]") {
  // Dimension
  const unsigned Dim = 3;

  // Write JSON file
  const std::string fname = "mpm-explicit-usf";
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
                  (char*)"-i",  (char*)"mpm-explicit-usf-3d.json"};
  // clang-format on

  SECTION("Check initialisation") {
    const bool resume = false;
    REQUIRE(mpm_test::write_json(3, resume, analysis, mpm_scheme, fname) ==
            true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());
    // Initialise mesh
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    // Initialise particles
    REQUIRE_NOTHROW(mpm->initialise_particles());

    // Renitialise materials
    REQUIRE_THROWS(mpm->initialise_materials());
  }

  SECTION("Check solver") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));
    // Solve
    REQUIRE(mpm->solve() == true);
    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == false);
  }

  SECTION("Check resume") {
    // Write JSON file
    const std::string fname = "mpm-explicit-usf";
    const std::string analysis = "MPMExplicit3D";
    const std::string mpm_scheme = "usf";
    const bool resume = true;
    REQUIRE(mpm_test::write_json(3, resume, analysis, mpm_scheme, fname) ==
            true);

    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));

    // Initialise materials
    REQUIRE_NOTHROW(mpm->initialise_materials());
    // Initialise mesh
    REQUIRE_NOTHROW(mpm->initialise_mesh());
    // Test check point restart
    REQUIRE(mpm->checkpoint_resume() == true);
    {
      // Solve
      auto io = std::make_unique<mpm::IO>(argc, argv);
      // Run explicit MPM
      auto mpm_resume = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));
      REQUIRE(mpm_resume->solve() == true);
    }
  }

  SECTION("Check rotation JSON parsing error handling in 3D") {
    std::string err_fname = "mpm-bad-rotation";

    REQUIRE(mpm_test::write_json(3, false, analysis, mpm_scheme, err_fname) ==
            true);

    std::ifstream file_in("mpm-bad-rotation-3d.json");
    nlohmann::json j;
    file_in >> j;
    file_in.close();

    j["external_loading_conditions"]["rotation_forces"]["origin"] = {0.0, 0.0};

    std::ofstream file_out("mpm-bad-rotation-3d.json");
    file_out << j.dump(2);
    file_out.close();

    int argc_err = 5;
    char* argv_err[] = {(char*)"./mpm", (char*)"-f", (char*)"./", (char*)"-i",
                        (char*)"mpm-bad-rotation-3d.json"};

    auto io_err = std::make_unique<mpm::IO>(argc_err, argv_err);
    auto mpm_err = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io_err));

    REQUIRE_THROWS_AS(mpm_err->initialise_loads(), std::runtime_error);
  }

  SECTION("Check pressure smoothing") {
    // Create an IO object
    auto io = std::make_unique<mpm::IO>(argc, argv);
    // Run explicit MPM
    auto mpm = std::make_unique<mpm::MPMExplicit<Dim>>(std::move(io));
    // Pressure smoothing
    REQUIRE_NOTHROW(mpm->pressure_smoothing(0));
  }
}

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
