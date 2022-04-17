#include <limits>

#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "cell.h"
#include "hencky_hyper_elastic.h"
#include "material.h"
#include "node.h"
#include "particle_finite_strain.h"

//! Check HenckyHyperElastic class in 2D
//! Cohesion only, without softening
TEST_CASE("HenckyHyperElastic is checked in 2D",
          "[material][hencky_hyper_elastic][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  // Add particle
  mpm::Index pid = 0;
  Eigen::Matrix<double, Dim, 1> coords;
  coords.setZero();
  auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;

  //! Check for id = 0
  SECTION("HenckyHyperElastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  //! Check for positive id
  SECTION("HenckyHyperElastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("HenckyHyperElastic failed initialisation") {
    unsigned id = 0;
    auto logger =
        std::make_unique<spdlog::logger>("Hencky_Test", mpm::stdout_sink);
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    try {
      auto material =
          Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
              ->create("HenckyHyperElastic2D", std::move(id), jmaterial);
    } catch (std::exception& except) {
      logger->error("HenckyHyperElastic initialization failed: {}",
                    except.what());
    }
  }

  //! Check material properties
  SECTION("HenckyHyperElastic check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.empty() == true);
      const std::vector<std::string> state_vars = {};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("HenckyHyperElastic check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    REQUIRE(stress(0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.).epsilon(Tolerance));
    mpm::Material<Dim>::Vector6d updated_stress;
    updated_stress.setZero();

    // Initialise deformation gradient
    Eigen::Matrix<double, 3, 3> deformation_gradient;
    deformation_gradient.setIdentity();

    // Initialise deformation gradient increment
    Eigen::Matrix<double, 3, 3> deformation_gradient_increment;
    deformation_gradient_increment.setIdentity();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    updated_stress = material->compute_stress(
        updated_stress, deformation_gradient, deformation_gradient_increment,
        particle.get(), &state_vars);

    // Check stressees
    REQUIRE(updated_stress(0) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Set deformation gradient increment
    deformation_gradient_increment(0, 0) = 1.2;
    deformation_gradient_increment(1, 1) = 1.5;

    // Reset stress
    updated_stress.setZero();

    // Compute updated stress
    updated_stress = material->compute_stress(
        updated_stress, deformation_gradient, deformation_gradient_increment,
        particle.get(), &state_vars);

    // Check stressees
    REQUIRE(updated_stress(0) == Approx(2663083.5703869392).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(3616688.4905331349).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(1883931.6182760221).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Set deformation gradient increment
    deformation_gradient_increment.setIdentity();
    deformation_gradient_increment(0, 0) = 1.2;
    deformation_gradient_increment(0, 1) = 0.6;
    deformation_gradient_increment(1, 0) = 0.3;
    deformation_gradient_increment(1, 1) = 1.5;

    // Reset stress
    updated_stress.setZero();

    // Compute updated stress
    updated_stress = material->compute_stress(
        updated_stress, deformation_gradient, deformation_gradient_increment,
        particle.get(), &state_vars);

    // Check stressees
    REQUIRE(updated_stress(0) == Approx(2500707.2070595203).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(3226098.9330674084).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(1718041.8420380789).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(1692580.6940184042).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto de = material->compute_consistent_tangent_matrix(
        updated_stress, stress, deformation_gradient,
        deformation_gradient_increment, particle.get(), &state_vars);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> de_check;
    // clang-format off
      de_check <<  4873507.9588488182,     3561253.5612535612,     3561253.5612535612,                      0,                      0,                      0,
                   3561253.5612535612,     4873507.9588488182,     3561253.5612535612,                      0,                      0,                      0,
                   3561253.5612535612,     3561253.5612535612,     4873507.9588488182,                      0,                      0,                      0,
                                    0,                      0,                      0,     656127.19879762852,                      0,                      0,
                                    0,                      0,                      0,                      0,     656127.19879762852,                      0,
                                    0,                      0,                      0,                      0,                      0,     656127.19879762852;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < de.rows(); ++i)
      for (unsigned j = 0; j < de.cols(); ++j)
        REQUIRE(de(i, j) == Approx(de_check(i, j)).epsilon(Tolerance));
  }

  SECTION("HenckyHyperElastic check stresses fail") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Initialise stress and strain
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    mpm::Material<Dim>::Vector6d updated_stress;
    updated_stress.setZero();
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain.setZero();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    REQUIRE_THROWS(
        material->compute_stress(stress, dstrain, particle.get(), &state_vars));

    REQUIRE_THROWS(material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars));
  }
}

//! Check HenckyHyperElastic class in 3D
TEST_CASE("HenckyHyperElastic is checked in 3D",
          "[material][hencky_hyper_elastic][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  // Add particle
  mpm::Index pid = 0;
  Eigen::Matrix<double, Dim, 1> coords;
  coords.setZero();
  auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;

  //! Check for id = 0
  SECTION("HenckyHyperElastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("HenckyHyperElastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("HenckyHyperElastic failed initialisation") {
    unsigned id = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic3D", std::move(id), jmaterial);
  }

  //! Check material properties
  SECTION("HenckyHyperElastic check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.empty() == true);
      const std::vector<std::string> state_vars = {};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("HenckyHyperElastic check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    REQUIRE(stress(0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.).epsilon(Tolerance));
    mpm::Material<Dim>::Vector6d updated_stress;
    updated_stress.setZero();

    // Initialise deformation gradient
    Eigen::Matrix<double, 3, 3> deformation_gradient;
    deformation_gradient.setIdentity();

    // Initialise deformation gradient increment
    Eigen::Matrix<double, 3, 3> deformation_gradient_increment;
    deformation_gradient_increment.setIdentity();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    updated_stress = material->compute_stress(
        updated_stress, deformation_gradient, deformation_gradient_increment,
        particle.get(), &state_vars);

    // Check stressees
    REQUIRE(updated_stress(0) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Set deformation gradient increment
    deformation_gradient_increment(0, 0) = 1.2;
    deformation_gradient_increment(1, 1) = 1.5;
    deformation_gradient_increment(2, 2) = 0.9;

    // Reset stress
    updated_stress.setZero();

    // Compute updated stress
    updated_stress = material->compute_stress(
        updated_stress, deformation_gradient, deformation_gradient_increment,
        particle.get(), &state_vars);

    // Check stressees
    REQUIRE(updated_stress(0) == Approx(2583766.2332724314).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(3643327.2556570931).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(1217754.4932354854).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Set deformation gradient increment
    deformation_gradient_increment.setIdentity();
    deformation_gradient_increment(0, 0) = 1.2;
    deformation_gradient_increment(0, 1) = 0.6;
    deformation_gradient_increment(0, 2) = 0.8;
    deformation_gradient_increment(1, 0) = 0.3;
    deformation_gradient_increment(1, 1) = 1.5;
    deformation_gradient_increment(1, 2) = 0.5;
    deformation_gradient_increment(2, 0) = 0.2;
    deformation_gradient_increment(2, 1) = 0.7;
    deformation_gradient_increment(2, 2) = 0.9;

    // Reset stress
    updated_stress.setZero();

    // Compute updated stress
    updated_stress = material->compute_stress(
        updated_stress, deformation_gradient, deformation_gradient_increment,
        particle.get(), &state_vars);

    // Check stressees
    REQUIRE(updated_stress(0) == Approx(1789169.4090916624).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(1568230.513314954).epsilon(Tolerance));
    REQUIRE(updated_stress(2) ==
            Approx(-2731967.4967589919).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(2165104.51574643).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(3837466.7808993636).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(2951397.3803139897).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto de = material->compute_consistent_tangent_matrix(
        updated_stress, stress, deformation_gradient,
        deformation_gradient_increment, particle.get(), &state_vars);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> de_check;
    // clang-format off
      de_check <<  12831746.737720303,     5623031.9388214136,     5623031.9388214136,                      0,                      0,                      0,
                   5623031.9388214136,     12831746.737720303,     5623031.9388214136,                      0,                      0,                      0,
                   5623031.9388214136,     5623031.9388214136,     12831746.737720303,                      0,                      0,                      0,
                                    0,                      0,                      0,     3604357.3994494444,                      0,                      0,
                                    0,                      0,                      0,                      0,     3604357.3994494444,                      0,
                                    0,                      0,                      0,                      0,                      0,     3604357.3994494444;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < de.rows(); ++i)
      for (unsigned j = 0; j < de.cols(); ++j)
        REQUIRE(de(i, j) == Approx(de_check(i, j)).epsilon(Tolerance));
  }
}
