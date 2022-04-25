#include <limits>

#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "cell.h"
#include "material.h"
#include "node.h"
#include "particle.h"

//! Check linearelastic class in 2D
TEST_CASE("LinearElastic is checked in 2D", "[material][linear_elastic][2D]") {
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
  SECTION("LinearElastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("LinearElastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("LinearElastic failed initialisation") {
    unsigned id = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
  }

  //! Check material properties
  SECTION("LinearElastic check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));

    // Get material properties fail
    REQUIRE_THROWS(material->property<double>("shear_modulus"));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.empty() == true);
      const std::vector<std::string> state_vars = {};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("LinearElastic check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
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

    // Initialise strain
    mpm::Material<Dim>::Vector6d strain;
    strain.setZero();
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0000000;
    strain(3) = 0.0000000;
    strain(4) = 0.0000000;
    strain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.63461538461538e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.25000000000000e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.86538461538462e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Initialise strain
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0000000;
    strain(3) = 0.0000100;
    strain(4) = 0.0000000;
    strain(5) = 0.0000000;

    // Reset stress
    stress.setZero();

    // Compute updated stress
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.63461538461538e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.25000000000000e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.86538461538462e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(3.84615384615385e+01).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.00000000000000e+00).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.00000000000000e+00).epsilon(Tolerance));
  }

  SECTION("LinearElastic check properties earthquake") {
    unsigned id = 0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);

    // Get P-Wave and S-Wave Velocities
    REQUIRE(material->template property<double>("pwave_velocity") ==
            Approx(116.023870223).epsilon(Tolerance));
    REQUIRE(material->template property<double>("swave_velocity") ==
            Approx(62.0173672946).epsilon(Tolerance));
  }
}

//! Check linearelastic class in 3D
TEST_CASE("LinearElastic is checked in 3D", "[material][linear_elastic][3D]") {
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
  SECTION("LinearElastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("LinearElastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("LinearElastic failed initialisation") {
    unsigned id = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
  }

  //! Check material properties
  SECTION("LinearElastic check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
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

  SECTION("LinearElastic check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    //    mpm::Material<Dim>::Matrix6x6 de = material->elastic_tensor();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    REQUIRE(stress(0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.).epsilon(Tolerance));

    // Initialise strain
    mpm::Material<Dim>::Vector6d strain;
    strain.setZero();
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0005000;
    strain(3) = 0.0000000;
    strain(4) = 0.0000000;
    strain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.92307692307333e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Initialise strain
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0005000;
    strain(3) = 0.0000100;
    strain(4) = 0.0000200;
    strain(5) = 0.0000300;

    // Reset stress
    stress.setZero();

    // Compute updated stress
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.92307692307333e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(3.84615384615385e+01).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(7.69230769230769e+01).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(1.15384615384615e+02).epsilon(Tolerance));
  }

  SECTION("LinearElastic check stresses fail") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Initialise stress and strain
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
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
    REQUIRE_THROWS(material->compute_stress(stress, deformation_gradient,
                                            deformation_gradient_increment,
                                            particle.get(), &state_vars));

    REQUIRE_THROWS(material->compute_consistent_tangent_matrix(
        updated_stress, stress, deformation_gradient,
        deformation_gradient_increment, particle.get(), &state_vars));
  }

  SECTION("LinearElastic check properties earthquake") {
    unsigned id = 0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);

    // Get P-Wave and S-Wave Velocities
    REQUIRE(material->template property<double>("pwave_velocity") ==
            Approx(116.023870223).epsilon(Tolerance));
    REQUIRE(material->template property<double>("swave_velocity") ==
            Approx(62.0173672946).epsilon(Tolerance));
  }
}
