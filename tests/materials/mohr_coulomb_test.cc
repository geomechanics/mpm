#include <limits>

#include <cmath>

#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "cell.h"
#include "material.h"
#include "mohr_coulomb.h"
#include "node.h"
#include "particle.h"
#include <iostream>

//! Check MohrCoulomb class in 2D
//! Cohesion only, without softening
TEST_CASE("MohrCoulomb is checked in 2D (cohesion only, without softening)",
          "[material][mohr_coulomb][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

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
  jmaterial["softening"] = false;
  jmaterial["friction"] = 0.;
  jmaterial["dilation"] = 0.;
  jmaterial["cohesion"] = 2000.;
  jmaterial["residual_friction"] = 0.;
  jmaterial["residual_dilation"] = 0.;
  jmaterial["residual_cohesion"] = 1000.;
  jmaterial["peak_pdstrain"] = 0.;
  jmaterial["residual_pdstrain"] = 0.;
  jmaterial["tension_cutoff"] = 0.;
  jmaterial["packing_fraction"] = 0.6;

  //! Check for id = 0
  SECTION("MohrCoulomb id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  //! Check for positive id
  SECTION("MohrCoulomb id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("MohrCoulomb failed initialisation") {
    unsigned id = 0;
    auto logger = std::make_unique<spdlog::logger>("MC_Test", mpm::stdout_sink);
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    try {
      auto material =
          Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
              ->create("MohrCoulomb2D", std::move(id), jmaterial);
    } catch (std::exception& except) {
      logger->error("MohrCoulomb initialization failed: {}", except.what());
    }
  }

  //! Check material properties
  SECTION("MohrCoulomb check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("friction") ==
            Approx(jmaterial["friction"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dilation") ==
            Approx(jmaterial["dilation"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tension_cutoff") ==
            Approx(jmaterial["tension_cutoff"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("packing_fraction") ==
            Approx(jmaterial["packing_fraction"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.at("phi") ==
              Approx(jmaterial["friction"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("tension_cutoff") ==
              Approx(1.e-15).epsilon(Tolerance));
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "yield_state", "phi", "psi",   "cohesion", "tension_cutoff",
          "epsilon",     "rho", "theta", "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  //! Check yield correction based on trial stress
  SECTION("MohrCoulomb check yield correction based on trial stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    stress(0) = -5000.;
    stress(1) = -6000.;
    stress(2) = -7000.;
    stress(3) = -1000.;

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();
    // Check if stress invariants is computed correctly based on stress
    REQUIRE(mohr_coulomb->compute_stress_invariants(stress, &state_variables) ==
            true);
    REQUIRE(state_variables.at("phi") ==
            Approx(jmaterial["friction"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("psi") ==
            Approx(jmaterial["dilation"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("epsilon") ==
            Approx(-10392.30484541).epsilon(Tolerance));
    REQUIRE(state_variables.at("rho") == Approx(2000.).epsilon(Tolerance));
    REQUIRE(state_variables.at("theta") ==
            Approx(0.13545926).epsilon(Tolerance));
    REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

    // Initialise values of yield functions
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type =
        mohr_coulomb->compute_yield_state(&yield_function, state_variables);
    // Check if yield function and yield state is computed correctly
    REQUIRE(yield_function(0) == Approx(-4381.96601125).epsilon(Tolerance));
    REQUIRE(yield_function(1) == Approx(-690.98300563).epsilon(Tolerance));
    REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

    // Initialise plastic correction components
    mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
    double dp_dq = 0.;
    df_dsigma.setZero();
    dp_dsigma.setZero();
    double softening = 0.;
    // Compute plastic correction components
    mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                &df_dsigma, &dp_dsigma, &dp_dq, &softening);
    // Check plastic correction component based on stress
    // Check dF/dSigma
    REQUIRE(df_dsigma(0) == Approx(0.3618034).epsilon(Tolerance));
    REQUIRE(df_dsigma(1) == Approx(0.1381966).epsilon(Tolerance));
    REQUIRE(df_dsigma(2) == Approx(-0.5).epsilon(Tolerance));
    REQUIRE(df_dsigma(3) == Approx(2.0 * -0.2236068).epsilon(Tolerance));
    REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
    // Check dP/dSigma
    REQUIRE(dp_dsigma(0) == Approx(0.30618622).epsilon(Tolerance));
    REQUIRE(dp_dsigma(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(2) == Approx(-0.30618622).epsilon(Tolerance));
    REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.30618622).epsilon(Tolerance));
    REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

    //! Check for shear failure
    SECTION("Check yield correction for shear failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = -0.001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(jmaterial["friction"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-24826.06157515).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(5297.46320146).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.89359516).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(-11623.00067857).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(1492.38393682).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(-0.47906443).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.47906443).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * -0.14316868).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(-0.47720936).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.29640333).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.18080603).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * -0.11559730).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      std::cout << "PARTICLE_MASS_DENSITY: " << particle->mass_density()
                << std::endl;

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) ==
              Approx(-16697.4520573296).epsilon(Tolerance));
      REQUIRE(updated_stress(1) ==
              Approx(-12864.9366103543).epsilon(Tolerance));
      REQUIRE(updated_stress(2) ==
              Approx(-13437.6113323161).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(-572.6747219618).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  9105518.49606,     10125250.7347,     5769230.76923,    -1301799.07014,                 0,                 0,
                    8474833.23226,     10755935.9985,     5769230.76923,     808570.851021,                 0,                 0,
                    7419648.27168,     4118813.26678,     13461538.4615,     493228.219123,                 0,                 0,
                   -1055184.96058,     1055184.96058, 3.33596483125e-10,     3530811.21426,                 0,                 0,
                                0,                 0,                 0,                 0,     3846153.84615,                 0,
                                0,                 0,                 0,                 0,                 0,     3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for tensile failure
    SECTION("Check yield correction for tensile failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(jmaterial["friction"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(4041.45188433).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(7670.22471249).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.08181078).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(8575.09909665).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(2902.93416371).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.98726817).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.01273183).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * -0.11211480).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.87816487).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.06958427).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.05225086).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * -0.09302255).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-920.7979744249).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-4207.8691181054).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-4877.8540405517).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(-378.1586271491).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  183764.142704, -60255.5706282,  37052.5716226,  856882.687847,              0,              0,
                   -939059.937324,   10516324.294,  2873179.30699,  432920.309794,              0,              0,
                   -753803.520658,  2905351.75049,  10645464.4689,  420964.765707,              0,              0,
                    755781.300528,  331818.922475,  326280.066901,  3797379.41328,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }

  //! Check yield correction based on current stress
  SECTION("MohrCoulomb check yield correction based on current stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    stress(0) = -2000.;
    stress(1) = -5000.;
    stress(2) = -6000.;

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();
    // Check if stress invariants is computed correctly based on stress
    REQUIRE(mohr_coulomb->compute_stress_invariants(stress, &state_variables) ==
            true);
    REQUIRE(state_variables.at("phi") ==
            Approx(jmaterial["friction"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("psi") ==
            Approx(jmaterial["dilation"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("epsilon") ==
            Approx(-7505.55349947).epsilon(Tolerance));
    REQUIRE(state_variables.at("rho") ==
            Approx(2943.92028878).epsilon(Tolerance));
    REQUIRE(state_variables.at("theta") ==
            Approx(0.24256387).epsilon(Tolerance));
    REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

    // Initialise values of yield functions
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type =
        mohr_coulomb->compute_yield_state(&yield_function, state_variables);
    // Check if yield function and yield state is computed correctly
    REQUIRE(yield_function(0) == Approx(-2000.).epsilon(Tolerance));
    REQUIRE(yield_function(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

    // Initialise plastic correction components
    mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
    double dp_dq = 0.;
    df_dsigma.setZero();
    dp_dsigma.setZero();
    double softening = 0.;
    // Compute plastic correction components
    mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                &df_dsigma, &dp_dsigma, &dp_dq, &softening);
    // Check plastic correction component based on stress
    // Check dF/dSigma
    REQUIRE(df_dsigma(0) == Approx(0.5).epsilon(Tolerance));
    REQUIRE(df_dsigma(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(2) == Approx(-0.5).epsilon(Tolerance));
    REQUIRE(df_dsigma(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
    // Check dP/dSigma
    REQUIRE(dp_dsigma(0) == Approx(0.48536267).epsilon(Tolerance));
    REQUIRE(dp_dsigma(1) == Approx(-0.13867505).epsilon(Tolerance));
    REQUIRE(dp_dsigma(2) == Approx(-0.34668762).epsilon(Tolerance));
    REQUIRE(dp_dsigma(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

    //! Check for shear failure
    SECTION("Check yield correction for shear failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(jmaterial["friction"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(6928.20323028).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(9165.79698223).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.07722297).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(11461.53846154).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(3846.15384615).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(1.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.88874614).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.05882082).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.05243303).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-1348.42814081).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-488.58203346).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<              0, 9.31322574615e-10,                 0,                 0,                 0,                 0,
                   -689009.219446,     10693721.3235,     3001413.63123,                 0,                 0,                 0,
                   -1310545.07741,     2735041.12067,      10427348.813,                 0,                 0,                 0,
                                0,                 0,                 0,     3846153.84615,                 0,                 0,
                                0,                 0,                 0,                 0,     3846153.84615,                 0,
                                0,                 0,                 0,                 0,                 0,     3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }
}

//! Check MohrCoulomb class in 2D
//! Cohesion and friction, without softening
TEST_CASE("MohrCoulomb is checked in 2D (c & phi, without softening)",
          "[material][mohr_coulomb][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

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
  jmaterial["softening"] = false;
  jmaterial["friction"] = 30.;
  jmaterial["dilation"] = 0.;
  jmaterial["cohesion"] = 2000.;
  jmaterial["residual_friction"] = 0.;
  jmaterial["residual_dilation"] = 0.;
  jmaterial["residual_cohesion"] = 1000.;
  jmaterial["peak_pdstrain"] = 0.;
  jmaterial["residual_pdstrain"] = 0.;
  jmaterial["tension_cutoff"] = 0.;
  jmaterial["packing_fraction"] = 0.6;

  //! Check yield correction based on trial stress
  SECTION("MohrCoulomb check yield correction based on trial stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    stress(0) = -5000.;
    stress(1) = -6000.;
    stress(2) = -7000.;
    stress(3) = -1000.;

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();
    // Check if stress invariants is computed correctly based on stress
    REQUIRE(mohr_coulomb->compute_stress_invariants(stress, &state_variables) ==
            true);
    REQUIRE(state_variables.at("phi") == Approx(0.52359878).epsilon(Tolerance));
    REQUIRE(state_variables.at("psi") ==
            Approx(jmaterial["dilation"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("epsilon") ==
            Approx(-10392.30484541).epsilon(Tolerance));
    REQUIRE(state_variables.at("rho") == Approx(2000.).epsilon(Tolerance));
    REQUIRE(state_variables.at("theta") ==
            Approx(0.13545926).epsilon(Tolerance));
    REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_variables.at("tension_cutoff") ==
            Approx(1.e-15).epsilon(Tolerance));
    REQUIRE(state_variables.at("yield_state") == Approx(0.).epsilon(Tolerance));

    // Initialise values of yield functions
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type =
        mohr_coulomb->compute_yield_state(&yield_function, state_variables);
    // Check if yield function and yield state is computed correctly
    REQUIRE(yield_function(0) == Approx(-4381.96601125).epsilon(Tolerance));
    REQUIRE(yield_function(1) == Approx(-3774.1679421).epsilon(Tolerance));
    REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

    // Initialise plastic correction components
    mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
    double dp_dq = 0.;
    df_dsigma.setZero();
    dp_dsigma.setZero();
    double softening = 0.;
    // Compute plastic correction components
    mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                &df_dsigma, &dp_dsigma, &dp_dq, &softening);
    // Check plastic correction component based on stress
    // Check dF/dSigma
    REQUIRE(df_dsigma(0) == Approx(0.62666187).epsilon(Tolerance));
    REQUIRE(df_dsigma(1) == Approx(0.23936353).epsilon(Tolerance));
    REQUIRE(df_dsigma(2) == Approx(-0.28867513).epsilon(Tolerance));
    REQUIRE(df_dsigma(3) == Approx(2.0 * -0.38729833).epsilon(Tolerance));
    REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
    // Check dP/dSigma
    REQUIRE(dp_dsigma(0) == Approx(0.39868466).epsilon(Tolerance));
    REQUIRE(dp_dsigma(1) == Approx(-0.04368136).epsilon(Tolerance));
    REQUIRE(dp_dsigma(2) == Approx(-0.35500330).epsilon(Tolerance));
    REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.44236602).epsilon(Tolerance));
    REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

    //! Check for shear failure
    SECTION("Check yield correction for shear failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.002;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(9669.89676021).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.36378823).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(2212.05893238).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(3174.54763108).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.36433344).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.21301683).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.57237152).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.20717328).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.07007866).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.27725194).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.51857529).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-4615.3075306146).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5748.3019110924).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-5136.390558293).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(4285.6743957206).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  11754588.9102,  4386218.09206,  4842242.10068, -1225326.43698,              0,              0,
                    5312658.03793,  13091611.9602,  5521280.99945, -327748.782988,              0,              0,
                    7932753.05185,  7522169.94772,  14636476.8999,  1553075.21996,              0,              0,
                   -4729686.23224, -3832108.57825, -2568538.44315,  450968.807151,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for tensile failure
    SECTION("Check yield correction for tensile failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(4041.45188433).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(7670.22471249).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.08181078).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(8575.09909665).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(5781.54613102).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.98726817).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.01273183).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * -0.11211480).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.87816487).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.06958427).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.05225086).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * -0.09302255).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-58.2831643542).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-4519.4698892777).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-5428.76807945).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(-513.2338710089).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  183764.142704, -60255.5706282,  37052.5716226,  856882.687847,              0,              0,
                   -939059.937324,   10516324.294,  2873179.30699,  432920.309794,              0,              0,
                   -753803.520658,  2905351.75049,  10645464.4689,  420964.765707,              0,              0,
                    755781.300528,  331818.922475,  326280.066901,  3797379.41328,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }

  //! Check yield correction based on current stress
  SECTION("MohrCoulomb check yield correction based on current stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    stress(0) = -1000.;
    stress(1) = -7000.;
    stress(2) = -9928.20323028;

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();
    // Check if stress invariants is computed correctly based on stress
    REQUIRE(mohr_coulomb->compute_stress_invariants(stress, &state_variables) ==
            true);
    REQUIRE(state_variables.at("phi") == Approx(0.52359878).epsilon(Tolerance));
    REQUIRE(state_variables.at("psi") ==
            Approx(jmaterial["dilation"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("epsilon") ==
            Approx(-10350.85296109).epsilon(Tolerance));
    REQUIRE(state_variables.at("rho") ==
            Approx(6436.54117983).epsilon(Tolerance));
    REQUIRE(state_variables.at("theta") ==
            Approx(0.32751078).epsilon(Tolerance));
    REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

    // Initialise values of yield functions
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type =
        mohr_coulomb->compute_yield_state(&yield_function, state_variables);
    // Check if yield function and yield state is computed correctly
    REQUIRE(yield_function(0) == Approx(-1000.).epsilon(Tolerance));
    REQUIRE(yield_function(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

    // Initialise plastic correction components
    mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
    double dp_dq = 0.;
    df_dsigma.setZero();
    dp_dsigma.setZero();
    double softening = 0.;
    // Compute plastic correction components
    mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                &df_dsigma, &dp_dsigma, &dp_dq, &softening);
    ;
    // Check plastic correction component based on stress
    // Check dF/dSigma
    REQUIRE(df_dsigma(0) == Approx(0.86602540).epsilon(Tolerance));
    REQUIRE(df_dsigma(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(2) == Approx(-0.28867513).epsilon(Tolerance));
    REQUIRE(df_dsigma(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
    // Check dP/dSigma
    REQUIRE(dp_dsigma(0) == Approx(0.66416840).epsilon(Tolerance));
    REQUIRE(dp_dsigma(1) == Approx(-0.28438471).epsilon(Tolerance));
    REQUIRE(dp_dsigma(2) == Approx(-0.37978369).epsilon(Tolerance));
    REQUIRE(dp_dsigma(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

    //! Check for shear failure
    SECTION("Check yield correction for shear failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(jmaterial["dilation"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8907.47728811).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(8891.8404917).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.09473338).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(2084.86947857).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(2505.03198892).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.71907297).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.14695243).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(-0.28867513).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.32506849).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.50383165).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.15419846).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.34963318).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.37388074).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-1790.7690510275).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5719.5499679831).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-7917.8842112694).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(2805.9964832764).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  7688392.86819,   2569156.5921,  5021462.45502, -1837727.65592,              0,              0,
                    7168160.90157,  14236970.1772,   5950427.5899,  445312.274102,              0,              0,
                    10143446.2302,  8193873.23072,  14028109.9551,  1392415.38182,              0,              0,
                   -5122408.11831, -2839368.18829, -663481.358883,  2215571.19208,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }
}

//! Check MohrCoulomb class in 2D
//! Cohesion, friction and dilation, without softening
TEST_CASE("MohrCoulomb is checked in 2D (c & phi & psi, without softening)",
          "[material][mohr_coulomb][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

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
  jmaterial["softening"] = false;
  jmaterial["friction"] = 30.;
  jmaterial["dilation"] = 15.;
  jmaterial["cohesion"] = 2000.;
  jmaterial["residual_friction"] = 0.;
  jmaterial["residual_dilation"] = 0.;
  jmaterial["residual_cohesion"] = 1000.;
  jmaterial["peak_pdstrain"] = 0.;
  jmaterial["residual_pdstrain"] = 0.;
  jmaterial["tension_cutoff"] = 0.;
  jmaterial["packing_fraction"] = 0.6;

  //! Check yield correction based on trial stress
  SECTION("MohrCoulomb check yield correction based on trial stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    stress(0) = -5000.;
    stress(1) = -6000.;
    stress(2) = -7000.;
    stress(3) = -1000.;

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();
    // Check if stress invariants is computed correctly based on stress
    REQUIRE(mohr_coulomb->compute_stress_invariants(stress, &state_variables) ==
            true);
    REQUIRE(state_variables.at("phi") == Approx(0.52359878).epsilon(Tolerance));
    REQUIRE(state_variables.at("psi") == Approx(0.26179939).epsilon(Tolerance));
    REQUIRE(state_variables.at("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("epsilon") ==
            Approx(-10392.30484541).epsilon(Tolerance));
    REQUIRE(state_variables.at("rho") == Approx(2000.).epsilon(Tolerance));
    REQUIRE(state_variables.at("theta") ==
            Approx(0.13545926).epsilon(Tolerance));
    REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

    // Initialise values of yield functions
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type =
        mohr_coulomb->compute_yield_state(&yield_function, state_variables);
    // Check if yield function and yield state is computed correctly
    REQUIRE(yield_function(0) == Approx(-4381.96601125).epsilon(Tolerance));
    REQUIRE(yield_function(1) == Approx(-3774.1679421).epsilon(Tolerance));
    REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

    // Initialise plastic correction components
    mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
    double dp_dq = 0.;
    df_dsigma.setZero();
    dp_dsigma.setZero();
    double softening = 0.;
    // Compute plastic correction components
    mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                &df_dsigma, &dp_dsigma, &dp_dq, &softening);

    // Check plastic correction component based on stress
    // Check dF/dSigma
    REQUIRE(df_dsigma(0) == Approx(0.62666187).epsilon(Tolerance));
    REQUIRE(df_dsigma(1) == Approx(0.23936353).epsilon(Tolerance));
    REQUIRE(df_dsigma(2) == Approx(-0.28867513).epsilon(Tolerance));
    REQUIRE(df_dsigma(3) == Approx(2.0 * -0.38729833).epsilon(Tolerance));
    REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
    // Check dP/dSigma
    REQUIRE(dp_dsigma(0) == Approx(0.48778797).epsilon(Tolerance));
    REQUIRE(dp_dsigma(1) == Approx(0.04565839).epsilon(Tolerance));
    REQUIRE(dp_dsigma(2) == Approx(-0.26549716).epsilon(Tolerance));
    REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.44212958).epsilon(Tolerance));
    REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

    //! Check for shear failure
    SECTION("Check yield correction for shear failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.002;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(9669.89676021).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.36378823).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(2212.05893238).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(3174.54763108).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.36433344).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.21301683).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.57237152).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.29648451).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.15939331).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.18792863).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.51856235).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-5508.1587191762).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-6766.3597033789).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-6471.6605527693).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(4759.2819837233).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check << 9952170.39151,  2925854.10003,  3863407.34746,   -2519184.864,              0,              0,
                   3257482.16826,   11426457.721,  4405181.96677, -1803047.99367,              0,              0,
                   5480576.55012,    5535355.993,  13304779.7629, -207209.196986,              0,              0,
                  -3773604.07862,  -3057467.2083, -2049321.38608,  1137288.29318,              0,              0,
                               0,              0,              0,              0,  3846153.84615,              0,
                               0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for tensile failure
    SECTION("Check yield correction for tensile failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(4041.45188433).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(7670.22471249).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.08181078).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(8575.09909665).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(5781.54613102).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.98726817).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.01273183).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * -0.11211480).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.87816487).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.06958427).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.05225086).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * -0.09302255).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-58.2831643542).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-4519.4698892777).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-5428.76807945).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(-513.2338710089).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check << 183764.142704, -60255.5706282,  37052.5716226,  856882.687847,              0,              0,
                  -939059.937324,   10516324.294,  2873179.30699,  432920.309794,              0,              0,
                  -753803.520658,  2905351.75049,  10645464.4689,  420964.765707,              0,              0,
                   755781.300528,  331818.922475,  326280.066901,  3797379.41328,              0,              0,
                               0,              0,              0,              0,  3846153.84615,              0,
                               0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }

  //! Check yield correction based on current stress
  SECTION("MohrCoulomb check yield correction based on current stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    stress(0) = -1000.;
    stress(1) = -7000.;
    stress(2) = -9928.20323028;

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();
    // Check if stress invariants is computed correctly based on stress
    REQUIRE(mohr_coulomb->compute_stress_invariants(stress, &state_variables) ==
            true);
    REQUIRE(state_variables.at("phi") == Approx(0.52359878).epsilon(Tolerance));
    REQUIRE(state_variables.at("psi") == Approx(0.26179939).epsilon(Tolerance));
    REQUIRE(state_variables.at("cohesion") ==
            Approx(jmaterial["cohesion"]).epsilon(Tolerance));
    REQUIRE(state_variables.at("epsilon") ==
            Approx(-10350.85296109).epsilon(Tolerance));
    REQUIRE(state_variables.at("rho") ==
            Approx(6436.54117983).epsilon(Tolerance));
    REQUIRE(state_variables.at("theta") ==
            Approx(0.32751078).epsilon(Tolerance));
    REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

    // Initialise values of yield functions
    Eigen::Matrix<double, 2, 1> yield_function;
    auto yield_type =
        mohr_coulomb->compute_yield_state(&yield_function, state_variables);
    // Check if yield function and yield state is computed correctly
    REQUIRE(yield_function(0) == Approx(-1000.).epsilon(Tolerance));
    REQUIRE(yield_function(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

    // Initialise plastic correction components
    mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
    double dp_dq = 0.;
    df_dsigma.setZero();
    dp_dsigma.setZero();
    double softening = 0.;
    // Compute plastic correction components
    mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                &df_dsigma, &dp_dsigma, &dp_dq, &softening);
    ;
    // Check plastic correction component based on stress
    // Check dF/dSigma
    REQUIRE(df_dsigma(0) == Approx(0.86602540).epsilon(Tolerance));
    REQUIRE(df_dsigma(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(2) == Approx(-0.28867513).epsilon(Tolerance));
    REQUIRE(df_dsigma(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
    // Check dP/dSigma
    REQUIRE(dp_dsigma(0) == Approx(0.75344809).epsilon(Tolerance));
    REQUIRE(dp_dsigma(1) == Approx(-0.19505260).epsilon(Tolerance));
    REQUIRE(dp_dsigma(2) == Approx(-0.29044631).epsilon(Tolerance));
    REQUIRE(dp_dsigma(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

    //! Check for shear failure
    SECTION("Check yield correction for shear failure") {
      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8907.47728811).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(8891.8404917).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.09473338).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(2084.86947857).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(2505.03198892).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.71907297).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.14695243).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(-0.28867513).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.32506849).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.59313451).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.06487792).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.26030739).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.37387070).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-2277.6593888847).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-6762.4063766522).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-9133.9391756068).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(3048.9962358498).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  5749844.69116,  1582320.57796,  4777680.43909,  -2396356.3372,              0,              0,
                    4049916.63137,  12528071.1715,  5548165.65023, -534265.163088,              0,              0,
                    6364488.73556,  6092414.19947,  13538075.2446,  184972.360465,              0,              0,
                   -4073974.07104, -2211882.89692, -523821.413996,  2580194.13958,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }
}

//! Check MohrCoulomb class in 2D
//! Cohesion, friction and dilation, with softening
TEST_CASE("MohrCoulomb is checked in 2D (c & phi & psi, with softening)",
          "[material][mohr_coulomb][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

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
  jmaterial["softening"] = true;
  jmaterial["friction"] = 30.;
  jmaterial["dilation"] = 15.;
  jmaterial["cohesion"] = 2000.;
  jmaterial["residual_friction"] = 0.;
  jmaterial["residual_dilation"] = 0.;
  jmaterial["residual_cohesion"] = 1000.;
  jmaterial["peak_pdstrain"] = 0.;
  jmaterial["residual_pdstrain"] = 0.001;
  jmaterial["tension_cutoff"] = 0.;
  jmaterial["packing_fraction"] = 0.6;

  //! Check yield correction based on trial stress
  SECTION("MohrCoulomb check yield correction based on trial stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();

    //! Check for shear failure ( pdstrain_peak < pdstrain <  pdstrain_residual)
    SECTION(
        "Check for shear failure ( pdstrain_peak < pdstrain <  "
        "pdstrain_residual)") {

      // Tolerance for computation of stress
      const double Tolerance_stress = 1.E-5;

      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -7000.;
      stress(3) = -1000.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") == Approx(2000.).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.13545926).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      state_variables.at("pdstrain") = 0.00004761;
      // Modified MC parameters
      state_variables.at("phi") = 0.49867048772358;
      state_variables.at("psi") = 0.24933524386179;
      state_variables.at("cohesion") = 1952.39047714;
      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-4381.96601125).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-3561.03580708).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.60900389).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.23261879).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.29704523).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.3763851).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.47570740).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(0.04310326).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.26417672).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.43260415).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.002;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.49867048772358).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.24933524386179).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1952.39047714).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(9669.89676021).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.36378823).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00004761).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(2212.05893238).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(3262.66672836).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.34689653).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.19768091).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.56442433).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.28593455).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.15124777).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.18254838).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.50946739).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) ==
              Approx(-6033.5560234765).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(1) ==
              Approx(-7130.7196599537).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(2) ==
              Approx(-6465.1265437883).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(3) ==
              Approx(4150.140711892).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance_stress));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.000455237).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  7978638.63652,  1369480.14981,   2804435.6359, -4097129.60376,              0,              0,
                    1915830.08314,  10369378.7292,  3685562.64369, -2879476.65104,              0,              0,
                    5142227.72325,  5266092.44317,  13122496.0499, -468531.766643,              0,              0,
                   -6163757.61245, -4946104.65973, -3332958.68166, -759750.801089,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain <  pdstrain_peak)
    SECTION("Check for shear failure (pdstrain <  pdstrain_peak)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -7000.;
      stress(3) = -1000.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") == Approx(2000.).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.13545926).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-4381.96601125).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-3774.1679421).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.62666187).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.23936353).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.28867513).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.38729833).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.48778797).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(0.04565839).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.26549716).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.44212958).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.002;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(9669.89676021).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.36378823).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(2212.05893238).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(3174.54763108).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.36433344).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.21301683).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.57237152).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.29648451).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(0.15939331).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.18792863).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.51856235).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-5508.1587191762).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-6766.3597033789).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-6471.6605527693).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(4759.2819837233).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0003103425).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<   8132157.0739,  1451234.36746,  2875017.43241, -3825673.64218,              0,              0,
                    1954849.95271,  10371032.9333,   3697764.8658, -2738136.95992,              0,              0,
                     5330875.9452,  5414064.86625,  13223482.2434,  -314671.13615,              0,              0,
                   -5730654.33423, -4643117.65197, -3112131.59586, -267571.865006,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain >  pdstrain_residual)
    SECTION("Check for shear failure (pdstrain >  pdstrain_residual)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -6500.;
      stress(3) = 0.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10103.62971082).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(1080.12344973).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.33347317).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      state_variables.at("pdstrain") = 0.00129099;
      // Modified MC parameters
      state_variables.at("phi") = 0.;
      state_variables.at("psi") = 0.;
      state_variables.at("cohesion") = 1000.;

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-5000.).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-250.).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.47245559).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.09449112).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.37796447).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.002;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1000.).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8660.25403784).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(11008.46903673).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.42072067).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00129099).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(3204.54446772).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(6743.00600619).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.05712351).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(-0.05712351).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.49672619).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.07488303).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.02353467).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.05134836).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.42790303).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-4826.1458346954).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5054.6398805243).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-5119.2142847803).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(993.452373169).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0023085701).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<   13385144.92,       5845624.3108,      5769230.76923,     -664291.665797,                  0,                  0,
                   5793240.16801,      13437529.0628,      5769230.76923,      208777.380679,                  0,                  0,
                   5821614.91202,      5716846.62644,      13461538.4615,      455514.285118,                  0,                  0,
                  -436534.523238,      436534.523238, -3.62569791679e-09,      50201.4701724,                  0,                  0,
                               0,                  0,                  0,                  0,      3846153.84615,                  0,
                               0,                  0,                  0,                  0,                  0,      3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }

  //! Check yield correction based on current stress
  SECTION("MohrCoulomb check yield correction based on current stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb2D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();

    //! Check for shear failure ( pdstrain_peak < pdstrain <  pdstrain_residual)
    SECTION(
        "Check for shear failure ( pdstrain_peak < pdstrain <  "
        "pdstrain_residual)") {

      // Tolerance for computation of stress
      const double Tolerance_stress = 1.E-5;

      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -7000.;
      stress(3) = -4186.6;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(6087.30146452).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.32101934).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      state_variables.at("pdstrain") = 0.00009117;
      // Modified MC parameters
      state_variables.at("phi") = 0.47586473847588;
      state_variables.at("psi") = 0.23793236923794;
      state_variables.at("cohesion") = 1908.83470445882;
      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-1283.64854880).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(0.00464538).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.32438701).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.19097903).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.55852584).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.27456889).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(0.15490510).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.18694764).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.50098440).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0005;
      dstrain(1) = -0.0005;
      dstrain(2) = 0.;
      dstrain(3) = 0.0001;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.47586473847588).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.23793236923794).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1908.83470445882).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(8257.6195444).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.3747326).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00009117).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(274.43852423).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(1752.8367834).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.68104703).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(-0.16568099).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * -0.37035584).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.59002858).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.17216973).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.17533250).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * -0.33338284).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) ==
              Approx(-3619.2832146878).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(1) ==
              Approx(-9920.2880566994).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(2) ==
              Approx(-7022.9047083088).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(3) ==
              Approx(-2701.4900468407).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance_stress));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0003514387).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  -842057.884925,  2869052.25808, 608098.311947,  4889096.35121,              0,              0,
                     5943806.09973,  13496935.1275, 5832222.36817, -59671.3994647,              0,              0,
                      5117718.9962,  5637131.09317, 13226455.0268,  222692.514186,              0,              0,
                     6207364.96201,  1258597.21133,   2239788.652,  1724421.33558,              0,              0,
                                 0,              0,             0,              0,  3846153.84615,              0,
                                 0,              0,             0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain <  pdstrain_peak)
    SECTION("Check for shear failure (pdstrain <  pdstrain_peak)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -1000.;
      stress(1) = -6000.;
      stress(2) = -9350.4;
      stress(3) = -1000.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-9439.90784136).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(6108.85587542).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.37419232).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-807.41759643).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-0.01617146).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.8350549).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.0309705).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.28867513).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.16081688).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.71673988).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.15232546).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.29646522).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.17381307).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-7996.53216838).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(7665.51627945).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.20272541).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(1513.89550814).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(1843.75660036).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.74124692).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.12477848).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(-0.28867513).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.30412443).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.61875527).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.07639146).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.27441462).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.34293905).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-1350.6363584257).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5681.3871173645).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-8625.7417682106).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(2003.6532357271).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0001796693).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  1155960.57855,  70934.0683857,  4268907.88724, -3056906.54043,              0,              0,
                    3958227.83255,  12622924.2853,  5548429.34316, -449882.709655,              0,              0,
                    6971101.19316,  6325776.29165,  13608073.1237,  298564.244165,              0,              0,
                   -4855391.18255, -2248367.35177,  -591979.87786,  2639995.37043,              0,              0,
                                0,              0,              0,              0,  3846153.84615,              0,
                                0,              0,              0,              0,              0,  3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain >  pdstrain_residual)
    SECTION("Check for shear failure (pdstrain >  pdstrain_residual)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -7000.;
      stress(3) = 0;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(1414.21356237).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      state_variables.at("pdstrain") = 0.00129099;
      // Modified MC parameters
      state_variables.at("phi") = 0.;
      state_variables.at("psi") = 0.;
      state_variables.at("cohesion") = 1000.;

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-5000.).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(0.).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.43301270).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.4330127).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(0.).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.002;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1000.).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(11057.85395646).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.38398831).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00129099).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(3204.54446772).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(6743.00600619).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.05712351).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(-0.05712351).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.49672619).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.08377842).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.01419973).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.06957869).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.42599199).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) == Approx(0.).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-5020.5904156363).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5199.8658146281).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-5279.5437697356).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(995.9744388432).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(0.).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(0.).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0023627923).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<     13411177.1573,     5819592.07344,     5769230.76923,    -559570.046724,                 0,                 0,
                       5780676.52019,     13450092.7106,     5769230.76923,     127175.010619,                 0,                 0,
                       5808146.32248,     5730315.21598,     13461538.4615,     432395.036105,                 0,                 0,
                      -343372.528672,     343372.528672, 4.48273959994e-09,     30903.5275804,                 0,                 0,
                                   0,                 0,                 0,                 0,     3846153.84615,                 0,
                                   0,                 0,                 0,                 0,                 0,     3846153.84615;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }
}

//! Check MohrCoulomb class in 3D
//! Cohesion, friction and dilation, with softening
TEST_CASE("MohrCoulomb is checked in 3D (c & phi & psi, with softening)",
          "[material][mohr_coulomb][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  const double dt = 1.0;

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
  jmaterial["softening"] = true;
  jmaterial["friction"] = 30.;
  jmaterial["dilation"] = 15.;
  jmaterial["cohesion"] = 2000.;
  jmaterial["residual_friction"] = 0.;
  jmaterial["residual_dilation"] = 0.;
  jmaterial["residual_cohesion"] = 1000.;
  jmaterial["peak_pdstrain"] = 1.E-16;
  jmaterial["residual_pdstrain"] = 0.001;
  jmaterial["tension_cutoff"] = 0.;
  jmaterial["packing_fraction"] = 0.6;

  //! Check yield correction based on trial stress
  SECTION("MohrCoulomb check yield correction based on trial stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb3D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();

    //! Check for shear failure ( pdstrain_peak < pdstrain <  pdstrain_residual)
    SECTION(
        "Check for shear failure ( pdstrain_peak < pdstrain <  "
        "pdstrain_residual)") {

      // Tolerance for computation of stress
      const double Tolerance_stress = 1.E-5;

      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -7000.;
      stress(3) = -1000.;
      stress(4) = -2000.;
      stress(5) = -3000.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(5477.22557505).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.76870359).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Define current plastic strain
      state_variables.at("pdstrain") = 0.00009110433579;
      // Modified MC parameters
      state_variables.at("phi") = 0.4758966569262;
      state_variables.at("psi") = 0.2379483284631;
      state_variables.at("cohesion") = 1908.89566421;
      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-2785.3725926).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-1054.08171963).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.41269182).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(-0.04682703).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(0.14954165).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * 0.02146534).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(2.0 * -0.17569850).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(2.0 * -0.50403985).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.28435720).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.09012669).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(0.04831274).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * 0.00165987).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(2.0 * -0.16765469).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(2.0 * -0.43955392).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      dstrain(4) = 0.002;
      dstrain(5) = 0.003;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.4758966569262).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.2379483284631).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1908.89566421).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(15190.44130048).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.33392734).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00009110433579).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(6551.16818741).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(7898.08532474).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.25450103).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.13260639).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.12829902).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.18427295).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) ==
              Approx(2.0 * 0.30524021).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) ==
              Approx(2.0 * 0.45418703).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.15987109).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.01440272).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.09707488).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.24193193).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) ==
              Approx(2.0 * 0.27805767).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) ==
              Approx(2.0 * 0.40514467).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) ==
              Approx(-7322.6057761629).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(1) ==
              Approx(-7221.1260046918).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(2) ==
              Approx(-9417.7620537043).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(3) ==
              Approx(249.17936472).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(4) ==
              Approx(2707.5480094066).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(5) ==
              Approx(4189.5095402315).epsilon(Tolerance_stress));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0009600559).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  9215134.72118,  2330266.53067,  2358798.93664, -1220638.29308, -2021934.87571, -3008570.14024,
                    3159522.29215,  11348057.3433,  3673284.78571, -750166.563442, -1242618.67403, -1848974.20946,
                    2748578.89575,  3322945.77016,  11035549.8052, -868293.165761,  -1438290.3143, -2140126.93711,
                   -2388481.22167, -1934319.48727, -1918270.82115,  3159579.55418, -1137280.80921, -1692235.05899,
                   -3467488.06706, -2808156.78145, -2784858.06016, -996737.233253,  2195101.45392, -2456709.65317,
                   -5125250.25407, -4150701.02023, -4116263.47498, -1473264.69743, -2440399.65794,  214922.381021;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain <  pdstrain_peak)
    SECTION("Check for shear failure (pdstrain <  pdstrain_peak)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -7000.;
      stress(3) = -1000.;
      stress(4) = -2000.;
      stress(5) = -3000.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10392.30484541).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(5477.22557505).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.76870359).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-2785.37259260).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-1438.89947208).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.44409275).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(-0.04248842).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(0.17574594).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * 0.03028355).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(2.0 * -0.17437140).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(2.0 * -0.51998947).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.30645560).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.10320141).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(0.06469501).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * 0.01388217).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(2.0 * -0.16475347).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(2.0 * -0.45889980).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      dstrain(4) = 0.002;
      dstrain(5) = 0.003;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8948.92917244).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(15190.44130048).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.33392734).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(6551.16818741).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(7872.57466925).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.27802532).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.14348948).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.15583547).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.20025425).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) ==
              Approx(2.0 * 0.31387747).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) ==
              Approx(2.0 * 0.46578947).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.17141044).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.01406976).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(0.11060851).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.25638521).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) ==
              Approx(2.0 * 0.28781782).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) ==
              Approx(2.0 * 0.41867392).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-7093.4329819478).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-7149.3913315503).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-9301.0389519637).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(477.9349866159).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(3033.7472196572).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(4671.1890153894).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0007751567).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<   9284775.7303,  2382755.31319,  2310232.76763, -1176328.74898, -1843771.63129, -2736129.50159,
                     3321865.1738,  11477240.0816,   3742437.9518, -689267.428969, -1080354.22326,  -1603229.4956,
                    2574011.19323,   3178580.2183,  10815408.2312, -899890.390816, -1410483.57043, -2093136.50512,
                    -2466375.9816, -1999711.80182,  -2042536.3231,  3151532.39896, -1088746.08389, -1615682.89115,
                   -3241708.94335, -2628343.64283, -2684630.53286, -912983.492542,  2415148.21542, -2123590.93542,
                   -4764441.16572, -3862958.97269, -3945685.57789, -1341840.43401, -2103193.79519,  725045.736611;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain >  pdstrain_residual)
    SECTION("Check for shear failure (pdstrain >  pdstrain_residual)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -6000.;
      stress(3) = -100.;
      stress(4) = -200.;
      stress(5) = -300.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-9814.95457622).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(972.96796796).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.33010649).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Define current plastic strain
      state_variables.at("pdstrain") = 0.00244948974278;
      // Modified MC parameters
      state_variables.at("phi") = 0.;
      state_variables.at("psi") = 0.;
      state_variables.at("cohesion") = 1000.;

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-4915.13437757).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-324.84658240).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Elastic);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.44026109).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(-0.20330136).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.23695973).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.09170367).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(2.0 * -0.22968759).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(2.0 * -0.20779427).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.41959068).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.20979534).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.20979534).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.06293860).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(2.0 * -0.12587720).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(2.0 * -0.18881581).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      dstrain(4) = 0.002;
      dstrain(5) = 0.003;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1000.).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8371.57890325).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(19875.34922720).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.30645028).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00244948974278).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(10638.75879839).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(12723.94688137).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.03593039).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.04871853).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(-0.08464892).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.04939135).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) ==
              Approx(2.0 * 0.26456775).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) ==
              Approx(2.0 * 0.41490895).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.03634077).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.01817038).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.01817038).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.11542144).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) ==
              Approx(2.0 * 0.23084287).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) ==
              Approx(2.0 * 0.34626431).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-4747.3896037574).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-4876.3051981213).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-4876.3051981213).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(272.964758501).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(545.929517002).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(818.8942755031).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0044054288).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<   13437784.677,  5737022.67807,  5825192.64494, -32652.9002539, -174907.224933, -274298.634505,
                     5781107.6615,  13477642.5071,  5741249.83138,  16326.4501269,  87453.6124665,  137149.317253,
                     5781107.6615,  5785334.81481,  13433557.5237,  16326.4501269,  87453.6124665,  137149.317253,
                   -75444.0852629, -102295.698238,  177739.783501,  3742445.39556, -555520.555711, -871196.315245,
                   -150888.170526, -204591.396476,  355479.567002, -207416.901178,  2735112.73473, -1742392.63049,
                   -226332.255789, -306887.094714,  533219.350503, -311125.351767, -1666561.66713,  1232564.90042;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }

  //! Check yield correction based on current stress
  SECTION("MohrCoulomb check yield correction based on current stress") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "MohrCoulomb3D", std::move(0), jmaterial);

    auto mohr_coulomb = std::make_shared<mpm::MohrCoulomb<Dim>>(id, jmaterial);

    REQUIRE(material->id() == 0);

    // Assign particle mass and volume
    particle->assign_volume(1.0);
    particle->assign_material(mohr_coulomb, 0);
    particle->compute_mass();

    // Calculate modulus values
    const double K =
        material->template property<double>("youngs_modulus") /
        (3.0 *
         (1. - 2. * material->template property<double>("poisson_ratio")));
    const double G =
        material->template property<double>("youngs_modulus") /
        (2.0 * (1. + material->template property<double>("poisson_ratio")));
    const double a1 = K + (4.0 / 3.0) * G;
    const double a2 = K - (2.0 / 3.0) * G;
    // Compute elastic tensor
    mpm::Material<Dim>::Matrix6x6 de;
    de.setZero();
    de(0, 0) = a1;
    de(0, 1) = a2;
    de(0, 2) = a2;
    de(1, 0) = a2;
    de(1, 1) = a1;
    de(1, 2) = a2;
    de(2, 0) = a2;
    de(2, 1) = a2;
    de(2, 2) = a1;
    de(3, 3) = G;
    de(4, 4) = G;
    de(5, 5) = G;

    // Initialise state variables
    mpm::dense_map state_variables = material->initialise_state_variables();

    //! Check for shear failure ( pdstrain_peak < pdstrain <  pdstrain_residual)
    SECTION(
        "Check for shear failure ( pdstrain_peak < pdstrain <  "
        "pdstrain_residual)") {

      // Tolerance for computation of stress
      const double Tolerance_stress = 1.E-5;

      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -6000.;
      stress(3) = -3674.5;
      stress(4) = -1000.;
      stress(5) = -2000.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-9814.95457622).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(6137.6353074).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.62535818).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Define current plastic strain
      state_variables.at("pdstrain") = 0.00011976829482;
      // Modified MC parameters
      state_variables.at("phi") = 0.46088824307428;
      state_variables.at("psi") = 0.23044412153714;
      state_variables.at("cohesion") = 1880.23170517852;
      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-1603.69045498).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(0.05571368).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.36180859).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(0.15867151).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.02392456).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.49615858).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(2.0 * 0.01495318).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(2.0 * -0.22036225).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.26805725).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(0.07640756).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.10985293).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.44892570).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(2.0 * 0.03081729).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(2.0 * -0.19365653).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0005;
      dstrain(1) = -0.0005;
      dstrain(2) = 0.;
      dstrain(3) = 0.0001;
      dstrain(4) = 0.0002;
      dstrain(5) = 0.0003;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.46088824307428).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.23044412153714).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1880.23170517852).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-9814.95457622).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(7818.56228978).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.46506728).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00011976829482).epsilon(Tolerance));

      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(39.14507588).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(1560.72384677).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.68454836).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(-0.19666374).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(0.00867092).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * -0.33068771).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) ==
              Approx(2.0 * 0.00151934).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) ==
              Approx(2.0 * -0.10123189).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.58476919).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.21114283).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.13901448).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * -0.29693668).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) ==
              Approx(2.0 * 0.01666924).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) ==
              Approx(2.0 * -0.10091892).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) ==
              Approx(-3505.5428972498).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(1) ==
              Approx(-9698.6386632491).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(2) ==
              Approx(-6103.8408725349).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(3) ==
              Approx(-2407.3321062492).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(4) ==
              Approx(-282.7379093982).epsilon(Tolerance_stress));
      REQUIRE(updated_stress(5) ==
              Approx(-549.7637416201).epsilon(Tolerance_stress));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0003546242).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check << -470722.393674,  3400015.11875,   721667.28512,  4429501.17951, -29647.1374233,  1370407.12606,
                    6728114.42943,  13624599.0156,  6116627.80687, -304859.085542,  2040.45531032, -94317.8580023,
                    4340407.52174,  5526255.81815,  12943885.5031,  454267.568334, -3040.46267956,  140542.125976,
                    5687250.90526,   967131.17799,  2060452.37688,  2037999.02801,  12102.1785984, -559410.224172,
                   -184915.161901, -31445.2837296, -66993.5072684,  58790.3095026,  3845760.35616,  18188.6528122,
                    1855361.87567,  315508.906906,  672185.006495, -589877.529721,  3948.11503099,  3663656.46384;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) ==
                  Approx(dep_check(i, j)).epsilon(Tolerance_stress));
    }

    //! Check for shear failure (pdstrain <  pdstrain_peak)
    SECTION("Check for shear failure (pdstrain <  pdstrain_peak)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -1000.;
      stress(1) = -6000.;
      stress(2) = -9293;
      stress(3) = -600.;
      stress(4) = -700.;
      stress(5) = -800.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-9406.76793591).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(6152.44390466).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.43146734).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-867.93424523).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(-0.02983913).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.84706337).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(-0.00340207).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.26631103).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.09585142).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(2.0 * -0.05137356).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(2.0 * -0.10302998).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.72719051).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.16927182).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.28996950).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.09776988).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(2.0 * -0.01852365).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(2.0 * -0.09120985).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.001;
      dstrain(4) = 0.0002;
      dstrain(5) = 0.0003;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-7963.39226293).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(7963.60445905).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.16536572).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(1815.88679617).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(2093.14192205).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.71936796).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(0.14481314).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(-0.28683083).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.32351289).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) ==
              Approx(2.0 * 0.00413022).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) ==
              Approx(2.0 * 0.04028029).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.59469771).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.05840592).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.26834260).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.36711838).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) ==
              Approx(2.0 * 0.01217858).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) ==
              Approx(2.0 * 0.03214063).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-1586.3024514519).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5772.7074804481).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-8553.2117176438).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(2321.1100892241).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(38.1040241342).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(267.5707659057).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0002112522).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<   1902632.7949,  9623.80865723,  4307338.05814, -3216270.16175, -41559.2128541, -391530.119492,
                    3659722.93144,  12410406.4274,  5502434.39412,  -586971.40632, -7584.58350441, -71454.5026683,
                    6830126.07235,  6297856.93407,  13595713.3701,  295194.545796,  3814.37265687,  35935.2759489,
                   -5232804.25658, -2607417.75119, -661809.917123,  2390123.95214, -18814.1708389,  -177248.65441,
                     -164177.1079, -81806.6727681, -20764.0173118, -45682.3464617,  3845563.55919, -5561.10453082,
                   -458597.556979, -228511.396966, -58000.3371606, -127604.955115, -1648.85448801,  3830619.95727;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }

    //! Check for shear failure (pdstrain >  pdstrain_residual)
    SECTION("Check for shear failure (pdstrain >  pdstrain_residual)") {
      // Initialise stress
      mpm::Material<Dim>::Vector6d stress;
      stress.setZero();
      stress(0) = -5000.;
      stress(1) = -6000.;
      stress(2) = -6853.;
      stress(3) = -100.;
      stress(4) = -200.;
      stress(5) = -300.;

      // Check if stress invariants is computed correctly based on stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") ==
              Approx(0.52359878).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") ==
              Approx(0.26179939).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(jmaterial["cohesion"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-10307.43435584).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(1414.35709777).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.51890420).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      // Define current plastic strain
      state_variables.at("pdstrain") = 0.00244948974278;
      // Modified MC parameters
      state_variables.at("phi") = 0.;
      state_variables.at("psi") = 0.;
      state_variables.at("cohesion") = 1000.;

      // Initialise values of yield functions
      Eigen::Matrix<double, 2, 1> yield_function;
      auto yield_type =
          mohr_coulomb->compute_yield_state(&yield_function, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function(0) == Approx(-4948.19884736).epsilon(Tolerance));
      REQUIRE(yield_function(1) == Approx(0.09047424).epsilon(Tolerance));
      REQUIRE(yield_type == mpm::mohrcoulomb::FailureState::Shear);

      // Initialise plastic correction components
      mpm::Material<Dim>::Vector6d df_dsigma, dp_dsigma;
      double dp_dq = 0.;
      df_dsigma.setZero();
      dp_dsigma.setZero();
      double softening = 0.;
      // Compute plastic correction components
      mohr_coulomb->compute_df_dp(yield_type, &state_variables, stress,
                                  &df_dsigma, &dp_dsigma, &dp_dq, &softening);
      // Check plastic correction component based on stress
      // Check dF/dSigma
      REQUIRE(df_dsigma(0) == Approx(0.47410554).epsilon(Tolerance));
      REQUIRE(df_dsigma(1) == Approx(-0.02200121).epsilon(Tolerance));
      REQUIRE(df_dsigma(2) == Approx(-0.45210433).epsilon(Tolerance));
      REQUIRE(df_dsigma(3) == Approx(2.0 * -0.04987491).epsilon(Tolerance));
      REQUIRE(df_dsigma(4) == Approx(2.0 * -0.10089051).epsilon(Tolerance));
      REQUIRE(df_dsigma(5) == Approx(2.0 * -0.15001459).epsilon(Tolerance));
      // Check dP/dSigma
      REQUIRE(dp_dsigma(0) == Approx(0.41175329).epsilon(Tolerance));
      REQUIRE(dp_dsigma(1) == Approx(-0.02121547).epsilon(Tolerance));
      REQUIRE(dp_dsigma(2) == Approx(-0.39053782).epsilon(Tolerance));
      REQUIRE(dp_dsigma(3) == Approx(2.0 * -0.04329688).epsilon(Tolerance));
      REQUIRE(dp_dsigma(4) == Approx(2.0 * -0.08659375).epsilon(Tolerance));
      REQUIRE(dp_dsigma(5) == Approx(2.0 * -0.12989063).epsilon(Tolerance));

      // Initialise incremental of strain
      mpm::Material<Dim>::Vector6d dstrain;
      dstrain.setZero();
      dstrain(0) = 0.0001;
      dstrain(1) = 0.;
      dstrain(2) = 0.;
      dstrain(3) = 0.0001;
      dstrain(4) = 0.0002;
      dstrain(5) = 0.0003;
      // Compute trial stress
      mpm::Material<Dim>::Vector6d trial_stress = stress + de * dstrain;
      // Check if stress invariants is computed correctly based on trial stress
      REQUIRE(mohr_coulomb->compute_stress_invariants(
                  trial_stress, &state_variables) == true);
      REQUIRE(state_variables.at("phi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("psi") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("cohesion") ==
              Approx(1000.).epsilon(Tolerance));
      REQUIRE(state_variables.at("epsilon") ==
              Approx(-8864.05868287).epsilon(Tolerance));
      REQUIRE(state_variables.at("rho") ==
              Approx(2417.87632461).epsilon(Tolerance));
      REQUIRE(state_variables.at("theta") ==
              Approx(0.41113704).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.00244948974278).epsilon(Tolerance));
      // Initialise values of yield functions based on trial stress
      Eigen::Matrix<double, 2, 1> yield_function_trial;
      auto yield_type_trial = mohr_coulomb->compute_yield_state(
          &yield_function_trial, state_variables);
      // Check if yield function and yield state is computed correctly
      REQUIRE(yield_function_trial(0) ==
              Approx(-3307.99391393).epsilon(Tolerance));
      REQUIRE(yield_function_trial(1) ==
              Approx(698.89632011).epsilon(Tolerance));
      REQUIRE(yield_type_trial == mpm::mohrcoulomb::FailureState::Shear);
      // Initialise plastic correction components based on trial stress
      mpm::Material<Dim>::Vector6d df_dsigma_trial, dp_dsigma_trial;
      double dp_dq_trial = 0.;
      df_dsigma_trial.setZero();
      dp_dsigma_trial.setZero();
      double softening_trial = 0.;
      // Compute plastic correction components based on trial stress
      mohr_coulomb->compute_df_dp(
          yield_type_trial, &state_variables, trial_stress, &df_dsigma_trial,
          &dp_dsigma_trial, &dp_dq_trial, &softening_trial);

      // Check plastic correction component based on trial stress
      // Check dFtrial/dSigma
      REQUIRE(df_dsigma_trial(0) == Approx(0.40686166).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(1) == Approx(-0.04116315).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(2) == Approx(-0.36569852).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(3) ==
              Approx(2.0 * 0.05724361).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(4) ==
              Approx(2.0 * 0.19277061).epsilon(Tolerance));
      REQUIRE(df_dsigma_trial(5) ==
              Approx(2.0 * 0.24306285).epsilon(Tolerance));
      // Check dPtrial/dSigma
      REQUIRE(dp_dsigma_trial(0) == Approx(0.37073994).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(1) == Approx(-0.07735086).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(2) == Approx(-0.29338908).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(3) ==
              Approx(2.0 * 0.07208417).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(4) ==
              Approx(2.0 * 0.14416835).epsilon(Tolerance));
      REQUIRE(dp_dsigma_trial(5) ==
              Approx(2.0 * 0.21625252).epsilon(Tolerance));

      // Initialise elastic state
      material->initialise(&state_variables);

      // Check compute stress
      mpm::Material<Dim>::Vector6d updated_stress =
          mohr_coulomb->compute_stress(stress, dstrain, particle.get(),
                                       &state_variables, dt);
      // Check update stress
      REQUIRE(updated_stress(0) == Approx(-4206.8535416451).epsilon(Tolerance));
      REQUIRE(updated_stress(1) == Approx(-5293.085016198).epsilon(Tolerance));
      REQUIRE(updated_stress(2) == Approx(-5853.0614421569).epsilon(Tolerance));
      REQUIRE(updated_stress(3) == Approx(149.2284505527).epsilon(Tolerance));
      REQUIRE(updated_stress(4) == Approx(298.4569011055).epsilon(Tolerance));
      REQUIRE(updated_stress(5) == Approx(447.6853516582).epsilon(Tolerance));

      // Check plastic strain
      REQUIRE(state_variables.at("pdstrain") ==
              Approx(0.0025425174).epsilon(Tolerance));

      // Compute consistent tangent matrix
      auto dep = material->compute_consistent_tangent_matrix(
          updated_stress, stress, dstrain, particle.get(), &state_variables,
          dt);

      // Values of reduced constitutive relations matrix
      Eigen::Matrix<double, 6, 6> dep_check;
      // clang-format off
      dep_check <<  10453238.0876,   6026482.6126,  8520279.29976, -365788.816249, -1215892.94017, -1522348.73607,
                    6348615.35646,  13411992.9592,   5239391.6843,  70449.2158279,  234175.295582,  293197.249091,
                    8198146.55589,  5561524.42816,  11240329.0159,  295339.600422,  981717.644591,  1229151.48698,
                    -492882.66853,  42148.3759213,  450734.292609,  3786222.67391, -199213.004858, -249423.001101,
                   -985765.337061,  84296.7518425,  901468.585218, -119862.344489,  3447727.83644, -498846.002201,
                   -1478648.00559,  126445.127764,  1352202.87783, -179793.516734, -597639.014575,  3097884.84285;
      // clang-format on
      // Check cell stiffness matrix
      for (unsigned i = 0; i < dep.rows(); ++i)
        for (unsigned j = 0; j < dep.cols(); ++j)
          REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
    }
  }
}
