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

  SECTION("HenckyHyperElastic check simple shear stresses") {
    unsigned id = 0;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 210.E9;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
    mpm::dense_map state_vars = material->initialise_state_variables();
    const double G = 8.076923076923077e+10;

    // Shear displacement
    double shear_disp = 10.0;
    int step = 20;
    double shear_inc = shear_disp / double(step);

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();

    // Assign deformation gradient increment
    Eigen::Matrix<double, 3, 3> F_inc = Eigen::Matrix<double, 3, 3>::Identity();
    F_inc(0, 1) = shear_inc;

    // Expected solution for step 20
    Eigen::VectorXd exp_normalized_sigma_xy(step);
    exp_normalized_sigma_xy << 0.48015546341516, 0.86081788192801,
        1.10903548889591, 1.24645048028046, 1.30885233370922, 1.32547078214369,
        1.31565902088005, 1.29122682289201, 1.25915754948669, 1.22353257912461,
        1.18672164153249, 1.15008652284837, 1.11439252642938, 1.08004989020512,
        1.04725676503674, 1.01608483384284, 0.98653112251039, 0.95854961982105,
        0.93207067967821, 0.90701294047153;

    for (unsigned i = 0; i < step; ++i) {
      particle->assign_deformation_gradient_increment(F_inc);
      Eigen::Matrix3d F = particle->deformation_gradient();

      // Compute updated stress
      stress = material->compute_stress(stress, F, F_inc, particle.get(),
                                        &state_vars);

      // Check stresses
      REQUIRE((stress(3) / G) ==
              Approx(exp_normalized_sigma_xy(i)).epsilon(Tolerance));

      // Update deformation gradient
      particle->update_deformation_gradient();
    }

    // Now increase number of step
    step = 1000;
    shear_inc = shear_disp / double(step);

    // Initialise stress and deformation gradient
    stress.setZero();
    F_inc.setIdentity();
    F_inc(0, 1) = shear_inc;
    particle->assign_deformation_gradient(
        Eigen::Matrix<double, 3, 3>::Identity());

    // Expected solution for step 1000: only check every 10 indices
    Eigen::VectorXd exp_normalized_sigma_xy_fine(step / 10);
    exp_normalized_sigma_xy_fine << 0.009999833336667, 0.109778702115125,
        0.208469986269071, 0.305128339720869, 0.398885943525585,
        0.488980981808193, 0.574777789109029, 0.655777918655055,
        0.731622443315117, 0.802086648310069, 0.867068811041632,
        0.926574974349042, 0.980701554565920, 1.029617369221540,
        1.073546312070880, 1.112751523010190, 1.147521552022290,
        1.178158730195470, 1.204969747751010, 1.228258295880370,
        1.248319545370720, 1.265436196901100, 1.279875832115930,
        1.291889309589100, 1.301709976544670, 1.309553499162390,
        1.315618147089720, 1.320085398725520, 1.323120761491620,
        1.324874725072750, 1.325483785426830, 1.325071493504310,
        1.323749495484450, 1.321618541412270, 1.318769446872900,
        1.315283998205060, 1.311235796110890, 1.306691035689540,
        1.301709223172760, 1.296343831188790, 1.290642895398830,
        1.284649555974340, 1.278402547718490, 1.271936642763120,
        1.265283049755330, 1.258469773331070, 1.251521937492930,
        1.244462076289450, 1.237310394953910, 1.230085004412960,
        1.222802131830590, 1.215476309615310, 1.208120545093100,
        1.200746472837230, 1.193364491450340, 1.185983886413840,
        1.178612940454970, 1.171259032732430, 1.163928728005770,
        1.156627856831570, 1.149361587719420, 1.142134492082070,
        1.134950602725370, 1.127813466544750, 1.120726192023810,
        1.113691492067620, 1.106711722646790, 1.099788917677870,
        1.092924820520920, 1.086120912434760, 1.079378438294940,
        1.072698429847360, 1.066081726742180, 1.059528995567220,
        1.053040747077530, 1.046617351797410, 1.040259054153290,
        1.033965985279750, 1.027738174626340, 1.021575560480400,
        1.015477999509030, 1.009445275413530, 1.003477106780060,
        0.997573154202077, 0.991733026742921, 0.985956287799803,
        0.980242460424916, 0.974591032153762, 0.969001459386063,
        0.963473171360289, 0.958005573758851, 0.952598051977662,
        0.947249974090389, 0.941960693535062, 0.936729551548041,
        0.931555879367977, 0.926439000230461, 0.921378231171984,
        0.916372884660250, 0.911422270066273;

    for (unsigned i = 0; i < step; ++i) {
      particle->assign_deformation_gradient_increment(F_inc);
      Eigen::Matrix3d F = particle->deformation_gradient();

      // Compute updated stress
      stress = material->compute_stress(stress, F, F_inc, particle.get(),
                                        &state_vars);

      // Check stresses
      if (i % 10 == 0)
        REQUIRE(
            (stress(3) / G) ==
            Approx(exp_normalized_sigma_xy_fine(i / 10)).epsilon(Tolerance));

      // Update deformation gradient
      particle->update_deformation_gradient();
    }
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
