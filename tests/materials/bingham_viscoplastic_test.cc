#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "element.h"
#include "factory.h"
#include "hexahedron_element.h"
#include "material.h"
#include "mesh.h"
#include "node.h"

//! \brief Check Bingham Viscoplastic class
//! Check Bingham Viscoplastic 2D without thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 2D (without thixotropy)",
          "[material][bingham][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 0.;
  jmaterial["flocculation_parameter"] = 0.;
  jmaterial["deflocculation_rate"] = 0.;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 7);
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {"yield_state",
                                                   "pressure",
                                                   "volumetric_strain",
                                                   "shear_stress_ratio",
                                                   "lambda",
                                                   "pgamma_dot",
                                                   "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 2;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 4;
    const double dt = 1;

    coords << -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED2Q4");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0000000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(4305.6531945081).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0046875).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0085233869).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0015145055).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0008744002).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-4360.2902901182).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-4261.94351802).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-4294.7257753861).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8616049.09167563, 8620625.12983505, 8597244.94553788,                0,                0,                0,
                  8620625.12983505, 8630077.20225394, 8583216.83495957,                0,                0,                0,
                  8597244.94553788, 8583216.83495957, 8653457.38655111,                0,                0,                0,
                                 0,                0,                0, 32782.2573660532,                0,                0,
                                 0,                0,                0,                0, 32782.2573660532,                0,
                                 0,                0,                0,                0,                0, 32782.2573660532;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}

//! Check Bingham Viscoplastic 2D with thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 2D (with thixotropy)",
          "[material][bingham][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 1.0;
  jmaterial["flocculation_parameter"] = 1.0;
  jmaterial["deflocculation_rate"] = 0.01;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 7);
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {"yield_state",
                                                   "pressure",
                                                   "volumetric_strain",
                                                   "shear_stress_ratio",
                                                   "lambda",
                                                   "pgamma_dot",
                                                   "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 2;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 4;
    const double dt = 1;

    coords << -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED2Q4");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0000000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(4305.6531945081).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0046875).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0172038514).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(1.0199846873).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0015012459).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0008667447).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-4415.9342934026).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-4217.4283153925).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-4283.5969747292).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<   8620817.9705534, 8630164.58752109, 8582936.60897407,                0,                0,                0,
                  8630164.58752109, 8649154.75768162, 8554599.82184585,                0,                0,                0,
                  8582936.60897407, 8554599.82184585, 8696382.73622865,                0,                0,                0,
                                 0,                0,                0, 66168.6593366952,                0,                0,
                                 0,                0,                0,                0, 66168.6593366952,                0,
                                 0,                0,                0,                0,                0, 66168.6593366952;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}

//! Check Bingham Viscoplastic 3D without thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 3D (without thixotropy)",
          "[material][bingham][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 0.;
  jmaterial["flocculation_parameter"] = 0.;
  jmaterial["deflocculation_rate"] = 0.;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {"yield_state",
                                                   "pressure",
                                                   "volumetric_strain",
                                                   "shear_stress_ratio",
                                                   "lambda",
                                                   "pgamma_dot",
                                                   "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 3;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 8;
    const double dt = 1;

    coords << -2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    coords << -2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    coords << 2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    coords << 2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    coords << -2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->assign_velocity_constraint(2, 0.04);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0004000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::Material<Dim>::Vector6d stress;
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(847.8131628497).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0024609375).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0077634533).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0016642778).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0009608712).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-905.5414049792).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-815.9630982265).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-821.9349853434).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8478339.57124586, 8480076.17288255,  8475979.1413622,                0,                0,                0,
                  8480076.17288255, 8505888.57629644, 8448430.13631162,                0,                0,                0,
                   8475979.1413622, 8448430.13631162, 8509985.60781679,                0,                0,                0,
                                 0,                0,                0, 29859.4355842309,                0,                0,
                                 0,                0,                0,                0, 29859.4355842309,                0,
                                 0,                0,                0,                0,                0, 29859.4355842309;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}

//! Check Bingham Viscoplastic 3D with thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 3D (with thixotropy)",
          "[material][bingham][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 1.0;
  jmaterial["flocculation_parameter"] = 1.0;
  jmaterial["deflocculation_rate"] = 0.01;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {"yield_state",
                                                   "pressure",
                                                   "volumetric_strain",
                                                   "shear_stress_ratio",
                                                   "lambda",
                                                   "pgamma_dot",
                                                   "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 3;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 8;
    const double dt = 1;

    coords << -2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    coords << -2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    coords << 2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    coords << 2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    coords << -2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->assign_velocity_constraint(2, 0.04);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0004000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::Material<Dim>::Vector6d stress;
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(847.8131628497).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0024609375).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0156687852).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(1.0199831596).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0016510182).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0009532158).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-964.3246427223).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-783.5309670579).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-795.5838787689).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8478482.9938005, 8482093.94822968, 8473817.94346043,                0,                0,                0,
                 8482093.94822968, 8534131.99138683, 8418168.94587409,                0,                0,                0,
                 8473817.94346043, 8418168.94587409, 8542407.99615608,                0,                0,                0,
                                0,                0,                0, 60264.5585547855,                0,                0,
                                0,                0,                0,                0, 60264.5585547855,                0,
                                0,                0,                0,                0,                0, 60264.5585547855;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}