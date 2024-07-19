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

//! \brief Check Bingham class
//! Check Bingham 2D without thixotropy
TEST_CASE("Bingham is checked in 2D (without thixotropy)",
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
  jmaterial["regularization_parameter"] = 10.0;
  jmaterial["flocculation_state"] = 0;
  jmaterial["flocculation_parameter"] = 0;
  jmaterial["deflocculation_rate"] = 0;

  //! Check for id = 0
  SECTION("Bingham id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
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
    REQUIRE(material->template property<double>("regularization_parameter") ==
            Approx(jmaterial["regularization_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 4);
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("gamma_dot") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "pressure", "volumetric_strain", "lambda", "gamma_dot"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
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
    REQUIRE(state_vars.at("pressure") ==
            Approx(53239.4547055938).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0062500000).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_vars.at("gamma_dot") ==
            Approx(0.0054932487).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-53239.7900274234).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-53240.7959929121).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(1.6766091478).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(-2.5149137217).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));
  }
}

//! Check Bingham 2D with thixotropy
TEST_CASE("Bingham is checked in 2D (with thixotropy)",
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
  jmaterial["regularization_parameter"] = 10.0;
  jmaterial["flocculation_state"] = 1.0;
  jmaterial["flocculation_parameter"] = 1.0;
  jmaterial["deflocculation_rate"] = 0.01;

  //! Check for id = 0
  SECTION("Bingham id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
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
    REQUIRE(material->template property<double>("regularization_parameter") ==
            Approx(jmaterial["regularization_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 4);
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("gamma_dot") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "pressure", "volumetric_strain", "lambda", "gamma_dot"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham2D", std::move(id), jmaterial);
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
    REQUIRE(state_vars.at("pressure") ==
            Approx(53239.4547055938).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0062500000).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(1.0199450675).epsilon(Tolerance));
    REQUIRE(state_vars.at("gamma_dot") ==
            Approx(0.0054932487).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-53240.1001639861).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-53242.0365391630).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(3.2272919615).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(-4.8409379422).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));
  }
}

//! Check Bingham 3D without thixotropy
TEST_CASE("Bingham is checked in 3D (without thixotropy)",
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
  jmaterial["regularization_parameter"] = 10.0;
  jmaterial["flocculation_state"] = 0;
  jmaterial["flocculation_parameter"] = 0;
  jmaterial["deflocculation_rate"] = 0;

  //! Check for id = 0
  SECTION("Bingham id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
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
    REQUIRE(material->template property<double>("regularization_parameter") ==
            Approx(jmaterial["regularization_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 4);
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("gamma_dot") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "pressure", "volumetric_strain", "lambda", "gamma_dot"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham check stresses with strain rate") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
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
    REQUIRE(state_vars.at("pressure") ==
            Approx(15727.9891467669).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0018750000).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_vars.at("gamma_dot") ==
            Approx(0.0050513114).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-15728.3671338958).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-15725.9732154129).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-15729.6270909921).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(-0.5669806933).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(-0.1889935644).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(-1.8899356444).epsilon(Tolerance));
  }
}

//! Check Bingham 3D with thixotropy
TEST_CASE("Bingham is checked in 3D (with thixotropy)",
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
  jmaterial["regularization_parameter"] = 10.0;
  jmaterial["flocculation_state"] = 1.0;
  jmaterial["flocculation_parameter"] = 1.0;
  jmaterial["deflocculation_rate"] = 0.01;

  //! Check for id = 0
  SECTION("Bingham id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
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
    REQUIRE(material->template property<double>("regularization_parameter") ==
            Approx(jmaterial["regularization_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 4);
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("gamma_dot") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "pressure", "volumetric_strain", "lambda", "gamma_dot"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham check stresses with strain rate") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Bingham3D", std::move(id), jmaterial);
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
    REQUIRE(state_vars.at("pressure") ==
            Approx(15727.9891467669).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0018750000).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(1.0199494869).epsilon(Tolerance));
    REQUIRE(state_vars.at("gamma_dot") ==
            Approx(0.0050513114).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-15728.7168040748).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-15724.1083077915).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-15731.1423284344).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(-1.0914859618).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(-0.3638286539).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(-3.6382865394).epsilon(Tolerance));
  }
}