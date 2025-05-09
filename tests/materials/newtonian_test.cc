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

//! \brief Check Newtonian class
TEST_CASE("Newtonian is checked in 2D", "[material][newtonian][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["bulk_modulus"] = 8333333.333333333;
  jmaterial["dynamic_viscosity"] = 8.9E-4;

  //! Check for id = 0
  SECTION("Newtonian id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Newtonian id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Newtonian failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["dynamic_viscosity"] = 8.9E-4;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Newtonian check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 1);
      REQUIRE(state_variables.at("pressure") == 0.0);
      const std::vector<std::string> state_vars = {"pressure"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Newtonian check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
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
    dstrain(2) = 0.0000500;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto check_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check stresses
    REQUIRE(check_stress.size() == 6);
    REQUIRE(check_stress(0) == Approx(-52083.3333338896).epsilon(Tolerance));
    REQUIRE(check_stress(1) == Approx(-52083.3333355583).epsilon(Tolerance));
    REQUIRE(check_stress(2) == Approx(-52083.3333305521).epsilon(Tolerance));
    REQUIRE(check_stress(3) == Approx(-0.0000041719).epsilon(Tolerance));
    REQUIRE(check_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(check_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Calculate modulus values
    const double K = 8333333.333333333;
    // Calculate pressure
    const double volumetric_strain = -0.00625;
    REQUIRE(particle->dvolumetric_strain() ==
            Approx(volumetric_strain).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(-K * volumetric_strain).epsilon(Tolerance));
  }

  SECTION("Newtonian check stresses incompressible") {
    unsigned id = 0;
    jmaterial["incompressible"] = true;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
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
    dstrain(2) = 0.0000500;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto check_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check stresses
    REQUIRE(check_stress.size() == 6);
    REQUIRE(check_stress(0) == Approx(-0.0000033375).epsilon(Tolerance));
    REQUIRE(check_stress(1) == Approx(-0.00000500625).epsilon(Tolerance));
    REQUIRE(check_stress(2) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(check_stress(3) == Approx(-0.0000041719).epsilon(Tolerance));
    REQUIRE(check_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(check_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Check pressure value
    REQUIRE(state_vars.at("pressure") == Approx(0.000e+00).epsilon(Tolerance));
  }

  SECTION("Newtonian check stresses fail") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

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
                                            particle.get(), &state_vars, dt));

    REQUIRE_THROWS(material->compute_consistent_tangent_matrix(
        updated_stress, stress, deformation_gradient,
        deformation_gradient_increment, particle.get(), &state_vars, dt));
  }
}

TEST_CASE("Newtonian is checked in 3D", "[material][newtonian][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["bulk_modulus"] = 8333333.333333333;
  jmaterial["dynamic_viscosity"] = 8.9E-4;

  //! Check for id = 0
  SECTION("Newtonian id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Newtonian id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Newtonian failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["dynamic_viscosity"] = 8.9E-4;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Newtonian check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 1);
      REQUIRE(state_variables.at("pressure") == 0.0);
      const std::vector<std::string> state_vars = {"pressure"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Newtonian check stresses") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(id), jmaterial);
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
    dstrain(3) = 0.0000100;
    dstrain(4) = 0.0000200;
    dstrain(5) = 0.0000300;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto check_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check stresses
    REQUIRE(check_stress.size() == 6);
    REQUIRE(check_stress(0) == Approx(-15625.0000006258).epsilon(Tolerance));
    REQUIRE(check_stress(1) == Approx(-15624.9999966625).epsilon(Tolerance));
    REQUIRE(check_stress(2) == Approx(-15625.0000027117).epsilon(Tolerance));
    REQUIRE(check_stress(3) == Approx(-0.0000009387).epsilon(Tolerance));
    REQUIRE(check_stress(4) == Approx(-0.0000003129).epsilon(Tolerance));
    REQUIRE(check_stress(5) == Approx(-0.0000031289).epsilon(Tolerance));

    // Calculate modulus values
    const double K = 8333333.333333333;
    // Calculate pressure
    const double volumetric_strain = -0.001875;
    REQUIRE(particle->dvolumetric_strain() ==
            Approx(volumetric_strain).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(-K * volumetric_strain).epsilon(Tolerance));
  }

  SECTION("Newtonian check stresses incompressible") {
    unsigned id = 0;
    jmaterial["incompressible"] = true;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(id), jmaterial);
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
    dstrain(3) = 0.0000100;
    dstrain(4) = 0.0000200;
    dstrain(5) = 0.0000300;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto check_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check stresses
    REQUIRE(check_stress.size() == 6);
    REQUIRE(check_stress(0) == Approx(-0.0000020859).epsilon(Tolerance));
    REQUIRE(check_stress(1) == Approx(0.0000018773).epsilon(Tolerance));
    REQUIRE(check_stress(2) == Approx(-0.0000041719).epsilon(Tolerance));
    REQUIRE(check_stress(3) == Approx(-0.0000009387).epsilon(Tolerance));
    REQUIRE(check_stress(4) == Approx(-0.0000003129).epsilon(Tolerance));
    REQUIRE(check_stress(5) == Approx(-0.0000031289).epsilon(Tolerance));

    // Check pressure value
    REQUIRE(state_vars.at("pressure") == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent false
    REQUIRE_THROWS(material->compute_consistent_tangent_matrix(
        stress, stress, dstrain, particle.get(), &state_vars, dt));
  }
}
