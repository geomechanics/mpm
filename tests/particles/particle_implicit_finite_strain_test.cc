#include <limits>

#include "catch.hpp"

#include "cell.h"
#include "element.h"
#include "function_base.h"
#include "hexahedron_element.h"
#include "linear_function.h"
#include "material.h"
#include "node.h"
#include "particle_bbar.h"
#include "particle_finite_strain.h"
#include "pod_particle.h"
#include "quadrilateral_element.h"

//! \brief Check ParticleFiniteStrain class for 2D case
TEST_CASE("Implicit ParticleFiniteStrain is checked for 2D case",
          "[particle][2D][Implicit][FiniteStrain]") {
  // Dimension
  const unsigned Dim = 2;
  // Degree of freedom
  const unsigned Dof = 2;
  // Number of nodes per cell
  const unsigned Nnodes = 4;
  // Number of phases
  const unsigned Nphases = 1;
  // Phase
  const unsigned phase = 0;
  // Tolerance
  const double Tolerance = 1.E-7;

  // Coordinates
  Eigen::Vector2d coords;
  coords.setZero();

  //! Test particle, cell and node functions
  SECTION("Test particle, cell and node functions") {
    // Add particle
    mpm::Index id = 0;
    coords << 0.75, 0.75;
    auto particle =
        std::make_shared<mpm::ParticleFiniteStrain<Dim>>(id, coords);

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Shape function
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::QuadrilateralElement<Dim, 4>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Add nodes to cell
    coords << 0.5, 0.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);

    coords << 1.5, 0.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);

    coords << 1.5, 1.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);

    coords << 0.5, 1.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    REQUIRE(cell->nnodes() == 4);

    std::vector<std::shared_ptr<mpm::NodeBase<Dim>>> nodes;
    nodes.emplace_back(node0);
    nodes.emplace_back(node1);
    nodes.emplace_back(node2);
    nodes.emplace_back(node3);

    // Assign active id
    unsigned int active_id = 0;
    for (const auto& node : nodes) {
      node->initialise_implicit();
      node->assign_active_id(active_id);
      active_id++;
    }

    // Initialise cell properties
    cell->initialise();

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Check compute shape functions of a particle
    // TODO Assert: REQUIRE_NOTHROW(particle->compute_shapefn());
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);
    // Compute updated particle location should fail
    // TODO Assert:
    // REQUIRE_NOTHROW(particle->compute_updated_position(dt) == false);
    // Compute updated particle location from nodal velocity should fail
    // TODO Assert: REQUIRE_NOTHROW(particle->compute_updated_position(dt,
    // true)); Compute volume
    // TODO Assert: REQUIRE(particle->compute_volume() == false);
    // Update volume should fail
    // TODO Assert: REQUIRE(particle->update_volume() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());
    // Check volume
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));

    // Check reference location
    coords << -0.5, -0.5;
    REQUIRE(particle->compute_reference_location() == true);
    auto ref_coordinates = particle->reference_location();
    for (unsigned i = 0; i < ref_coordinates.size(); ++i)
      REQUIRE(ref_coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Assign material
    unsigned mid = 1;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic2D", std::move(mid), jmaterial);

    // Check compute mass before material and volume
    // TODO Assert: REQUIRE(particle->compute_mass() == false);

    // Test compute stress before material assignment
    // TODO Assert: REQUIRE(particle->compute_stress() == false);

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);

    // Check material id
    REQUIRE(particle->material_id() == 1);

    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());
    // Mass
    REQUIRE(particle->mass() == Approx(1000.).epsilon(Tolerance));

    // Map particle mass to nodes
    particle->assign_mass(std::numeric_limits<double>::max());
    // TODO Assert: REQUIRE_NOTHROW(particle->map_mass_momentum_to_nodes());

    // Map particle pressure to nodes
    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);

    // Assign mass to nodes
    REQUIRE(particle->compute_reference_location() == true);
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Check velocity and acceleration
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = i;
    REQUIRE(particle->assign_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(i).epsilon(Tolerance));

    Eigen::VectorXd acceleration;
    acceleration.resize(Dim);
    for (unsigned i = 0; i < acceleration.size(); ++i) acceleration(i) = i;
    REQUIRE(particle->assign_acceleration(acceleration) == true);
    for (unsigned i = 0; i < acceleration.size(); ++i)
      REQUIRE(particle->acceleration()(i) == Approx(i).epsilon(Tolerance));

    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE_NOTHROW(particle->map_mass_momentum_inertia_to_nodes());

    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);
    REQUIRE(particle->compute_pressure_smoothing() == false);

    // Values of nodal mass
    std::array<double, 4> nodal_mass{562.5, 187.5, 62.5, 187.5};
    // Check nodal mass
    for (unsigned i = 0; i < nodes.size(); ++i)
      REQUIRE(nodes.at(i)->mass(phase) ==
              Approx(nodal_mass.at(i)).epsilon(Tolerance));

    // Values of nodal momentum
    Eigen::Matrix<double, 4, 2> nodal_momentum;
    // clang-format off
    nodal_momentum << 0., 562.5,
                      0., 187.5,
                      0., 62.5,
                      0., 187.5;
    // clang-format on
    // Check nodal momentum
    for (unsigned i = 0; i < nodal_momentum.rows(); ++i)
      for (unsigned j = 0; j < nodal_momentum.cols(); ++j)
        REQUIRE(nodes.at(i)->momentum(phase)(j) ==
                Approx(nodal_momentum(i, j)).epsilon(Tolerance));

    // Values of nodal inertia
    Eigen::Matrix<double, 4, 2> nodal_inertia;
    // clang-format off
    nodal_inertia << 0., 562.5,
                     0., 187.5,
                     0., 62.5,
                     0., 187.5;
    // clang-format on
    // Check nodal inertia
    for (unsigned i = 0; i < nodal_inertia.rows(); ++i)
      for (unsigned j = 0; j < nodal_inertia.cols(); ++j)
        REQUIRE(nodes.at(i)->inertia(phase)(j) ==
                Approx(nodal_inertia(i, j)).epsilon(Tolerance));

    // Compute nodal velocity and acceleration
    for (const auto& node : nodes) node->compute_velocity_acceleration();

    // Values of nodal velocity
    Eigen::Matrix<double, 4, 2> nodal_velocity;
    // clang-format off
      nodal_velocity << 0., 1.,
                        0., 1.,
                        0., 1.,
                        0., 1.;
    // clang-format on
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j)
        REQUIRE(nodes.at(i)->velocity(phase)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));

    // Values of nodal acceleration
    Eigen::Matrix<double, 4, 2> nodal_acceleration;
    // clang-format off
      nodal_acceleration << 0., 1.,
                            0., 1.,
                            0., 1.,
                            0., 1.;
    // clang-format on
    // Check nodal acceleration
    for (unsigned i = 0; i < nodal_acceleration.rows(); ++i)
      for (unsigned j = 0; j < nodal_acceleration.cols(); ++j)
        REQUIRE(nodes.at(i)->acceleration(phase)(j) ==
                Approx(nodal_acceleration(i, j)).epsilon(Tolerance));

    // Set nodal displacement to get non-zero strain
    // nodal displacement
    // clang-format off
      Eigen::VectorXd nodal_displacement;
      nodal_displacement.resize(4 * 2);
      nodal_displacement << 0., 0.,
                            0., 0.,
                            0.1, 0.2,
                            0.3, 0.4;
    // clang-format on
    // Assign nodal displacement
    for (const auto& node : nodes)
      node->update_displacement_increment(nodal_displacement, phase, 4);
    // Check nodal displacement
    for (unsigned i = 0; i < Nnodes; ++i)
      for (unsigned j = 0; j < 2; ++j)
        REQUIRE(nodes.at(i)->displacement(phase)(j) ==
                Approx(nodal_displacement(Nnodes * j + i)).epsilon(Tolerance));

    // Check pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Compute strain increment
    particle->compute_strain_volume_newmark();

    // Compute stress
    REQUIRE_NOTHROW(particle->compute_stress_newmark());
    Eigen::Matrix<double, 6, 1> stress;
    // clang-format off
    stress << 1027068.9206402331,
              2405908.7918860661,
              1029893.3137578896,
              122021.22754387908,
                               0,
                               0;
    // clang-format on
    // Check stress
    for (unsigned i = 0; i < stress.rows(); ++i)
      REQUIRE(particle->stress()(i) == Approx(stress(i)).epsilon(Tolerance));

    // Update stress and strain
    particle->update_stress_strain();

    // Deformation gradient
    Eigen::Matrix<double, 3, 3> deformation_gradient;
    // clang-format off
    deformation_gradient <<   1.,   0., 0.,
                            0.05, 1.25, 0.,
                              0.,   0., 1.;
    // clang-format on

    // Check deformation gradient increment
    for (unsigned i = 0; i < deformation_gradient.rows(); ++i)
      for (unsigned j = 0; j < deformation_gradient.cols(); ++j)
        REQUIRE(particle->deformation_gradient()(i, j) ==
                Approx(deformation_gradient(i, j)).epsilon(Tolerance));

    // Check updated pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Check updated volume
    REQUIRE(particle->volume() == Approx(1.25).epsilon(Tolerance));
  }
}

//! \brief Check ParticleFiniteStrain class for 3D case
TEST_CASE("Implicit ParticleFiniteStrain is checked for 3D case",
          "[particle][3D][Implicit][FiniteStrain]") {
  // Dimension
  const unsigned Dim = 3;
  // Dimension
  const unsigned Dof = 6;
  // Number of nodes per cell
  const unsigned Nnodes = 8;
  // Number of phases
  const unsigned Nphases = 1;
  // Tolerance
  const double Tolerance = 1.E-7;

  // Coordinates
  Eigen::Vector3d coords;
  coords.setZero();

  //! Test particle, cell and node functions
  SECTION("Test particle, cell and node functions") {
    // Add particle
    mpm::Index id = 0;
    coords << 1.5, 1.5, 1.5;
    auto particle =
        std::make_shared<mpm::ParticleFiniteStrain<Dim>>(id, coords);

    // Phase
    const unsigned phase = 0;

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Assign hexahedron shape function
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::HexahedronElement<Dim, 8>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Add nodes
    coords << 0, 0, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);

    coords << 2, 0, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);

    coords << 2, 2, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);

    coords << 0, 2, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    coords << 0, 0, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);

    coords << 2, 0, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);

    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);

    coords << 0, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    std::vector<std::shared_ptr<mpm::NodeBase<Dim>>> nodes;
    nodes.emplace_back(node0);
    nodes.emplace_back(node1);
    nodes.emplace_back(node2);
    nodes.emplace_back(node3);
    nodes.emplace_back(node4);
    nodes.emplace_back(node5);
    nodes.emplace_back(node6);
    nodes.emplace_back(node7);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);
    REQUIRE(cell->nnodes() == 8);

    // Assign active id
    unsigned int active_id = 0;
    for (const auto& node : nodes) {
      node->initialise_implicit();
      node->assign_active_id(active_id);
      active_id++;
    }

    // Initialise cell properties
    cell->initialise();

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Check compute shape functions of a particle
    // TODO Assert: REQUIRE(particle->compute_shapefn() == false);
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);
    // Compute updated particle location should fail
    // TODO Assert: REQUIRE(particle->compute_updated_position(dt) == false);
    // Compute updated particle location from nodal velocity should fail
    // TODO Assert: REQUIRE_NOTHROW(particle->compute_updated_position(dt,
    // true));

    // Compute volume
    // TODO Assert: REQUIRE(particle->compute_volume() == false);
    // Update volume should fail
    // TODO Assert: REQUIRE(particle->update_volume() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());
    // Check volume
    REQUIRE(particle->volume() == Approx(8.0).epsilon(Tolerance));

    // Check reference location
    coords << 0.5, 0.5, 0.5;
    REQUIRE(particle->compute_reference_location() == true);
    auto ref_coordinates = particle->reference_location();
    for (unsigned i = 0; i < ref_coordinates.size(); ++i)
      REQUIRE(ref_coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Assign material
    unsigned mid = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "HenckyHyperElastic3D", std::move(mid), jmaterial);

    // Check compute mass before material and volume
    // TODO Assert: REQUIRE(particle->compute_mass() == false);

    // Test compute stress before material assignment
    // TODO Assert: REQUIRE(particle->compute_stress() == false);

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);

    // Check material id from particle
    REQUIRE(particle->material_id() == 0);

    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());
    // Mass
    REQUIRE(particle->mass() == Approx(8000.).epsilon(Tolerance));

    // Map particle mass to nodes
    particle->assign_mass(std::numeric_limits<double>::max());
    // TODO Assert: REQUIRE(particle->map_mass_momentum_to_nodes() == false);

    // Map particle pressure to nodes
    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);

    // Assign mass to nodes
    REQUIRE(particle->compute_reference_location() == true);
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Check velocity and acceleration
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = i;
    REQUIRE(particle->assign_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(i).epsilon(Tolerance));

    Eigen::VectorXd acceleration;
    acceleration.resize(Dim);
    for (unsigned i = 0; i < acceleration.size(); ++i) acceleration(i) = i;
    REQUIRE(particle->assign_acceleration(acceleration) == true);
    for (unsigned i = 0; i < acceleration.size(); ++i)
      REQUIRE(particle->acceleration()(i) == Approx(i).epsilon(Tolerance));

    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE_NOTHROW(particle->map_mass_momentum_inertia_to_nodes());

    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);
    REQUIRE(particle->compute_pressure_smoothing() == false);

    // Values of nodal mass
    std::array<double, 8> nodal_mass{125., 375.,  1125., 375.,
                                     375., 1125., 3375., 1125.};
    // Check nodal mass
    for (unsigned i = 0; i < nodes.size(); ++i)
      REQUIRE(nodes.at(i)->mass(phase) ==
              Approx(nodal_mass.at(i)).epsilon(Tolerance));

    // Compute nodal velocity and acceleration
    for (const auto& node : nodes) node->compute_velocity_acceleration();

    // Values of nodal momentum
    Eigen::Matrix<double, 8, 3> nodal_momentum;
    // clang-format off
    nodal_momentum << 0.,  125.,  250.,
                      0.,  375.,  750.,
                      0., 1125., 2250.,
                      0.,  375.,  750.,
                      0.,  375.,  750.,
                      0., 1125., 2250.,
                      0., 3375., 6750.,
                      0., 1125., 2250.;
    // clang-format on

    // Check nodal momentum
    for (unsigned i = 0; i < nodal_momentum.rows(); ++i)
      for (unsigned j = 0; j < nodal_momentum.cols(); ++j)
        REQUIRE(nodes.at(i)->momentum(phase)[j] ==
                Approx(nodal_momentum(i, j)).epsilon(Tolerance));

    // Values of nodal velocity
    Eigen::Matrix<double, 8, 3> nodal_velocity;
    // clang-format off
    nodal_velocity << 0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.;
    // clang-format on
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j)
        REQUIRE(nodes.at(i)->velocity(phase)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));

    // Values of nodal inertia
    Eigen::Matrix<double, 8, 3> nodal_inertia;
    // clang-format off
    nodal_inertia << 0.,  125.,  250.,
                      0.,  375.,  750.,
                      0., 1125., 2250.,
                      0.,  375.,  750.,
                      0.,  375.,  750.,
                      0., 1125., 2250.,
                      0., 3375., 6750.,
                      0., 1125., 2250.;
    // clang-format on

    // Check nodal inertia
    for (unsigned i = 0; i < nodal_inertia.rows(); ++i)
      for (unsigned j = 0; j < nodal_inertia.cols(); ++j)
        REQUIRE(nodes.at(i)->inertia(phase)[j] ==
                Approx(nodal_inertia(i, j)).epsilon(Tolerance));

    // Values of nodal acceleration
    Eigen::Matrix<double, 8, 3> nodal_acceleration;
    // clang-format off
    nodal_acceleration << 0., 1., 2.,
                          0., 1., 2.,
                          0., 1., 2.,
                          0., 1., 2.,
                          0., 1., 2.,
                          0., 1., 2.,
                          0., 1., 2.,
                          0., 1., 2.;
    // clang-format on
    // Check nodal acceleration
    for (unsigned i = 0; i < nodal_acceleration.rows(); ++i)
      for (unsigned j = 0; j < nodal_acceleration.cols(); ++j)
        REQUIRE(nodes.at(i)->acceleration(phase)(j) ==
                Approx(nodal_acceleration(i, j)).epsilon(Tolerance));

    // Set nodal displacement to get non-zero strain
    // nodal displacement
    // clang-format off
    Eigen::VectorXd nodal_displacement;
    nodal_displacement.resize(8 * 3);
    nodal_displacement << 0., 0., 0.,
                          0., 0., 0.,
                          0., 0.,
                          0.1, 0.2, 0.3,
                          0.4, 0.5, 0.6,
                          0.7, 0.8,
                          0.2, 0.4, 0.6,
                          0.8, 1.0, 1.2,
                          1.4, 1.6;
    // clang-format on
    // Compute nodal displacement
    for (const auto& node : nodes)
      node->update_displacement_increment(nodal_displacement, phase, 8);
    // Check nodal displacement
    for (unsigned i = 0; i < Nnodes; ++i)
      for (unsigned j = 0; j < 3; ++j)
        REQUIRE(nodes.at(i)->displacement(phase)(j) ==
                Approx(nodal_displacement(Nnodes * j + i)).epsilon(Tolerance));

    // Check pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Compute strain increment
    particle->compute_strain_volume_newmark();

    // Compute stress
    REQUIRE_NOTHROW(particle->compute_stress_newmark());
    Eigen::Matrix<double, 6, 1> stress;
    // clang-format off
    stress <<   1517827.6913974155,
                1855975.5651401044,
                3213620.2992554531,
               -44953.742519450374,
                761509.28187618684,
               -85213.534390811867;
    // clang-format on
    // Check stress
    for (unsigned i = 0; i < stress.rows(); ++i)
      REQUIRE(particle->stress()(i) == Approx(stress(i)).epsilon(Tolerance));

    // Update stress and strain
    particle->update_stress_strain();

    // Deformation gradient
    Eigen::Matrix<double, 3, 3> deformation_gradient;
    // clang-format off
    deformation_gradient <<     1.,    0.,   0.,
                            -0.025, 1.075,   .2,
                             -0.05,  0.15,  1.4;
    // clang-format on
    // Check deformation gradient increment
    for (unsigned i = 0; i < deformation_gradient.rows(); ++i)
      for (unsigned j = 0; j < deformation_gradient.cols(); ++j)
        REQUIRE(particle->deformation_gradient()(i, j) ==
                Approx(deformation_gradient(i, j)).epsilon(Tolerance));

    // Check updated pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Check updated volume
    REQUIRE(particle->volume() == Approx(11.8).epsilon(Tolerance));
  }
}
