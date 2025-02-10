#include <limits>
#include <memory>

#include "Eigen/Dense"
#include "catch.hpp"

#include "cell.h"
#include "element.h"
#include "factory.h"
#include "hexahedron_element.h"
#include "hexahedron_quadrature.h"
#include "node.h"
#include "particle_bbar.h"
#include "quadrilateral_element.h"
#include "quadrilateral_quadrature.h"

//! \brief Check cell class for 2D case with B-bar method
TEST_CASE("Implicit Cell is checked for 2D case with B-bar method",
          "[cell][2D][Implicit][Bbar]") {
  // Dimension
  const unsigned Dim = 2;
  // Degrees of freedom
  const unsigned Dof = 2;
  // Number of phases
  const unsigned Nphases = 1;
  // Number of nodes per cell
  const unsigned Nnodes = 4;
  // Tolerance
  const double Tolerance = 1.E-7;

  // Coordinates
  Eigen::Vector2d coords;

  // Check cell stiffness matrix calculation
  SECTION("Check cell stiffness matrix calculation") {
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

    // Initialise cell properties
    cell->initialise();

    // Add particle to cell
    mpm::Index id = 0;
    coords << 0.75, 0.75;
    auto particle = std::make_shared<mpm::ParticleBbar<Dim>>(id, coords);

    REQUIRE(cell->status() == false);
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

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
    jmaterial["youngs_modulus"] = 1.0;
    jmaterial["poisson_ratio"] = 0.25;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(mid), jmaterial);
    REQUIRE(material->id() == 1);

    mpm::dense_map state_variables = material->initialise_state_variables();

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);

    // Check material id
    REQUIRE(particle->material_id() == 1);

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());
    // Mass
    REQUIRE(particle->mass() == Approx(1000.).epsilon(Tolerance));

    // Initialize cell stiffness matrix
    REQUIRE(cell->initialise_element_stiffness_matrix() == true);
    Eigen::Matrix<double, 8, 8> stiffness_matrix = cell->stiffness_matrix();

    // Values of cell stiffness matrix
    Eigen::Matrix<double, 8, 8> cell_stiffness_matrix;
    // clang-format off
      cell_stiffness_matrix << 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0.;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < cell_stiffness_matrix.rows(); ++i)
      for (unsigned j = 0; j < cell_stiffness_matrix.cols(); ++j)
        REQUIRE(stiffness_matrix(i, j) ==
                Approx(cell_stiffness_matrix(i, j)).epsilon(Tolerance));

    // Constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dmatrix =
        material->compute_consistent_tangent_matrix(
            Eigen::Matrix<double, 6, 1>::Zero(),
            Eigen::Matrix<double, 6, 1>::Zero(),
            Eigen::Matrix<double, 6, 1>::Zero(), particle.get(),
            &state_variables);
    // Reduce constitutive relations matrix depending on the dimension
    Eigen::MatrixXd reduced_dmatrix;
    reduced_dmatrix = particle->reduce_dmatrix(dmatrix);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 3, 3> particle_reduced_dmatrix;
    // clang-format off
    particle_reduced_dmatrix << 1.2, 0.4, 0.0,
                                0.4, 1.2, 0.0,
                                0.0, 0.0, 0.4;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < particle_reduced_dmatrix.rows(); ++i)
      for (unsigned j = 0; j < particle_reduced_dmatrix.cols(); ++j)
        REQUIRE(reduced_dmatrix(i, j) ==
                Approx(particle_reduced_dmatrix(i, j)).epsilon(Tolerance));

    // Calculate B matrix
    Eigen::MatrixXd bmatrix;
    bmatrix = particle->compute_bmatrix();

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 3, 8> particle_bmatrix;
    // clang-format off
    particle_bmatrix << -0.625, 0.125,    0.625,  -0.125,    0.375, 0.125,   -0.375, -0.125,
                        0.125,    -0.625, -0.125,    -0.375, 0.125,   0.375, -0.125,    0.625,
                        -0.75, -0.75, -0.25, 0.75,  0.25, 0.25, 0.75,  -0.25;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < particle_bmatrix.rows(); ++i)
      for (unsigned j = 0; j < particle_bmatrix.cols(); ++j)
        REQUIRE(bmatrix(i, j) ==
                Approx(particle_bmatrix(i, j)).epsilon(Tolerance));

    // Compute local material stiffness matrix
    cell->compute_local_material_stiffness_matrix(bmatrix, reduced_dmatrix,
                                                  particle->volume());
    stiffness_matrix.setZero();
    stiffness_matrix = cell->stiffness_matrix();

    // Values of cell stiffness matrix
    // clang-format off
    cell_stiffness_matrix <<  0.65,   0.2,   -0.35,   -0.1,   -0.35,   -0.2,   0.05,   0.1,
                               0.2,  0.65,     0.1,   0.05,    -0.2,  -0.35,   -0.1, -0.35,
                             -0.35,   0.1,    0.45,   -0.2,    0.25,    0.1,  -0.35,   0.0,
                              -0.1,  0.05,    -0.2,   0.45,    -0.1,  -0.15,    0.4, -0.35,
                             -0.35,  -0.2,    0.25,   -0.1,    0.25,    0.2,  -0.15,   0.1,
                              -0.2, -0.35,     0.1,  -0.15,     0.2,   0.25,   -0.1,  0.25,
                              0.05,  -0.1,   -0.35,    0.4,   -0.15,   -0.1,   0.45,  -0.2,
                               0.1, -0.35,     0.0,  -0.35,     0.1,   0.25,   -0.2,  0.45;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < cell_stiffness_matrix.rows(); ++i)
      for (unsigned j = 0; j < cell_stiffness_matrix.cols(); ++j)
        REQUIRE(stiffness_matrix(i, j) ==
                Approx(cell_stiffness_matrix(i, j)).epsilon(Tolerance));
  }
}

//! \brief Check cell class for 3D case with B-bar method
TEST_CASE("Implicit Cell is checked for 3D case with B-bar method",
          "[cell][3D][Implicit][Bbar]") {
  // Dimension
  const unsigned Dim = 3;
  // Degrees of freedom
  const unsigned Dof = 6;
  // Number of phases
  const unsigned Nphases = 1;
  // Number of nodes per cell
  const unsigned Nnodes = 8;
  // Tolerance
  const double Tolerance = 1.E-7;

  // Coordinates
  Eigen::Vector3d coords;

  // Check cell stiffness matrix calculation
  SECTION("Check cell stiffness matrix calculation") {
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

    // Initialise cell properties
    cell->initialise();

    // Add particle to cell
    mpm::Index id = 0;
    coords << 1.5, 1.5, 1.5;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::ParticleBbar<Dim>>(id, coords);

    REQUIRE(cell->status() == false);
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

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
    jmaterial["youngs_modulus"] = 1.0;
    jmaterial["poisson_ratio"] = 0.25;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(mid), jmaterial);
    REQUIRE(material->id() == 0);

    mpm::dense_map state_variables = material->initialise_state_variables();

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);

    // Check material id
    REQUIRE(particle->material_id() == 0);

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());
    // Mass
    REQUIRE(particle->mass() == Approx(8000.).epsilon(Tolerance));

    // Initialize cell stiffness matrix
    REQUIRE(cell->initialise_element_stiffness_matrix() == true);
    Eigen::Matrix<double, 24, 24> stiffness_matrix = cell->stiffness_matrix();

    // Values of cell stiffness matrix
    Eigen::Matrix<double, 24, 24> cell_stiffness_matrix =
        Eigen::Matrix<double, 24, 24>::Zero();
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < cell_stiffness_matrix.rows(); ++i)
      for (unsigned j = 0; j < cell_stiffness_matrix.cols(); ++j)
        REQUIRE(stiffness_matrix(i, j) ==
                Approx(cell_stiffness_matrix(i, j)).epsilon(Tolerance));

    // Constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dmatrix =
        material->compute_consistent_tangent_matrix(
            Eigen::Matrix<double, 6, 1>::Zero(),
            Eigen::Matrix<double, 6, 1>::Zero(),
            Eigen::Matrix<double, 6, 1>::Zero(), particle.get(),
            &state_variables);
    // Reduce constitutive relations matrix depending on the dimension
    Eigen::MatrixXd reduced_dmatrix;
    reduced_dmatrix = particle->reduce_dmatrix(dmatrix);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> particle_reduced_dmatrix;
    // clang-format off
    particle_reduced_dmatrix << 1.2,       0.4,       0.4,       0.0,       0.0,       0.0,
                                0.4,       1.2,       0.4,       0.0,       0.0,       0.0,
                                0.4,       0.4,       1.2,       0.0,       0.0,       0.0,
                                0.0,       0.0,       0.0,       0.4,       0.0,       0.0,
                                0.0,       0.0,       0.0,       0.0,       0.4,       0.0,
                                0.0,       0.0,       0.0,       0.0,       0.0,       0.4;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < particle_reduced_dmatrix.rows(); ++i)
      for (unsigned j = 0; j < particle_reduced_dmatrix.cols(); ++j)
        REQUIRE(reduced_dmatrix(i, j) ==
                Approx(particle_reduced_dmatrix(i, j)).epsilon(Tolerance));

    // Calculate B matrix
    Eigen::MatrixXd bmatrix;
    bmatrix = particle->compute_bmatrix();

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 24> particle_bmatrix;
    // clang-format off
    particle_bmatrix << -0.0625,-0.03125,-0.03125,0.0625,-0.0104166667,-0.0104166667,0.1041666667,0.0104166667,0.0520833333,-0.1041666667,0.03125,-0.0104166667,-0.1041666667,-0.0104166667,0.03125,0.1041666667,0.0520833333,0.0104166667,0.2291666667,-0.0520833333,-0.0520833333,-0.2291666667,0.0104166667,0.0104166667,
                        -0.03125,-0.0625,-0.03125,0.03125,-0.1041666667,-0.0104166667,0.0104166667,0.1041666667,0.0520833333,-0.0104166667,0.0625,-0.0104166667,-0.0104166667,-0.1041666667,0.03125,0.0104166667,-0.2291666667,0.0104166667,-0.0520833333,0.2291666667,-0.0520833333,0.0520833333,0.1041666667,0.0104166667,
                        -0.03125,-0.03125,-0.0625,0.03125,-0.0104166667,-0.1041666667,0.0104166667,0.0104166667,-0.2291666667,-0.0104166667,0.03125,-0.1041666667,-0.0104166667,-0.0104166667,0.0625,0.0104166667,0.0520833333,0.1041666667,-0.0520833333,-0.0520833333,0.2291666667,0.0520833333,0.0104166667,0.1041666667,
                        -0.03125,-0.03125,0,-0.09375,0.03125,0,0.09375,0.09375,0,0.03125,-0.09375,0,-0.09375,-0.09375,0,-0.28125,0.09375,0,0.28125,0.28125,0,0.09375,-0.28125,0,
                               0,-0.03125,-0.03125,0,-0.09375,-0.09375,0,-0.28125,0.09375,0,-0.09375,0.03125,0,0.03125,-0.09375,0,0.09375,-0.28125,0,0.28125,0.28125,0,0.09375,0.09375,
                        -0.03125,0,-0.03125,-0.09375,0,0.03125,-0.28125,0,0.09375,-0.09375,0,-0.09375,0.03125,0,-0.09375,0.09375,0,0.09375,0.28125,0,0.28125,0.09375,0,-0.28125;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < particle_bmatrix.rows(); ++i)
      for (unsigned j = 0; j < particle_bmatrix.cols(); ++j)
        REQUIRE(bmatrix(i, j) ==
                Approx(particle_bmatrix(i, j)).epsilon(Tolerance));

    // Compute local material stiffness matrix
    cell->compute_local_material_stiffness_matrix(bmatrix, reduced_dmatrix,
                                                  particle->volume());
    stiffness_matrix.setZero();
    stiffness_matrix = cell->stiffness_matrix();

    // Values of cell stiffness matrix
    // clang-format off
    cell_stiffness_matrix <<  0.09375,0.084375,0.084375,-0.06875,0.0739583333,0.0739583333,-0.0770833333,-0.0864583333,0.0552083333,0.1020833333,-0.071875,0.0864583333,0.1020833333,0.0864583333,-0.071875,-0.0770833333,0.0552083333,-0.0864583333,-0.1770833333,-0.0927083333,-0.0927083333,0.1020833333,-0.0489583333,-0.0489583333,
                              0.084375,0.09375,0.084375,-0.071875,0.1020833333,0.0864583333,-0.0864583333,-0.0770833333,0.0552083333,0.0739583333,-0.06875,0.0739583333,0.0864583333,0.1020833333,-0.071875,-0.0489583333,0.1020833333,-0.0489583333,-0.0927083333,-0.1770833333,-0.0927083333,0.0552083333,-0.0770833333,-0.0864583333,
                              0.084375,0.084375,0.09375,-0.071875,0.0864583333,0.1020833333,-0.0489583333,-0.0489583333,0.1020833333,0.0864583333,-0.071875,0.1020833333,0.0739583333,0.0739583333,-0.06875,-0.0864583333,0.0552083333,-0.0770833333,-0.0927083333,-0.0927083333,-0.1770833333,0.0552083333,-0.0864583333,-0.0770833333,
                              -0.06875,-0.071875,-0.071875,0.14375,-0.0864583333,-0.0864583333,0.1520833333,0.0489583333,-0.0927083333,-0.0770833333,0.109375,-0.0489583333,-0.0770833333,-0.0489583333,0.109375,0.1520833333,-0.0927083333,0.0489583333,-0.0479166667,-0.0197916667,-0.0197916667,-0.1770833333,0.1614583333,0.1614583333,
                              0.0739583333,0.1020833333,0.0864583333,-0.0864583333,0.1520833333,0.0927083333,-0.0552083333,-0.0270833333,-0.0010416667,0.0677083333,-0.0770833333,0.0552083333,0.0552083333,0.1020833333,-0.0489583333,-0.0927083333,0.1770833333,0.0197916667,0.0010416667,-0.2520833333,-0.1114583333,0.0364583333,-0.1770833333,-0.0927083333,
                              0.0739583333,0.0864583333,0.1020833333,-0.0864583333,0.0927083333,0.1520833333,-0.0927083333,0.0197916667,0.1770833333,0.0552083333,-0.0489583333,0.1020833333,0.0677083333,0.0552083333,-0.0770833333,-0.0552083333,-0.0010416667,-0.0270833333,0.0010416667,-0.1114583333,-0.2520833333,0.0364583333,-0.0927083333,-0.1770833333,
                              -0.0770833333,-0.0864583333,-0.0489583333,0.1520833333,-0.0552083333,-0.0927083333,0.4020833333,0.0927083333,-0.1114583333,-0.0270833333,0.0489583333,0.0197916667,-0.1770833333,-0.0927083333,0.1614583333,-0.0479166667,0.0010416667,-0.0197916667,0.0270833333,0.1114583333,-0.2260416667,-0.2520833333,-0.0197916667,0.3177083333,
                              -0.0864583333,-0.0770833333,-0.0489583333,0.0489583333,-0.0270833333,0.0197916667,0.0927083333,0.4020833333,-0.1114583333,-0.0552083333,0.1520833333,-0.0927083333,-0.0927083333,-0.1770833333,0.1614583333,-0.0197916667,-0.2520833333,0.3177083333,0.1114583333,0.0270833333,-0.2260416667,0.0010416667,-0.0479166667,-0.0197916667,
                              0.0552083333,0.0552083333,0.1020833333,-0.0927083333,-0.0010416667,0.1770833333,-0.1114583333,-0.1114583333,0.4770833333,-0.0010416667,-0.0927083333,0.1770833333,0.0364583333,0.0364583333,-0.1770833333,0.0010416667,-0.0572916667,-0.2520833333,0.1697916667,0.1697916667,-0.2520833333,-0.0572916667,0.0010416667,-0.2520833333,
                              0.1020833333,0.0739583333,0.0864583333,-0.0770833333,0.0677083333,0.0552083333,-0.0270833333,-0.0552083333,-0.0010416667,0.1520833333,-0.0864583333,0.0927083333,0.1020833333,0.0552083333,-0.0489583333,-0.1770833333,0.0364583333,-0.0927083333,-0.2520833333,0.0010416667,-0.1114583333,0.1770833333,-0.0927083333,0.0197916667,
                              -0.071875,-0.06875,-0.071875,0.109375,-0.0770833333,-0.0489583333,0.0489583333,0.1520833333,-0.0927083333,-0.0864583333,0.14375,-0.0864583333,-0.0489583333,-0.0770833333,0.109375,0.1614583333,-0.1770833333,0.1614583333,-0.0197916667,-0.0479166667,-0.0197916667,-0.0927083333,0.1520833333,0.0489583333,
                              0.0864583333,0.0739583333,0.1020833333,-0.0489583333,0.0552083333,0.1020833333,0.0197916667,-0.0927083333,0.1770833333,0.0927083333,-0.0864583333,0.1520833333,0.0552083333,0.0677083333,-0.0770833333,-0.0927083333,0.0364583333,-0.1770833333,-0.1114583333,0.0010416667,-0.2520833333,-0.0010416667,-0.0552083333,-0.0270833333,
                              0.1020833333,0.0864583333,0.0739583333,-0.0770833333,0.0552083333,0.0677083333,-0.1770833333,-0.0927083333,0.0364583333,0.1020833333,-0.0489583333,0.0552083333,0.1520833333,0.0927083333,-0.0864583333,-0.0270833333,-0.0010416667,-0.0552083333,-0.2520833333,-0.1114583333,0.0010416667,0.1770833333,0.0197916667,-0.0927083333,
                              0.0864583333,0.1020833333,0.0739583333,-0.0489583333,0.1020833333,0.0552083333,-0.0927083333,-0.1770833333,0.0364583333,0.0552083333,-0.0770833333,0.0677083333,0.0927083333,0.1520833333,-0.0864583333,0.0197916667,0.1770833333,-0.0927083333,-0.1114583333,-0.2520833333,0.0010416667,-0.0010416667,-0.0270833333,-0.0552083333,
                              -0.071875,-0.071875,-0.06875,0.109375,-0.0489583333,-0.0770833333,0.1614583333,0.1614583333,-0.1770833333,-0.0489583333,0.109375,-0.0770833333,-0.0864583333,-0.0864583333,0.14375,0.0489583333,-0.0927083333,0.1520833333,-0.0197916667,-0.0197916667,-0.0479166667,-0.0927083333,0.0489583333,0.1520833333,
                              -0.0770833333,-0.0489583333,-0.0864583333,0.1520833333,-0.0927083333,-0.0552083333,-0.0479166667,-0.0197916667,0.0010416667,-0.1770833333,0.1614583333,-0.0927083333,-0.0270833333,0.0197916667,0.0489583333,0.4020833333,-0.1114583333,0.0927083333,0.0270833333,-0.2260416667,0.1114583333,-0.2520833333,0.3177083333,-0.0197916667,
                              0.0552083333,0.1020833333,0.0552083333,-0.0927083333,0.1770833333,-0.0010416667,0.0010416667,-0.2520833333,-0.0572916667,0.0364583333,-0.1770833333,0.0364583333,-0.0010416667,0.1770833333,-0.0927083333,-0.1114583333,0.4770833333,-0.1114583333,0.1697916667,-0.2520833333,0.1697916667,-0.0572916667,-0.2520833333,0.0010416667,
                              -0.0864583333,-0.0489583333,-0.0770833333,0.0489583333,0.0197916667,-0.0270833333,-0.0197916667,0.3177083333,-0.2520833333,-0.0927083333,0.1614583333,-0.1770833333,-0.0552083333,-0.0927083333,0.1520833333,0.0927083333,-0.1114583333,0.4020833333,0.1114583333,-0.2260416667,0.0270833333,0.0010416667,-0.0197916667,-0.0479166667,
                              -0.1770833333,-0.0927083333,-0.0927083333,-0.0479166667,0.0010416667,0.0010416667,0.0270833333,0.1114583333,0.1697916667,-0.2520833333,-0.0197916667,-0.1114583333,-0.2520833333,-0.1114583333,-0.0197916667,0.0270833333,0.1697916667,0.1114583333,0.9270833333,0.1677083333,0.1677083333,-0.2520833333,-0.2260416667,-0.2260416667,
                              -0.0927083333,-0.1770833333,-0.0927083333,-0.0197916667,-0.2520833333,-0.1114583333,0.1114583333,0.0270833333,0.1697916667,0.0010416667,-0.0479166667,0.0010416667,-0.1114583333,-0.2520833333,-0.0197916667,-0.2260416667,-0.2520833333,-0.2260416667,0.1677083333,0.9270833333,0.1677083333,0.1697916667,0.0270833333,0.1114583333,
                              -0.0927083333,-0.0927083333,-0.1770833333,-0.0197916667,-0.1114583333,-0.2520833333,-0.2260416667,-0.2260416667,-0.2520833333,-0.1114583333,-0.0197916667,-0.2520833333,0.0010416667,0.0010416667,-0.0479166667,0.1114583333,0.1697916667,0.0270833333,0.1677083333,0.1677083333,0.9270833333,0.1697916667,0.1114583333,0.0270833333,
                              0.1020833333,0.0552083333,0.0552083333,-0.1770833333,0.0364583333,0.0364583333,-0.2520833333,0.0010416667,-0.0572916667,0.1770833333,-0.0927083333,-0.0010416667,0.1770833333,-0.0010416667,-0.0927083333,-0.2520833333,-0.0572916667,0.0010416667,-0.2520833333,0.1697916667,0.1697916667,0.4770833333,-0.1114583333,-0.1114583333,
                             -0.0489583333,-0.0770833333,-0.0864583333,0.1614583333,-0.1770833333,-0.0927083333,-0.0197916667,-0.0479166667,0.0010416667,-0.0927083333,0.1520833333,-0.0552083333,0.0197916667,-0.0270833333,0.0489583333,0.3177083333,-0.2520833333,-0.0197916667,-0.2260416667,0.0270833333,0.1114583333,-0.1114583333,0.4020833333,0.0927083333,
                             -0.0489583333,-0.0864583333,-0.0770833333,0.1614583333,-0.0927083333,-0.1770833333,0.3177083333,-0.0197916667,-0.2520833333,0.0197916667,0.0489583333,-0.0270833333,-0.0927083333,-0.0552083333,0.1520833333,-0.0197916667,0.0010416667,-0.0479166667,-0.2260416667,0.1114583333,0.0270833333,-0.1114583333,0.0927083333,0.4020833333;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < cell_stiffness_matrix.rows(); ++i)
      for (unsigned j = 0; j < cell_stiffness_matrix.cols(); ++j)
        REQUIRE(stiffness_matrix(i, j) ==
                Approx(cell_stiffness_matrix(i, j)).epsilon(Tolerance));
  }
}
