#include <cmath>
#include <limits>
#include <memory>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>

#include "catch.hpp"
// MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

#include "constraints.h"
#include "element.h"
#include "function_base.h"
#include "hexahedron_element.h"
#include "linear_function.h"
#include "mesh.h"
#include "node.h"
#include "partio_writer.h"
#include "quadrilateral_element.h"

//! \brief Check mesh class for 3D case
TEST_CASE("Mesh is checked for 3D case", "[mesh][3D]") {
  // Dimension
  const unsigned Dim = 3;
  // Degrees of freedom
  const unsigned Dof = 6;
  // Number of phases
  const unsigned Nphases = 1;
  // Number of nodes per cell
  const unsigned Nnodes = 8;
  // Tolerance
  const double Tolerance = 1.E-9;
  // Json property
  Json jfunctionproperties;
  jfunctionproperties["id"] = 0;
  std::vector<double> x_values{{0.0, 0.5, 1.0, 1.5}};
  std::vector<double> fx_values{{0.0, 1.0, 1.0, 0.0}};
  jfunctionproperties["xvalues"] = x_values;
  jfunctionproperties["fxvalues"] = fx_values;

  // Assign material
  unsigned mid = 0;
  std::vector<unsigned> mids(1, mid);
  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;

  auto le_material =
      Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
          "LinearElastic3D", std::move(0), jmaterial);

  std::map<unsigned, std::shared_ptr<mpm::Material<Dim>>> materials;
  materials[mid] = le_material;

  // math function
  std::shared_ptr<mpm::FunctionBase> mfunction =
      std::make_shared<mpm::LinearFunction>(0, jfunctionproperties);

  // 8-noded hexahedron element
  std::shared_ptr<mpm::Element<Dim>> element =
      Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

  //! Check Mesh IDs
  SECTION("Check mesh ids") {
    //! Check for id = 0
    SECTION("Mesh id is zero") {
      unsigned id = 0;
      auto mesh = std::make_shared<mpm::Mesh<Dim>>(id);
      REQUIRE(mesh->id() == 0);
      REQUIRE(mesh->is_isoparametric() == true);
    }

    SECTION("Mesh id is zero and cartesian") {
      unsigned id = 0;
      auto mesh = std::make_shared<mpm::Mesh<Dim>>(id, false);
      REQUIRE(mesh->id() == 0);
      REQUIRE(mesh->is_isoparametric() == false);
    }

    SECTION("Mesh id is positive") {
      //! Check for id is a positive value
      unsigned id = std::numeric_limits<unsigned>::max();
      auto mesh = std::make_shared<mpm::Mesh<Dim>>(id);
      REQUIRE(mesh->id() == std::numeric_limits<unsigned>::max());
    }
  }

  SECTION("Add neighbours") {
    auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);
    auto neighbourmesh = std::make_shared<mpm::Mesh<Dim>>(1);
    REQUIRE(mesh->nneighbours() == 0);
    mesh->add_neighbour(0, neighbourmesh);
    REQUIRE(mesh->nneighbours() == 1);
  }

  // Check add / remove particle
  SECTION("Check add / remove particle functionality") {
    // Particle 1
    mpm::Index id1 = 0;
    Eigen::Vector3d coords;
    coords.setZero();
    std::shared_ptr<mpm::ParticleBase<Dim>> particle1 =
        std::make_shared<mpm::Particle<Dim>>(id1, coords);

    // Particle 2
    mpm::Index id2 = 1;
    coords << 2., 2., 2.;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle2 =
        std::make_shared<mpm::Particle<Dim>>(id2, coords);

    auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);
    // Check mesh is active
    REQUIRE(mesh->status() == false);

    // Check nodal coordinates size
    REQUIRE(mesh->nodal_coordinates().size() == 0);
    // Check node pairs size
    REQUIRE(mesh->node_pairs().size() == 0);

    // Define nodes
    coords << 0, 0, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    REQUIRE(mesh->add_node(node0) == true);

    coords << 2, 0, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    REQUIRE(mesh->add_node(node1) == true);

    coords << 2, 2, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    REQUIRE(mesh->add_node(node2) == true);

    coords << 0, 2, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    REQUIRE(mesh->add_node(node3) == true);

    coords << 0, 0, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    REQUIRE(mesh->add_node(node4) == true);

    coords << 2, 0, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    REQUIRE(mesh->add_node(node5) == true);

    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    REQUIRE(mesh->add_node(node6) == true);

    coords << 0, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);
    REQUIRE(mesh->add_node(node7) == true);

    // Create cell1
    auto cell1 = std::make_shared<mpm::Cell<Dim>>(id1, Nnodes, element);

    // Add nodes to cell
    cell1->add_node(0, node0);
    cell1->add_node(1, node1);
    cell1->add_node(2, node2);
    cell1->add_node(3, node3);
    cell1->add_node(4, node4);
    cell1->add_node(5, node5);
    cell1->add_node(6, node6);
    cell1->add_node(7, node7);

    REQUIRE(cell1->nnodes() == 8);

    REQUIRE(mesh->add_cell(cell1) == true);

    REQUIRE(cell1->initialise() == true);

    // Check nodal coordinates size
    REQUIRE(mesh->nodal_coordinates().size() == 8);
    // Check node pairs size
    REQUIRE(mesh->node_pairs().size() == 12);

    // Add particle 1 and check
    REQUIRE(mesh->add_particle(particle1) == true);
    // Add particle 2 and check
    REQUIRE(mesh->add_particle(particle2) == true);
    // Add particle 2 again and check
    REQUIRE(mesh->add_particle(particle2) == false);

    // Check mesh is active
    REQUIRE(mesh->status() == true);
    // Check number of particles in mesh
    REQUIRE(mesh->nparticles() == 2);
    REQUIRE(mesh->nparticles("P3D") == 2);

    // Remove particle 2 and check
    REQUIRE(mesh->remove_particle(particle2) == true);
    // Check number of particles in mesh
    REQUIRE(mesh->nparticles() == 1);
    REQUIRE(mesh->nparticles("P3D") == 1);

    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size == 1) cell1->rank(1);

    mesh->find_domain_shared_nodes();
    REQUIRE(node0->mpi_ranks().size() == mpi_size);
    REQUIRE(node1->mpi_ranks().size() == mpi_size);
    REQUIRE(node2->mpi_ranks().size() == mpi_size);
    REQUIRE(node3->mpi_ranks().size() == mpi_size);
    REQUIRE(node4->mpi_ranks().size() == mpi_size);
    REQUIRE(node5->mpi_ranks().size() == mpi_size);
    REQUIRE(node6->mpi_ranks().size() == mpi_size);
    REQUIRE(node7->mpi_ranks().size() == mpi_size);

    // Check mesh ghost boundary cells
    mesh->find_ghost_boundary_cells();

    // Remove all non-rank particles in mesh
    mesh->remove_all_nonrank_particles();
    // Check number of particles in mesh
    REQUIRE(mesh->nparticles() == 0);
    REQUIRE(mesh->nparticles("P3D") == 0);

    // Add and use remove all particles
    REQUIRE(mesh->add_particle(particle1) == true);
    REQUIRE(mesh->add_particle(particle2) == true);

    // Check number of particles in mesh
    REQUIRE(mesh->nparticles() == 2);
    std::vector<mpm::Index> remove_pids = {{0, 1}};
    // Remove all particles
    mesh->remove_particles(remove_pids);
    // Check number of particles in mesh
    REQUIRE(mesh->nparticles() == 0);

    // Test assign node concentrated force
    SECTION("Check assign node concentrated force") {
      unsigned Nphase = 0;
      // Set external force to zero
      Eigen::Matrix<double, Dim, 1> force;
      force.setZero();
      REQUIRE_NOTHROW(node0->update_external_force(false, Nphase, force));
      REQUIRE_NOTHROW(node1->update_external_force(false, Nphase, force));

      // Check external force
      for (unsigned i = 0; i < Dim; ++i) {
        REQUIRE(node0->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));
        REQUIRE(node1->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));
      }

      tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
      node_sets[0] = std::vector<mpm::Index>{0, 1};

      REQUIRE(mesh->create_node_sets(node_sets, true) == true);

      REQUIRE(mesh->assign_nodal_concentrated_forces(mfunction, 0, 0, 10.5) ==
              true);
      REQUIRE(mesh->assign_nodal_concentrated_forces(mfunction, -1, 0, 0.5) ==
              true);
      REQUIRE(mesh->assign_nodal_concentrated_forces(mfunction, 5, 0, 0.5) ==
              false);
      REQUIRE(mesh->assign_nodal_concentrated_forces(mfunction, -5, 1, 0.5) ==
              false);

      double current_time = 0.0;
      node0->apply_concentrated_force(Nphase, current_time);
      node1->apply_concentrated_force(Nphase, current_time);
      // Check external force
      for (unsigned i = 0; i < Dim; ++i) {
        REQUIRE(node0->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));
        REQUIRE(node1->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));
      }

      current_time = 0.25;
      node0->apply_concentrated_force(Nphase, current_time);
      node1->apply_concentrated_force(Nphase, current_time);
      std::vector<double> ext_forces = {0.25, 0., 0.};
      // Check external force
      for (unsigned i = 0; i < Dim; ++i) {
        REQUIRE(node0->external_force(Nphase)(i) ==
                Approx(ext_forces.at(i)).epsilon(Tolerance));
        REQUIRE(node1->external_force(Nphase)(i) ==
                Approx(ext_forces.at(i)).epsilon(Tolerance));
      }

      current_time = 5.0;
      node0->apply_concentrated_force(Nphase, current_time);
      node1->apply_concentrated_force(Nphase, current_time);
      ext_forces = {0.25, 0., 0.};
      // Check external force
      for (unsigned i = 0; i < Dim; ++i) {
        REQUIRE(node0->external_force(Nphase)(i) ==
                Approx(ext_forces.at(i)).epsilon(Tolerance));
        REQUIRE(node1->external_force(Nphase)(i) ==
                Approx(ext_forces.at(i)).epsilon(Tolerance));
      }
    }

    // Test create nodal acceleration constraint
    SECTION("Check create nodal acceleration constraint") {
      tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
      node_sets[0] = std::vector<mpm::Index>{0, 1};

      REQUIRE(mesh->create_node_sets(node_sets, true) == true);

      int set_id = 0;
      int dir = 0;
      double constraint = 1.0;
      // Add acceleration constraint to mesh
      auto acceleration_constraint =
          std::make_shared<mpm::AccelerationConstraint>(set_id, mfunction, dir,
                                                        constraint);
      REQUIRE(mesh->create_nodal_acceleration_constraint(
                  set_id, acceleration_constraint) == true);

      set_id = -1;
      // Add acceleration constraint to mesh
      acceleration_constraint = std::make_shared<mpm::AccelerationConstraint>(
          set_id, mfunction, dir, constraint);
      REQUIRE(mesh->create_nodal_acceleration_constraint(
                  set_id, acceleration_constraint) == true);

      // When constraints fail: invalid direction
      dir = 3;
      // Add acceleration constraint to mesh
      acceleration_constraint = std::make_shared<mpm::AccelerationConstraint>(
          set_id, mfunction, dir, constraint);
      REQUIRE(mesh->create_nodal_acceleration_constraint(
                  set_id, acceleration_constraint) == false);

      // When constraints fail: invalid node set
      set_id = 1;
      // Add acceleration constraint to mesh
      acceleration_constraint = std::make_shared<mpm::AccelerationConstraint>(
          set_id, mfunction, dir, constraint);
      REQUIRE(mesh->create_nodal_acceleration_constraint(
                  set_id, acceleration_constraint) == false);

      // Vector of particle coordinates
      std::vector<std::tuple<mpm::Index, unsigned, double>>
          acceleration_constraints;
      //! Constraints object
      auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);
      // Add acceleration constraint to node
      acceleration_constraints.emplace_back(std::make_tuple(0, 0, 1.0));
      REQUIRE(constraints->assign_nodal_acceleration_constraints(
                  acceleration_constraints) == true);

      double current_time = 0.5;
      // Update acceleration constraint
      REQUIRE_NOTHROW(
          mesh->update_nodal_acceleration_constraints(current_time));
    }

    // Test nonlocal mesh functions
    SECTION("Check nonlocal mesh functions") {
      tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
      node_sets[0] = std::vector<mpm::Index>{0, 3};
      node_sets[1] = std::vector<mpm::Index>{1, 2, 4, 5, 6, 7};
      std::vector<unsigned> n0 = {4, 1, 0};
      std::vector<unsigned> n1 = {4, 0, 3};
      std::vector<unsigned> n2 = {4, 0, 3};
      std::vector<unsigned> n3 = {4, 1, 0};
      std::vector<unsigned> n4 = {4, 0, 3};
      std::vector<unsigned> n5 = {4, 0, 3};
      std::vector<unsigned> n6 = {4, 0, 3};
      std::vector<unsigned> n7 = {4, 0, 3};

      REQUIRE_NOTHROW(node0->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node1->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node2->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node3->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node4->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node5->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node6->initialise_nonlocal_node());
      REQUIRE_NOTHROW(node7->initialise_nonlocal_node());

      REQUIRE(mesh->create_node_sets(node_sets, true) == true);

      REQUIRE(mesh->assign_nodal_nonlocal_type(0, 1, 1) == true);
      REQUIRE(mesh->assign_nodal_nonlocal_type(1, 2, 3) == true);
      REQUIRE(mesh->assign_nodal_nonlocal_type(-1, 0, 4) == true);

      REQUIRE(node0->nonlocal_node_type() == n0);
      REQUIRE(node1->nonlocal_node_type() == n1);
      REQUIRE(node2->nonlocal_node_type() == n2);
      REQUIRE(node3->nonlocal_node_type() == n3);
      REQUIRE(node4->nonlocal_node_type() == n4);
      REQUIRE(node5->nonlocal_node_type() == n5);
      REQUIRE(node6->nonlocal_node_type() == n6);
      REQUIRE(node7->nonlocal_node_type() == n7);

      // Empty map
      tsl::robin_map<std::string, double> nonlocal_properties;
      REQUIRE_THROWS(
          mesh->upgrade_cells_to_nonlocal("ED3H8", 0, nonlocal_properties));
      REQUIRE(mesh->upgrade_cells_to_nonlocal("ED3H8P2B", 1,
                                              nonlocal_properties) == true);
    }
  }

  // Check add / remove node
  SECTION("Check add / remove node functionality") {
    // Node 1
    mpm::Index id1 = 0;
    Eigen::Vector3d coords;
    coords.setZero();
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id1, coords);

    // Node 2
    mpm::Index id2 = 1;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id2, coords);

    auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);
    // Check mesh is active
    REQUIRE(mesh->status() == false);

    // Check nodal coordinates size
    REQUIRE(mesh->nodal_coordinates().size() == 0);
    // Check node pairs size
    REQUIRE(mesh->node_pairs().size() == 0);

    // Add node 1 and check
    REQUIRE(mesh->add_node(node1) == true);
    // Add node 2 and check
    REQUIRE(mesh->add_node(node2) == true);
    // Add node 2 again and check
    REQUIRE(mesh->add_node(node2) == false);

    // Check nodal coordinates size
    REQUIRE(mesh->nodal_coordinates().size() == 2);

    // Check number of nodes in mesh
    REQUIRE(mesh->nnodes() == 2);

    // Update coordinates
    Eigen::Vector3d coordinates;
    coordinates << 7., 7., 7.;

    // Set only node2 to be active
    node2->assign_status(true);

    // Check iterate over functionality if nodes are active
    mesh->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Dim>::assign_coordinates,
                  std::placeholders::_1, coordinates),
        std::bind(&mpm::NodeBase<Dim>::status, std::placeholders::_1));

    // Node 1
    {
      // Check if nodal coordinate update has gone through
      auto check_coords = node1->coordinates();
      // Check if coordinates for each node is zero
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(0.).epsilon(Tolerance));
    }
    // Node 2
    {
      // Check if nodal coordinate update has gone through
      auto check_coords = node2->coordinates();
      // Check if coordinates for each node is zero
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(7.).epsilon(Tolerance));
    }

    coordinates.setZero();
    node1->assign_coordinates(coordinates);
    node2->assign_coordinates(coordinates);
    for (unsigned i = 0; i < coordinates.size(); ++i) {
      REQUIRE(node1->coordinates()[i] == Approx(0.).epsilon(Tolerance));
      REQUIRE(node2->coordinates()[i] == Approx(0.).epsilon(Tolerance));
    }

    REQUIRE(node1->status() == false);
    REQUIRE(node2->status() == true);

    mesh->find_active_nodes();

    // Check iterate over functionality if nodes are active
    coordinates.fill(5.3);

    mesh->iterate_over_active_nodes(
        std::bind(&mpm::NodeBase<Dim>::assign_coordinates,
                  std::placeholders::_1, coordinates));

    // Node 1
    {
      // Check if nodal coordinate update has gone through
      auto check_coords = node1->coordinates();
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(0.).epsilon(Tolerance));
    }
    // Node 2
    {
      // Check if nodal coordinate update has gone through
      auto check_coords = node2->coordinates();
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(5.3).epsilon(Tolerance));
    }

    coordinates.fill(7.0);

    // Check iterate over functionality
    mesh->iterate_over_nodes(std::bind(&mpm::NodeBase<Dim>::assign_coordinates,
                                       std::placeholders::_1, coordinates));

    // Node 1
    {
      // Check if nodal coordinate update has gone through
      auto check_coords = node1->coordinates();
      // Check if coordinates for each node is zero
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(7.).epsilon(Tolerance));
    }
    // Node 2
    {
      // Check if nodal coordinate update has gone through
      auto check_coords = node2->coordinates();
      // Check if coordinates for each node is zero
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(7.).epsilon(Tolerance));
    }

    mesh->iterate_over_nodes(std::bind(&mpm::NodeBase<Dim>::update_mass,
                                       std::placeholders::_1, false, 0, 10.0));

#ifdef USE_MPI
    // Get number of MPI ranks
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size == 1) {
      // Run if there is more than a single MPI task
      // MPI all reduce nodal mass
      mesh->template nodal_halo_exchange<double, 1>(
          std::bind(&mpm::NodeBase<Dim>::mass, std::placeholders::_1, 0),
          std::bind(&mpm::NodeBase<Dim>::update_mass, std::placeholders::_1,
                    false, 0, std::placeholders::_2));
      // MPI all reduce nodal momentum
      mesh->template nodal_halo_exchange<Eigen::Matrix<double, Dim, 1>, Dim>(
          std::bind(&mpm::NodeBase<Dim>::coordinates, std::placeholders::_1),
          std::bind(&mpm::NodeBase<Dim>::assign_coordinates,
                    std::placeholders::_1, std::placeholders::_2));
    }
#endif
    // Node 1
    {
      // Check mass
      REQUIRE(node1->mass(0) == Approx(10.).epsilon(Tolerance));
      // Check if nodal coordinate update has gone through
      auto check_coords = node1->coordinates();
      // Check if coordinates for each node is zero
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(7.).epsilon(Tolerance));
    }
    // Node 2
    {
      // Check mass
      REQUIRE(node2->mass(0) == Approx(10.).epsilon(Tolerance));
      // Check if nodal coordinate update has gone through
      auto check_coords = node2->coordinates();
      // Check if coordinates for each node is zero
      for (unsigned i = 0; i < check_coords.size(); ++i)
        REQUIRE(check_coords[i] == Approx(7.).epsilon(Tolerance));
    }

    // Remove node 2 and check
    REQUIRE(mesh->remove_node(node2) == true);
    // Check number of nodes in mesh
    REQUIRE(mesh->nnodes() == 1);
  }

  // Check add / remove cell
  SECTION("Check add / remove cell functionality") {
    // Cell 1
    mpm::Index id1 = 0;
    Eigen::Vector3d coords;
    coords.setZero();
    auto cell1 = std::make_shared<mpm::Cell<Dim>>(id1, Nnodes, element);

    // Cell 2
    mpm::Index id2 = 1;
    auto cell2 = std::make_shared<mpm::Cell<Dim>>(id2, Nnodes, element);

    auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);
    // Check mesh is active
    REQUIRE(mesh->status() == false);

    // Add cell 1 and check
    REQUIRE(mesh->add_cell(cell1) == true);
    // Add cell 2 and check
    REQUIRE(mesh->add_cell(cell2) == true);
    // Add cell 2 again and check
    REQUIRE(mesh->add_cell(cell2) == false);

    // Check number of cells in mesh
    REQUIRE(mesh->ncells() == 2);

    // Check iterate over functionality
    mesh->iterate_over_cells(
        std::bind(&mpm::Cell<Dim>::nnodes, std::placeholders::_1));

    // Remove cell 2 and check
    REQUIRE(mesh->remove_cell(cell2) == true);
    // Check number of cells in mesh
    REQUIRE(mesh->ncells() == 1);
  }

  SECTION("Check particle is in cell") {
    // Index
    mpm::Index id1 = 0;

    // Mesh
    auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);
    // Check mesh is inactive (false)
    REQUIRE(mesh->status() == false);

    // Coordinates
    Eigen::Vector3d coords;
    coords.setZero();

    // Define nodes
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

    // Create cell1
    coords.setZero();
    auto cell1 = std::make_shared<mpm::Cell<Dim>>(id1, Nnodes, element);

    // Add nodes to cell
    cell1->add_node(0, node0);
    cell1->add_node(1, node1);
    cell1->add_node(2, node2);
    cell1->add_node(3, node3);
    cell1->add_node(4, node4);
    cell1->add_node(5, node5);
    cell1->add_node(6, node6);
    cell1->add_node(7, node7);

    REQUIRE(cell1->nnodes() == 8);

    // Initialise cell and compute volume
    REQUIRE(cell1->initialise() == true);

    // Particle type 3D
    const std::string particle_type = "P3D";

    // Initialise material models
    mesh->initialise_material_models(materials);

    REQUIRE(mesh->nparticles() == 0);
    // Generate material points in cell
    REQUIRE(mesh->generate_material_points(1, particle_type, mids, -1, 0) ==
            false);
    REQUIRE(mesh->nparticles() == 0);

    // Add cell 1 and check
    REQUIRE(mesh->add_cell(cell1) == true);

    SECTION("Check generating 1 particle / cell") {
      // Generate material points in cell
      REQUIRE(mesh->generate_material_points(1, particle_type, mids, -1, 0) ==
              true);
      REQUIRE(mesh->nparticles() == 1);
    }

    SECTION("Check generating 2 particle / cell") {
      REQUIRE(mesh->generate_material_points(2, particle_type, mids, -1, 0) ==
              true);
      REQUIRE(mesh->nparticles() == 8);
    }

    SECTION("Check generating 3 particle / cell") {
      REQUIRE(mesh->generate_material_points(3, particle_type, mids, -1, 0) ==
              true);
      REQUIRE(mesh->nparticles() == 27);
    }

    SECTION("Check material point generation") {
      // Assign argc and argv to nput arguments of MPM
      int argc = 7;
      char* argv[] = {(char*)"./mpm",   (char*)"-f", (char*)"./",
                      (char*)"-p",      (char*)"8",  (char*)"-i",
                      (char*)"mpm.json"};

      // Create an IO object
      auto io = std::make_shared<mpm::IO>(argc, argv);

      tsl::robin_map<mpm::Index, std::vector<mpm::Index>> cell_sets;
      cell_sets[1] = std::vector<mpm::Index>{0};

      REQUIRE(mesh->create_cell_sets(cell_sets, true) == true);

      REQUIRE(mesh->nparticles() == 0);

      SECTION("Gauss point generation") {
        // Gauss point generation
        Json jgen;
        jgen["type"] = "gauss";
        jgen["material_id"] = mid;
        jgen["pset_id"] = 2;
        jgen["cset_id"] = 1;
        jgen["particle_type"] = "P3D";
        jgen["check_duplicates"] = false;
        jgen["nparticles_per_dir"] = 2;

        // Generate
        REQUIRE(mesh->generate_particles(io, jgen) == true);
        // Number of particles
        REQUIRE(mesh->nparticles() == 8);
      }

      SECTION("Inject points") {
        // Gauss point generation
        Json jgen;
        jgen["type"] = "inject";
        jgen["material_id"] = mid;
        jgen["cset_id"] = 1;
        jgen["particle_type"] = "P3D";
        jgen["check_duplicates"] = false;
        jgen["nparticles_per_dir"] = 2;
        jgen["velocity"] = {0., 0.};
        jgen["duration"] = {0.1, 0.2};

        // Generate
        REQUIRE(mesh->generate_particles(io, jgen) == true);
        // Inject particles
        REQUIRE_NOTHROW(mesh->inject_particles(0.05));
        // Number of particles
        REQUIRE(mesh->nparticles() == 0);
        // Inject particles
        REQUIRE_NOTHROW(mesh->inject_particles(0.25));
        // Number of particles
        REQUIRE(mesh->nparticles() == 0);
        // Inject particles
        REQUIRE_NOTHROW(mesh->inject_particles(0.15));
        // Number of particles
        REQUIRE(mesh->nparticles() == 8);
      }

      SECTION("Generate point fail") {
        // Gauss point generation
        Json jgen;
        jgen["type"] = "fail";

        // Generate
        REQUIRE(mesh->generate_particles(io, jgen) == false);
      }
    }

    // Particle 1
    coords << 1.0, 1.0, 1.0;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle1 =
        std::make_shared<mpm::Particle<Dim>>(100, coords);

    // Particle 2
    coords << 1.5, 1.5, 1.5;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle2 =
        std::make_shared<mpm::Particle<Dim>>(101, coords);

    // Add particle 1 and check
    REQUIRE(mesh->add_particle(particle1) == true);
    // Add particle 2 and check
    REQUIRE(mesh->add_particle(particle2) == true);

    // Check mesh is active
    REQUIRE(mesh->status() == true);

    // Locate particles in a mesh
    mesh->locate_particles_mesh();

    // Check location of particle 1
    REQUIRE(particle1->cell_id() == 0);
    // Check location of particle 2
    REQUIRE(particle2->cell_id() == 0);
  }

  //! Check create nodes and cells in a mesh
  SECTION("Check create nodes and cells") {
    // Vector of nodal coordinates
    std::vector<Eigen::Matrix<double, Dim, 1>> coordinates;

    // Nodal coordinates
    Eigen::Matrix<double, Dim, 1> node;

    // Cell 0
    // Node 0
    node << 0., 0., 0.;
    coordinates.emplace_back(node);
    // Node 1
    node << 0.5, 0., 0.;
    coordinates.emplace_back(node);
    // Node 2
    node << 0.5, 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 3
    node << 0., 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 4
    node << 0., 0., 0.5;
    coordinates.emplace_back(node);
    // Node 5
    node << 0.5, 0., 0.5;
    coordinates.emplace_back(node);
    // Node 6
    node << 0.5, 0.5, 0.5;
    coordinates.emplace_back(node);
    // Node 7
    node << 0., 0.5, 0.5;
    coordinates.emplace_back(node);

    // Cell 1
    // Node 8
    node << 1.0, 0., 0.;
    coordinates.emplace_back(node);
    // Node 9
    node << 1.0, 0.5, 0.;
    coordinates.emplace_back(node);
    // Node 10
    node << 1.0, 0., 0.5;
    coordinates.emplace_back(node);
    // Node 11
    node << 1.0, 0.5, 0.5;
    coordinates.emplace_back(node);

    // Create a new mesh
    unsigned meshid = 0;
    auto mesh = std::make_shared<mpm::Mesh<Dim>>(meshid);

    SECTION("Check creation of nodes") {
      // Node type 3D
      const std::string node_type = "N3D";
      // Global node index
      mpm::Index gnid = 0;
      mesh->create_nodes(gnid, node_type, coordinates);
      // Check if mesh has added nodes
      REQUIRE(mesh->nnodes() == coordinates.size());
      // Try again this shouldn't add more coordinates
      mesh->create_nodes(gnid, node_type, coordinates);
      // Check if mesh has added nodes
      REQUIRE(mesh->nnodes() == coordinates.size());
      // Clear coordinates and try creating a list of nodes with an empty list
      unsigned nnodes = coordinates.size();
      coordinates.clear();
      // This fails with empty list error in node creation
      mesh->create_nodes(gnid, node_type, coordinates);
      REQUIRE(mesh->nnodes() == nnodes);

      SECTION("Check creation of cells") {
        // Cell with node ids
        std::vector<std::vector<mpm::Index>> cells{// cell #0
                                                   {0, 1, 2, 3, 4, 5, 6, 7},
                                                   // cell #1
                                                   {1, 8, 9, 2, 5, 10, 11, 6}};
        // Assign 8-noded hexahedron element to cell
        std::shared_ptr<mpm::Element<Dim>> element =
            Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

        // Global cell index
        mpm::Index gcid = 0;
        mesh->create_cells(gcid, element, cells);
        // Check if mesh has added cells
        REQUIRE(mesh->ncells() == cells.size());
        // Try again this shouldn't add more cells
        mesh->create_cells(gcid, element, cells);
        // Check if mesh has added cells
        REQUIRE(mesh->ncells() == cells.size());
        // Clear cells and try creating a list of empty cells
        unsigned ncells = cells.size();
        cells.clear();
        // This fails with empty list error in node creation
        gcid = 100;
        mesh->create_cells(gcid, element, cells);
        REQUIRE(mesh->ncells() == ncells);

        // Try with invalid node ids
        cells = {// cell #0
                 {90, 91, 92, 93, 94, 95, 96, 97},
                 // cell #1
                 {71, 88, 89, 82, 85, 80, 81, 86}};
        gcid = 200;
        mesh->create_cells(gcid, element, cells);
        REQUIRE(mesh->ncells() == ncells);

        SECTION("Check creation of particles") {
          // Vector of particle coordinates
          std::vector<Eigen::Matrix<double, Dim, 1>> coordinates;
          coordinates.clear();

          // Particle coordinates
          Eigen::Matrix<double, Dim, 1> particle;

          // Cell 0
          // Particle 0
          particle << 0.125, 0.125, 0.125;
          coordinates.emplace_back(particle);
          // Particle 1
          particle << 0.25, 0.125, 0.125;
          coordinates.emplace_back(particle);
          // Particle 2
          particle << 0.25, 0.25, 0.125;
          coordinates.emplace_back(particle);
          // Particle 3
          particle << 0.125, 0.25, 0.125;
          coordinates.emplace_back(particle);
          // Particle 4
          particle << 0.125, 0.125, 0.25;
          coordinates.emplace_back(particle);
          // Particle 5
          particle << 0.25, 0.125, 0.25;
          coordinates.emplace_back(particle);
          // Particle 6
          particle << 0.25, 0.25, 0.25;
          coordinates.emplace_back(particle);
          // Particle 7
          particle << 0.125, 0.25, 0.25;
          coordinates.emplace_back(particle);

          // Cell 1
          // Particle 8
          particle << 0.675, 0.125, 0.125;
          coordinates.emplace_back(particle);
          // Particle 9
          particle << 0.85, 0.125, 0.125;
          coordinates.emplace_back(particle);
          // Particle 10
          particle << 0.85, 0.25, 0.125;
          coordinates.emplace_back(particle);
          // Particle 11
          particle << 0.675, 0.25, 0.125;
          coordinates.emplace_back(particle);
          // Particle 12
          particle << 0.675, 0.125, 0.25;
          coordinates.emplace_back(particle);
          // Particle 13
          particle << 0.85, 0.125, 0.25;
          coordinates.emplace_back(particle);
          // Particle 14
          particle << 0.85, 0.25, 0.25;
          coordinates.emplace_back(particle);
          // Particle 15
          particle << 0.675, 0.25, 0.25;
          coordinates.emplace_back(particle);

          // Initialise material models in mesh
          mesh->initialise_material_models(materials);

          SECTION("Check addition of particles to mesh") {
            // Particle type 3D
            const std::string particle_type = "P3D";
            // Create particles from file
            mesh->create_particles(particle_type, coordinates, mids, 0, false);
            // Check if mesh has added particles
            REQUIRE(mesh->nparticles() == coordinates.size());
            // Clear coordinates and try creating a list of particles with an
            // empty list
            unsigned nparticles = coordinates.size();
            coordinates.clear();
            // This fails with empty list error in particle creation
            mesh->create_particles(particle_type, coordinates, mids, 1, false);
            REQUIRE(mesh->nparticles() == nparticles);

            // Test assign particles cells again should fail
            SECTION("Check assign particles cells") {
              // Vector of particle cells
              std::vector<std::array<mpm::Index, 2>> particles_cells;
              // Particle cells
              particles_cells.emplace_back(std::array<mpm::Index, 2>({0, 0}));
              particles_cells.emplace_back(std::array<mpm::Index, 2>({1, 0}));
              particles_cells.emplace_back(std::array<mpm::Index, 2>({2, 0}));
              particles_cells.emplace_back(std::array<mpm::Index, 2>(
                  {3, std::numeric_limits<mpm::Index>::max()}));
              particles_cells.emplace_back(std::array<mpm::Index, 2>({50, 0}));

              REQUIRE(mesh->assign_particles_cells(particles_cells) == false);
            }

            // Particles coordinates
            REQUIRE(mesh->particle_coordinates().size() == mesh->nparticles());
            // Particle stresses
            std::string attribute = "stresses";
            REQUIRE(mesh->template particles_tensor_data<6>(attribute).size() ==
                    mesh->nparticles());
            // Particle strains
            attribute = "strains";
            REQUIRE(mesh->template particles_tensor_data<6>(attribute).size() ==
                    mesh->nparticles());
            // Particle velocities
            attribute = "velocities";
            REQUIRE(mesh->particles_vector_data(attribute).size() ==
                    mesh->nparticles());
            // Particle mass
            attribute = "mass";
            REQUIRE(mesh->particles_scalar_data(attribute).size() ==
                    mesh->nparticles());

            // Particle invalid data for tensor, vector, scalar
            attribute = "invalid";
            const auto& invalid_tensor =
                mesh->template particles_tensor_data<6>(attribute);
            REQUIRE(invalid_tensor.size() == mesh->nparticles());
            const auto& invalid_vector = mesh->particles_vector_data(attribute);
            REQUIRE(invalid_vector.size() == mesh->nparticles());
            const auto& invalid_scalar = mesh->particles_scalar_data(attribute);
            REQUIRE(invalid_scalar.size() == mesh->nparticles());

            // State variable
            attribute = "pdstrain";
            REQUIRE(mesh->particles_statevars_data(attribute).size() ==
                    mesh->nparticles());

            SECTION("Locate particles in mesh") {
              // Locate particles in a mesh
              auto particles = mesh->locate_particles_mesh();

              // Should find all particles in mesh
              REQUIRE(particles.size() == 0);
              // Create particle 100
              Eigen::Vector3d coords;
              coords << 100., 100., 100.;

              mpm::Index pid = 100;
              std::shared_ptr<mpm::ParticleBase<Dim>> particle100 =
                  std::make_shared<mpm::Particle<Dim>>(pid, coords);

              // Add particle100 and check
              REQUIRE(mesh->add_particle(particle100) == false);

              // Locate particles in a mesh
              particles = mesh->locate_particles_mesh();
              // Should miss particle100
              REQUIRE(particles.size() == 0);

              SECTION("Check return particles cells") {
                // Vector of particle cells
                std::vector<std::array<mpm::Index, 2>> particles_cells;
                // Particle cells
                particles_cells.emplace_back(std::array<mpm::Index, 2>({0, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({1, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({2, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({3, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({4, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({5, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({6, 0}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({7, 0}));

                particles_cells.emplace_back(std::array<mpm::Index, 2>({8, 1}));
                particles_cells.emplace_back(std::array<mpm::Index, 2>({9, 1}));
                particles_cells.emplace_back(
                    std::array<mpm::Index, 2>({10, 1}));
                particles_cells.emplace_back(
                    std::array<mpm::Index, 2>({11, 1}));
                particles_cells.emplace_back(
                    std::array<mpm::Index, 2>({12, 1}));
                particles_cells.emplace_back(
                    std::array<mpm::Index, 2>({13, 1}));
                particles_cells.emplace_back(
                    std::array<mpm::Index, 2>({14, 1}));
                particles_cells.emplace_back(
                    std::array<mpm::Index, 2>({15, 1}));

                auto check_particles_cells = mesh->particles_cells();

                REQUIRE(check_particles_cells.size() == mesh->nparticles());

                for (unsigned i = 0; i < particles_cells.size(); ++i)
                  for (unsigned j = 0; j < 2; ++j)
                    REQUIRE(check_particles_cells.at(i).at(j) ==
                            particles_cells.at(i).at(j));
              }
            }
            // Test HDF5
            SECTION("Write particles HDF5") {
              REQUIRE(mesh->write_particles_hdf5("particles-3d.h5") == true);

              auto phdf5 = mesh->particles_hdf5();
              REQUIRE(phdf5.size() == mesh->nparticles());

#ifdef USE_PARTIO
              REQUIRE_NOTHROW(mpm::partio::write_particles(
                  "partio-3d.bgeo", mesh->particles_hdf5()));
              // Check if .bgeo exists
              REQUIRE(boost::filesystem::exists("./partio-3d.bgeo") == true);
#endif
            }

            // Test assign particles volumes
            SECTION("Check assign particles volumes") {
              // Vector of particle coordinates
              std::vector<std::tuple<mpm::Index, double>> particles_volumes;
              // Volumes
              particles_volumes.emplace_back(std::make_tuple(0, 10.5));
              particles_volumes.emplace_back(std::make_tuple(1, 10.5));

              REQUIRE(mesh->nparticles() == 16);

              REQUIRE(mesh->assign_particles_volumes(particles_volumes) ==
                      true);

              // When volume assignment fails
              particles_volumes.emplace_back(std::make_tuple(2, 0.0));
              particles_volumes.emplace_back(std::make_tuple(3, -10.0));

              REQUIRE(mesh->assign_particles_volumes(particles_volumes) ==
                      false);
            }

            // Test assign particles tractions
            SECTION("Check assign particles tractions") {
              // Vector of particle coordinates
              tsl::robin_map<mpm::Index, std::vector<mpm::Index>> particle_sets;
              particle_sets[0] = std::vector<mpm::Index>{0};
              particle_sets[1] = std::vector<mpm::Index>{1};
              particle_sets[2] = std::vector<mpm::Index>{2};
              particle_sets[3] = std::vector<mpm::Index>{3};

              REQUIRE(mesh->create_particle_sets(particle_sets, true) == true);

              REQUIRE(mesh->nparticles() == 16);

              REQUIRE(mesh->create_particles_tractions(mfunction, 0, 0, 10.5) ==
                      true);
              REQUIRE(mesh->create_particles_tractions(mfunction, 1, 1,
                                                       -10.5) == true);
              REQUIRE(mesh->create_particles_tractions(mfunction, 2, 0,
                                                       -12.5) == true);
              REQUIRE(mesh->create_particles_tractions(mfunction, 3, 1, 0.5) ==
                      true);
              REQUIRE(mesh->create_particles_tractions(mfunction, -1, 1, 0.5) ==
                      true);
              REQUIRE(mesh->create_particles_tractions(mfunction, 5, 1, 0.5) ==
                      false);
              REQUIRE(mesh->create_particles_tractions(mfunction, -5, 1, 0.5) ==
                      false);

              // Locate particles in a mesh
              auto particles = mesh->locate_particles_mesh();
              REQUIRE(particles.size() == 0);
              mesh->iterate_over_particles(
                  std::bind(&mpm::ParticleBase<Dim>::compute_shapefn,
                            std::placeholders::_1));

              // Compute volume
              mesh->iterate_over_particles(
                  std::bind(&mpm::ParticleBase<Dim>::compute_volume,
                            std::placeholders::_1));

              mesh->apply_traction_on_particles(10);
            }

            // Test assign particles stresses
            SECTION("Check assign particles stresses") {
              // Vector of particle stresses
              std::vector<Eigen::Matrix<double, 6, 1>> particles_stresses;

              REQUIRE(mesh->nparticles() == 16);

              // Stresses
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.0));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.1));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.2));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.4));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.3));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.5));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.6));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.7));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.8));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.9));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.10));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.11));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.12));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.13));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(-0.14));
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.15));

              REQUIRE(mesh->assign_particles_stresses(particles_stresses) ==
                      true);
              // When stresses fail
              particles_stresses.emplace_back(
                  Eigen::Matrix<double, 6, 1>::Constant(0.16));
              REQUIRE(mesh->assign_particles_stresses(particles_stresses) ==
                      false);
              unsigned id = 1;
              auto mesh_fail = std::make_shared<mpm::Mesh<Dim>>(id);
              REQUIRE(mesh_fail->assign_particles_stresses(
                          particles_stresses) == false);
            }

            // Test assign particles velocity constraints
            SECTION("Check assign particles velocity constraints") {
              tsl::robin_map<mpm::Index, std::vector<mpm::Index>> particle_sets;
              particle_sets[0] = std::vector<mpm::Index>{0};
              particle_sets[1] = std::vector<mpm::Index>{1};
              particle_sets[2] = std::vector<mpm::Index>{2};
              particle_sets[3] = std::vector<mpm::Index>{3};

              REQUIRE(mesh->create_particle_sets(particle_sets, true) == true);

              REQUIRE(mesh->nparticles() == 16);

              int set_id = 0;
              int dir = 0;
              double constraint = 10.5;
              // Add velocity constraint to mesh
              auto velocity_constraint =
                  std::make_shared<mpm::VelocityConstraint>(set_id, dir,
                                                            constraint);
              REQUIRE(mesh->create_particle_velocity_constraint(
                          set_id, velocity_constraint) == true);

              // Add velocity constraint to all nodes in mesh
              velocity_constraint = std::make_shared<mpm::VelocityConstraint>(
                  -1, dir, constraint);
              REQUIRE(mesh->create_particle_velocity_constraint(
                          set_id, velocity_constraint) == true);

              // When constraints fail
              dir = 3;
              // Add velocity constraint to mesh
              velocity_constraint = std::make_shared<mpm::VelocityConstraint>(
                  set_id, dir, constraint);
              REQUIRE(mesh->create_particle_velocity_constraint(
                          set_id, velocity_constraint) == false);

              mesh->apply_particle_velocity_constraints();
            }
          }
        }

        // Test assign absorbing constraint
        SECTION("Check assign absorbing constraints") {
          tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
          node_sets[0] = std::vector<mpm::Index>{0, 2};
          node_sets[1] = std::vector<mpm::Index>{1, 3};
          node_sets[2] = std::vector<mpm::Index>{};

          REQUIRE(mesh->create_node_sets(node_sets, true) == true);

          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          // Assign nodal properties
          auto nodal_properties = std::make_shared<mpm::NodalProperties>();
          nodal_properties->create_property("density", 1, 1);
          nodal_properties->create_property("wave_velocities", Dim, 1);
          nodal_properties->create_property("displacements", Dim, 1);
          for (double i = 0; i < 4; ++i) {
            REQUIRE_NOTHROW(
                mesh->node(i)->initialise_property_handle(0, nodal_properties));
            mesh->node(i)->append_material_id(0);
          }

          int set_id = 0;
          unsigned dir = 0;
          double delta = 1;
          double h_min = 1;
          double a = 1;
          double b = 1;
          mpm::Position pos = mpm::Position::Edge;

          // Add absorbing constraint to mesh
          auto absorbing_constraint =
              std::make_shared<mpm::AbsorbingConstraint>(set_id, dir, delta,
                                                         h_min, a, b, pos);
          REQUIRE(constraints->assign_nodal_absorbing_constraint(
                      set_id, absorbing_constraint) == true);

          pos = mpm::Position::Corner;
          // Add absorbing constraint to mesh
          absorbing_constraint = std::make_shared<mpm::AbsorbingConstraint>(
              set_id, dir, delta, h_min, a, b, pos);
          REQUIRE(constraints->assign_nodal_absorbing_constraint(
                      set_id, absorbing_constraint) == true);

          pos = mpm::Position::Face;
          // Add absorbing constraint to mesh
          absorbing_constraint = std::make_shared<mpm::AbsorbingConstraint>(
              set_id, dir, delta, h_min, a, b, pos);
          REQUIRE(constraints->assign_nodal_absorbing_constraint(
                      set_id, absorbing_constraint) == true);

          // When constraints fail: invalid absorbing boundary position
          pos = mpm::Position::None;
          // Add absorbing constraint to mesh
          absorbing_constraint = std::make_shared<mpm::AbsorbingConstraint>(
              set_id, dir, delta, h_min, a, b, pos);
          REQUIRE(constraints->assign_nodal_absorbing_constraint(
                      set_id, absorbing_constraint) == false);
        }

        // Test assign non-conforming traction constraint
        SECTION("Check assign non-conforming traction constraint to nodes") {
          // Vector of particle coordinates
          std::vector<Eigen::Matrix<double, Dim, 1>> coordinates;
          coordinates.clear();

          // Particle coordinates
          Eigen::Matrix<double, Dim, 1> particle;

          // Cell 0
          // Particle 0
          particle << 0.125, 0.125, 0.125;
          coordinates.emplace_back(particle);
          // Particle 1
          particle << 0.375, 0.125, 0.125;
          coordinates.emplace_back(particle);
          // Particle 2
          particle << 0.375, 0.375, 0.125;
          coordinates.emplace_back(particle);
          // Particle 3
          particle << 0.125, 0.375, 0.125;
          coordinates.emplace_back(particle);
          // Particle 4
          particle << 0.125, 0.125, 0.375;
          coordinates.emplace_back(particle);
          // Particle 5
          particle << 0.375, 0.125, 0.375;
          coordinates.emplace_back(particle);
          // Particle 6
          particle << 0.375, 0.375, 0.375;
          coordinates.emplace_back(particle);
          // Particle 7
          particle << 0.125, 0.375, 0.375;
          coordinates.emplace_back(particle);

          // Initialise material models in mesh
          mesh->initialise_material_models(materials);

          SECTION("Check addition of particles to mesh") {
            // Particle type 2D
            const std::string particle_type = "P3D";
            // Create particles from file
            bool status = mesh->create_particles(particle_type, coordinates,
                                                 mids, 0, false);

            // Check if mesh has added particles
            REQUIRE(mesh->nparticles() == coordinates.size());

            // Vector of particle cells
            std::vector<std::array<mpm::Index, 2>> particles_cells;
            // Particle cells
            particles_cells.emplace_back(std::array<mpm::Index, 2>({0, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({1, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({2, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({3, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({4, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({5, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({6, 0}));
            particles_cells.emplace_back(std::array<mpm::Index, 2>({7, 0}));

            REQUIRE(mesh->assign_particles_cells(particles_cells) == true);

            // Locate particles
            auto missing_particles = mesh->locate_particles_mesh();
            REQUIRE(missing_particles.size() == 0);

            REQUIRE(mesh->particles_cells().size() == mesh->nparticles());

            mesh->iterate_over_particles(
                std::bind(&mpm::ParticleBase<Dim>::compute_shapefn,
                          std::placeholders::_1));
          }

          // Add cell neighbors
          auto cells = mesh->cells();
          unsigned count = 1;
          for (auto citr = cells.cbegin(); citr != cells.cend(); ++citr) {
            REQUIRE((*citr)->add_neighbour(count) == true);
            count -= 1;
          }

          // Update nodal mass
          auto nodes = mesh->nodes();
          for (auto nitr = nodes.cbegin(); nitr != nodes.cend(); ++nitr) {
            if ((*nitr) != nullptr) {
              (*nitr)->update_mass(false, 0, 100.);
            }
          }

          // Define bounding box
          std::vector<double> bounding_box{-5., 5., -5., 5., -5., 5.};

          // Constraint for hydrostatic case
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., true, true, nullptr, 0.) ==
                  true);
          // Constraint for hydrostatic case (above datum)
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 0., 1000., -10., true, true, nullptr, 0.) ==
                  true);
          // Constraint for hydrostatic case (outside bounding box)
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., true, false, nullptr,
                      0.) == true);
          // Constraint for constant case
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., false, true, mfunction,
                      100.) == true);
          // Constraint for constant case (no mathfunction)
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., false, true, nullptr,
                      100.) == true);

          bounding_box.at(0) = 5.;
          bounding_box.at(1) = -5.;
          // Constraint for hydrostatic case (outside bounding box)
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., true, true, nullptr, 0.) ==
                  true);
          // Constraint for hydrostatic case (outside bounding box)
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., true, false, nullptr,
                      0.) == true);

          // Constraint fails due to pressure = 0. && !hydrostatic
          REQUIRE(mesh->create_nonconforming_traction_constraint(
                      bounding_box, 2., 1000., -10., false, true, mfunction,
                      0.) == false);

          // Apply constraint
          mesh->apply_nonconforming_traction_constraint(10.);
        }

        // Test assign acceleration constraints to nodes
        SECTION("Check assign acceleration constraints to nodes") {
          tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
          node_sets[0] = std::vector<mpm::Index>{0, 2};
          node_sets[1] = std::vector<mpm::Index>{1, 3};
          node_sets[2] = std::vector<mpm::Index>{};

          REQUIRE(mesh->create_node_sets(node_sets, true) == true);

          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          int set_id = 0;
          int dir = 0;
          double constraint = 1.0;
          // Add acceleration constraint to mesh
          auto acceleration_constraint =
              std::make_shared<mpm::AccelerationConstraint>(set_id, mfunction,
                                                            dir, constraint);
          REQUIRE(constraints->assign_nodal_acceleration_constraint(
                      set_id, acceleration_constraint) == true);

          set_id = 1;
          dir = 1;
          constraint = -1.0;
          // Add acceleration constraint to mesh
          acceleration_constraint =
              std::make_shared<mpm::AccelerationConstraint>(set_id, mfunction,
                                                            dir, constraint);
          REQUIRE(constraints->assign_nodal_acceleration_constraint(
                      set_id, acceleration_constraint) == true);

          // When constraints fail: invalid direction
          dir = 3;
          // Add acceleration constraint to mesh
          acceleration_constraint =
              std::make_shared<mpm::AccelerationConstraint>(set_id, mfunction,
                                                            dir, constraint);
          REQUIRE(constraints->assign_nodal_acceleration_constraint(
                      set_id, acceleration_constraint) == false);

          // When constraints fail: empty node set
          dir = 0;
          set_id = 2;
          // Add acceleration constraint to mesh
          acceleration_constraint =
              std::make_shared<mpm::AccelerationConstraint>(set_id, mfunction,
                                                            dir, constraint);
          REQUIRE(constraints->assign_nodal_acceleration_constraint(
                      set_id, acceleration_constraint) == false);
        }

        // Test assign velocity constraints to nodes
        SECTION("Check assign velocity constraints to nodes") {
          tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
          node_sets[0] = std::vector<mpm::Index>{0, 2};
          node_sets[1] = std::vector<mpm::Index>{1, 3};

          REQUIRE(mesh->create_node_sets(node_sets, true) == true);

          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          int set_id = 0;
          int dir = 0;
          double constraint = 10.5;
          // Add velocity constraint to mesh
          auto velocity_constraint = std::make_shared<mpm::VelocityConstraint>(
              set_id, dir, constraint);
          REQUIRE(constraints->assign_nodal_velocity_constraint(
                      set_id, velocity_constraint) == true);

          set_id = 1;
          dir = 1;
          constraint = -12.5;
          // Add velocity constraint to mesh
          velocity_constraint = std::make_shared<mpm::VelocityConstraint>(
              set_id, dir, constraint);
          REQUIRE(constraints->assign_nodal_velocity_constraint(
                      set_id, velocity_constraint) == true);

          // Add velocity constraint to all nodes in mesh
          velocity_constraint =
              std::make_shared<mpm::VelocityConstraint>(-1, dir, constraint);
          REQUIRE(constraints->assign_nodal_velocity_constraint(
                      set_id, velocity_constraint) == true);

          // When constraints fail
          dir = 3;
          // Add velocity constraint to mesh
          velocity_constraint = std::make_shared<mpm::VelocityConstraint>(
              set_id, dir, constraint);
          REQUIRE(constraints->assign_nodal_velocity_constraint(
                      set_id, velocity_constraint) == false);
        }

        SECTION("Check assign pressure constraints to nodes") {
          tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
          node_sets[0] = std::vector<mpm::Index>{0, 2};
          node_sets[1] = std::vector<mpm::Index>{1, 3};

          REQUIRE(mesh->create_node_sets(node_sets, true) == true);

          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          int set_id = 0;
          double pressure = 500.2;
          // Add pressure constraint to mesh
          REQUIRE(constraints->assign_nodal_pressure_constraint(
                      mfunction, set_id, 0, pressure) == true);

          // Add pressure constraint to all nodes in mesh
          REQUIRE(constraints->assign_nodal_pressure_constraint(
                      mfunction, -1, 0, pressure) == true);
        }

        SECTION("Check assign friction constraints to nodes") {
          tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
          node_sets[0] = std::vector<mpm::Index>{0, 2};
          node_sets[1] = std::vector<mpm::Index>{1, 3};

          REQUIRE(mesh->create_node_sets(node_sets, true) == true);

          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          int set_id = 0;
          int dir = 0;
          int sign_n = 1;
          double friction = 0.5;
          // Add friction constraint to mesh
          auto friction_constraint = std::make_shared<mpm::FrictionConstraint>(
              set_id, dir, sign_n, friction);
          REQUIRE(constraints->assign_nodal_frictional_constraint(
                      set_id, friction_constraint) == true);

          set_id = 1;
          dir = 1;
          sign_n = -1;
          friction = -0.25;
          // Add friction constraint to mesh
          friction_constraint = std::make_shared<mpm::FrictionConstraint>(
              set_id, dir, sign_n, friction);
          REQUIRE(constraints->assign_nodal_frictional_constraint(
                      set_id, friction_constraint) == true);

          // Add friction constraint to all nodes in mesh
          friction_constraint = std::make_shared<mpm::FrictionConstraint>(
              -1, dir, sign_n, friction);
          REQUIRE(constraints->assign_nodal_frictional_constraint(
                      set_id, friction_constraint) == true);

          // When constraints fail
          dir = 3;
          // Add friction constraint to mesh
          friction_constraint = std::make_shared<mpm::FrictionConstraint>(
              set_id, dir, sign_n, friction);
          REQUIRE(constraints->assign_nodal_frictional_constraint(
                      set_id, friction_constraint) == false);
        }

        SECTION("Check assign adhesion constraints to nodes") {
          tsl::robin_map<mpm::Index, std::vector<mpm::Index>> node_sets;
          node_sets[0] = std::vector<mpm::Index>{0, 2};
          node_sets[1] = std::vector<mpm::Index>{1, 3};

          REQUIRE(mesh->create_node_sets(node_sets, true) == true);

          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          int set_id = 0;
          int dir = 0;
          int sign_n = -1;
          double adhesion = 1000;
          double h_min = 0.25;
          int nposition = 1;
          // Add adhesion constraint to mesh
          auto adhesion_constraint = std::make_shared<mpm::AdhesionConstraint>(
              set_id, dir, sign_n, adhesion, h_min, nposition);
          REQUIRE(constraints->assign_nodal_adhesional_constraint(
                      set_id, adhesion_constraint) == true);

          set_id = 1;
          dir = 2;
          sign_n = -1;
          adhesion = 1000;
          h_min = 0.25;
          nposition = 2;
          // Add adhesion constraint to mesh
          adhesion_constraint = std::make_shared<mpm::AdhesionConstraint>(
              set_id, dir, sign_n, adhesion, h_min, nposition);
          REQUIRE(constraints->assign_nodal_adhesional_constraint(
                      set_id, adhesion_constraint) == true);

          // Add adhesion constraint to all nodes in mesh
          adhesion_constraint = std::make_shared<mpm::AdhesionConstraint>(
              1, dir, sign_n, adhesion, h_min, nposition);
          REQUIRE(constraints->assign_nodal_adhesional_constraint(
                      set_id, adhesion_constraint) == true);

          // When constraints fail
          dir = 3;
          // Add adhesion constraint to mesh
          adhesion_constraint = std::make_shared<mpm::AdhesionConstraint>(
              set_id, dir, sign_n, adhesion, h_min, nposition);
          REQUIRE(constraints->assign_nodal_adhesional_constraint(
                      set_id, adhesion_constraint) == false);
        }

        // Test assign acceleration constraints to nodes
        SECTION("Check assign acceleration constraints to nodes") {
          // Vector of particle coordinates
          std::vector<std::tuple<mpm::Index, unsigned, double>>
              acceleration_constraints;
          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);
          // Constraint
          acceleration_constraints.emplace_back(std::make_tuple(0, 0, 1.5));
          acceleration_constraints.emplace_back(std::make_tuple(1, 1, -1.5));
          acceleration_constraints.emplace_back(std::make_tuple(2, 0, -1.5));
          acceleration_constraints.emplace_back(std::make_tuple(3, 1, 0.0));

          REQUIRE(constraints->assign_nodal_acceleration_constraints(
                      acceleration_constraints) == true);

          // When constraints fail: invalid direction
          acceleration_constraints.emplace_back(std::make_tuple(3, 3, 0.0));
          REQUIRE(constraints->assign_nodal_acceleration_constraints(
                      acceleration_constraints) == false);
        }

        // Test assign velocity constraints to nodes
        SECTION("Check assign velocity constraints to nodes") {
          // Vector of particle coordinates
          std::vector<std::tuple<mpm::Index, unsigned, double>>
              velocity_constraints;
          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);
          // Constraint
          velocity_constraints.emplace_back(std::make_tuple(0, 0, 10.5));
          velocity_constraints.emplace_back(std::make_tuple(1, 1, -10.5));
          velocity_constraints.emplace_back(std::make_tuple(2, 0, -12.5));
          velocity_constraints.emplace_back(std::make_tuple(3, 1, 0.0));

          REQUIRE(constraints->assign_nodal_velocity_constraints(
                      velocity_constraints) == true);
          // When constraints fail
          velocity_constraints.emplace_back(std::make_tuple(3, 3, 0.0));
          REQUIRE(constraints->assign_nodal_velocity_constraints(
                      velocity_constraints) == false);
        }

        // Test assign friction constraints to nodes
        SECTION("Check assign friction constraints to nodes") {
          // Vector of particle coordinates
          std::vector<std::tuple<mpm::Index, unsigned, int, double>>
              friction_constraints;
          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);
          // Constraint
          friction_constraints.emplace_back(std::make_tuple(0, 0, 1, 0.5));
          friction_constraints.emplace_back(std::make_tuple(1, 1, -1, 0.5));
          friction_constraints.emplace_back(std::make_tuple(2, 0, 1, 0.25));
          friction_constraints.emplace_back(std::make_tuple(3, 1, -1, 0.0));

          REQUIRE(constraints->assign_nodal_friction_constraints(
                      friction_constraints) == true);
          // When constraints fail
          friction_constraints.emplace_back(std::make_tuple(3, 3, -1, 0.0));
          REQUIRE(constraints->assign_nodal_friction_constraints(
                      friction_constraints) == false);
        }

        // Test assign adhesion constraints to nodes
        SECTION("Check assign adhesion constraints to nodes") {
          // Vector of particle coordinates
          std::vector<
              std::tuple<mpm::Index, unsigned, int, double, double, int>>
              adhesion_constraints;
          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);
          // Constraint
          adhesion_constraints.emplace_back(
              std::make_tuple(0, 0, -1, 100, 0.25, 1));
          adhesion_constraints.emplace_back(
              std::make_tuple(1, 0, -1, 100, 0.25, 2));
          adhesion_constraints.emplace_back(
              std::make_tuple(2, 1, -1, 100, 0.25, 3));
          adhesion_constraints.emplace_back(
              std::make_tuple(3, 2, -1, 100, 0.25, 3));

          REQUIRE(constraints->assign_nodal_adhesion_constraints(
                      adhesion_constraints) == true);
          // When constraints fail
          adhesion_constraints.emplace_back(
              std::make_tuple(3, 3, -1, 100, 0.25, 3));
          REQUIRE(constraints->assign_nodal_adhesion_constraints(
                      adhesion_constraints) == false);
        }

        // Test assign absorbing constraints to nodes
        SECTION("Check assign absorbing constraints to nodes") {
          // Vector of particle coordinates
          std::vector<std::tuple<mpm::Index, unsigned, double, double, double,
                                 double, mpm::Position>>
              absorbing_constraints;
          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);

          // Assign nodal properties
          auto nodal_properties = std::make_shared<mpm::NodalProperties>();
          nodal_properties->create_property("density", 1, 1);
          nodal_properties->create_property("wave_velocities", Dim, 1);
          nodal_properties->create_property("displacements", Dim, 1);
          for (double i = 0; i < 4; ++i) {
            REQUIRE_NOTHROW(
                mesh->node(i)->initialise_property_handle(0, nodal_properties));
          }

          // Constraint
          absorbing_constraints.emplace_back(
              std::make_tuple(0, 0, 1, 3, 2, 2, mpm::Position::Edge));
          absorbing_constraints.emplace_back(
              std::make_tuple(1, 1, 2, 4, 1, 1, mpm::Position::Corner));
          absorbing_constraints.emplace_back(
              std::make_tuple(2, 0, 1, 1, 1, 1, mpm::Position::Face));

          REQUIRE(constraints->assign_nodal_absorbing_constraints(
                      absorbing_constraints) == true);

          // When constraints fail: invalid direction
          absorbing_constraints.clear();
          absorbing_constraints.emplace_back(
              std::make_tuple(3, 3, 3, 2, 3, 3, mpm::Position::Edge));
          REQUIRE(constraints->assign_nodal_absorbing_constraints(
                      absorbing_constraints) == false);

          // When constraints fail: invalid delta
          absorbing_constraints.clear();
          absorbing_constraints.emplace_back(
              std::make_tuple(3, 1, 1, 3, 1, 1, mpm::Position::Edge));
          REQUIRE(constraints->assign_nodal_absorbing_constraints(
                      absorbing_constraints) == false);

          // When constraints fail: invalid position
          absorbing_constraints.clear();
          absorbing_constraints.emplace_back(
              std::make_tuple(0, 0, 1, 3, 2, 2, mpm::Position::None));
          REQUIRE(constraints->assign_nodal_absorbing_constraints(
                      absorbing_constraints) == false);
        }

        // Test assign pressure constraints to nodes
        SECTION("Check assign pressure constraints to nodes") {
          // Vector of pressure constraints
          std::vector<std::tuple<mpm::Index, double>> pressure_constraints;
          //! Constraints object
          auto constraints = std::make_shared<mpm::Constraints<Dim>>(mesh);
          // Constraint
          pressure_constraints.emplace_back(std::make_tuple(0, 500.5));
          pressure_constraints.emplace_back(std::make_tuple(1, 210.5));
          pressure_constraints.emplace_back(std::make_tuple(2, 320.2));
          pressure_constraints.emplace_back(std::make_tuple(3, 0.0));

          REQUIRE(constraints->assign_nodal_pressure_constraints(
                      0, pressure_constraints) == true);
          REQUIRE(constraints->assign_nodal_pressure_constraints(
                      1, pressure_constraints) == false);
        }

        // Test assign nodes concentrated_forces
        SECTION("Check assign nodes concentrated_forces") {
          // Vector of node coordinates
          std::vector<std::tuple<mpm::Index, unsigned, double>>
              nodes_concentrated_forces;
          // Concentrated_Forces
          nodes_concentrated_forces.emplace_back(std::make_tuple(0, 0, 10.5));
          nodes_concentrated_forces.emplace_back(std::make_tuple(1, 1, -10.5));
          nodes_concentrated_forces.emplace_back(std::make_tuple(2, 0, -12.5));
          nodes_concentrated_forces.emplace_back(std::make_tuple(3, 1, 0.0));

          REQUIRE(mesh->nnodes() == 12);

          REQUIRE(mesh->assign_nodal_concentrated_forces(
                      nodes_concentrated_forces) == true);
          // When concentrated_forces fail
          nodes_concentrated_forces.emplace_back(std::make_tuple(3, 3, 0.0));
          REQUIRE(mesh->assign_nodal_concentrated_forces(
                      nodes_concentrated_forces) == false);
          nodes_concentrated_forces.emplace_back(std::make_tuple(300, 0, 0.0));
          REQUIRE(mesh->assign_nodal_concentrated_forces(
                      nodes_concentrated_forces) == false);
        }

        // Test assign rotation matrices to nodes
        SECTION("Check assign rotation matrices to nodes") {
          // Map of nodal id and euler angles
          std::map<mpm::Index, Eigen::Matrix<double, Dim, 1>> euler_angles;
          // Insert euler angles and node id into map
          // Node 0 with Euler angles of 10, 20 and 30 deg
          euler_angles.emplace(std::make_pair(
              0, (Eigen::Matrix<double, Dim, 1>() << 10. * M_PI / 180,
                  20. * M_PI / 180, 30. * M_PI / 180)
                     .finished()));
          // Node 1 with Euler angles of 40, 50 and 60 deg
          euler_angles.emplace(std::make_pair(
              1, (Eigen::Matrix<double, Dim, 1>() << 40. * M_PI / 180,
                  50. * M_PI / 180, 60. * M_PI / 180)
                     .finished()));
          // Node 2 with Euler angles of 70, 80 and 90 deg
          euler_angles.emplace(std::make_pair(
              2, (Eigen::Matrix<double, Dim, 1>() << 70. * M_PI / 180,
                  80. * M_PI / 180, 90. * M_PI / 180)
                     .finished()));
          // Node 3 with Euler angles of 100, 110 and 120 deg
          euler_angles.emplace(std::make_pair(
              3, (Eigen::Matrix<double, Dim, 1>() << 100. * M_PI / 180,
                  110. * M_PI / 180, 120. * M_PI / 180)
                     .finished()));

          // Check compute and assign rotation matrix
          REQUIRE(mesh->compute_nodal_rotation_matrices(euler_angles) == true);

          // Check for failure when missing node id
          // Node 100 (non-existent) with Euler angles of 130, 140 and 150 deg
          euler_angles.emplace(std::make_pair(
              100, (Eigen::Matrix<double, Dim, 1>() << 130. * M_PI / 180,
                    140. * M_PI / 180, 150. * M_PI / 180)
                       .finished()));
          REQUIRE(mesh->compute_nodal_rotation_matrices(euler_angles) == false);

          // Check for failure of empty input
          std::map<mpm::Index, Eigen::Matrix<double, Dim, 1>>
              empty_euler_angles;
          REQUIRE(mesh->compute_nodal_rotation_matrices(empty_euler_angles) ==
                  false);

          // Check for failure when no nodes are assigned
          auto mesh_fail = std::make_shared<mpm::Mesh<Dim>>(1);
          REQUIRE(mesh_fail->compute_nodal_rotation_matrices(euler_angles) ==
                  false);
        }
      }
    }
  }

  //! Check if nodal properties is initialised
  SECTION("Check nodal properties initialisation") {
    // Create the different meshes
    std::shared_ptr<mpm::Mesh<Dim>> mesh = std::make_shared<mpm::Mesh<Dim>>(0);

    // Define nodes
    Eigen::Vector3d coords;
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

    // Add nodes 0 to 3 to the mesh
    REQUIRE(mesh->add_node(node0) == true);
    REQUIRE(mesh->add_node(node1) == true);
    REQUIRE(mesh->add_node(node2) == true);
    REQUIRE(mesh->add_node(node3) == true);
    REQUIRE(mesh->add_node(node4) == true);
    REQUIRE(mesh->add_node(node5) == true);
    REQUIRE(mesh->add_node(node6) == true);
    REQUIRE(mesh->add_node(node7) == true);

    // Initialise material models
    mesh->initialise_material_models(materials);

    // Check nodal properties creation
    REQUIRE_NOTHROW(mesh->create_nodal_properties());

    // Check nodal properties initialisation
    REQUIRE_NOTHROW(mesh->initialise_nodal_properties());
  }
}
