#include <cmath>
#include <limits>
#include <memory>

#include "Eigen/Dense"
#include "catch.hpp"

#include "function_base.h"
#include "geometry.h"
#include "node.h"

// Check node class for 1D case
TEST_CASE("Node is checked for 1D case", "[node][1D]") {
  const unsigned Dim = 1;
  const unsigned Dof = 1;
  const unsigned Nphases = 1;
  const unsigned Nphase = 0;
  Eigen::Matrix<double, 1, 1> coords;
  coords.setZero();

  // Check for id = 0
  SECTION("Node id is zero") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == 0);
  }

  // Check for id is a positive value
  SECTION("Node id is positive") {
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == std::numeric_limits<mpm::Index>::max());
  }

  // Check for degrees of freedom
  SECTION("Check degrees of freedom") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->dof() == 1);
  }

  // Check status
  SECTION("Check status") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->status() == false);
    node->assign_status(true);
    REQUIRE(node->status() == true);
  }

  SECTION("Boundary ghost id") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    node->ghost_id(5);
    REQUIRE(node->ghost_id() == 5);
  }

  // Check MPI Rank
  SECTION("Check MPI Rank") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == 0);

    // Assign MPI ranks
    node->mpi_rank(0);
    node->mpi_rank(0);
    node->mpi_rank(1);

    std::set<unsigned> ranks = node->mpi_ranks();
    REQUIRE(ranks.size() == 2);
    std::vector<unsigned> mpi_ranks = {0, 1};
    unsigned i = 0;
    for (auto it = ranks.begin(); it != ranks.end(); ++it, ++i)
      REQUIRE(*it == mpi_ranks.at(i));
  }

  // Test coordinates function
  SECTION("coordinates function is checked") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;

    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);

    // Check for coordinates being zero
    auto coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    // Check for negative value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = -1. * std::numeric_limits<double>::max();
    node->assign_coordinates(coords);
    coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);

    // Check for positive value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = std::numeric_limits<double>::max();
    node->assign_coordinates(coords);
    coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);
  }

  SECTION("Check nodal properties") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);

    // Check mass
    REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
    double mass = 100.5;
    // Update mass to 100.5
    REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.5).epsilon(Tolerance));
    // Update mass to 201
    REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(201.0).epsilon(Tolerance));
    // Assign mass to 100
    mass = 100.;
    REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.0).epsilon(Tolerance));

    SECTION("Check nodal pressure") {
      // Check pressure
      REQUIRE(node->pressure(Nphase) == Approx(0.0).epsilon(Tolerance));
      double pressure = 1000.7;
      // Update pressure to 1000.7
      REQUIRE_NOTHROW(node->update_mass_pressure(Nphase, mass * pressure));
      REQUIRE(node->pressure(Nphase) == Approx(1000.7).epsilon(Tolerance));
      // Update pressure to 2001.4
      REQUIRE_NOTHROW(node->update_mass_pressure(Nphase, mass * pressure));
      REQUIRE(node->pressure(Nphase) == Approx(2001.4).epsilon(Tolerance));
      // Assign pressure to 1000
      pressure = 1000.;
      node->assign_pressure(Nphase, pressure);
      REQUIRE(node->pressure(Nphase) == Approx(1000.0).epsilon(Tolerance));
      // Assign mass to 0
      mass = 0.;
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
      // Try to update pressure to 2000, should throw and keep to 1000.
      node->assign_pressure(Nphase, pressure);
      REQUIRE(node->pressure(Nphase) == Approx(1000.0).epsilon(Tolerance));
      // Check pressure constraints
      SECTION("Check nodal pressure constraints") {
        // Check assign pressure constraint
        REQUIRE(node->assign_pressure_constraint(mpm::NodePhase::NSolid, 8000,
                                                 nullptr) == true);
        // Check apply pressure constraint
        REQUIRE_NOTHROW(
            node->apply_pressure_constraint(mpm::NodePhase::NSolid));
        // Check pressure
        REQUIRE(node->pressure(mpm::NodePhase::NSolid) ==
                Approx(8000).epsilon(Tolerance));
      }
    }

    SECTION("Check external force") {
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10.;

      // Check current external force is zero
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));

      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_external_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Update force to 20.0
      REQUIRE_NOTHROW(node->update_external_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(20.).epsilon(Tolerance));

      // Assign force as 10.0
      REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      SECTION("Check concentrated force") {
        // Set external force to zero
        force.setZero();
        REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));

        // concentrated force
        std::shared_ptr<mpm::FunctionBase> ffunction = nullptr;
        double concentrated_force = 65.32;
        const unsigned Direction = 0;
        // Check external force
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->external_force(Nphase)(i) ==
                  Approx(0.).epsilon(Tolerance));

        REQUIRE(node->assign_concentrated_force(
                    Nphase, Direction, concentrated_force, ffunction) == true);

        double current_time = 0.0;
        node->apply_concentrated_force(Nphase, current_time);

        for (unsigned i = 0; i < Dim; ++i) {
          if (i == Direction)
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(concentrated_force).epsilon(Tolerance));
          else
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(0.).epsilon(Tolerance));
        }

        // Check for incorrect direction / phase
        const unsigned wrong_dir = 4;
        REQUIRE(node->assign_concentrated_force(
                    Nphase, wrong_dir, concentrated_force, ffunction) == false);

        // Check again to ensure value hasn't been updated
        for (unsigned i = 0; i < Dim; ++i) {
          if (i == Direction)
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(concentrated_force).epsilon(Tolerance));
          else
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(0.).epsilon(Tolerance));
        }
      }
    }

    SECTION("Check internal force") {
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10.;

      // Check current internal force is zero
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));

      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_internal_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Update force to 20.0
      REQUIRE_NOTHROW(node->update_internal_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(20.).epsilon(Tolerance));

      // Assign force as 10.0
      REQUIRE_NOTHROW(node->update_internal_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));
    }

    SECTION("Check compute acceleration and velocity") {
      // Time step
      const double dt = 0.1;

      // Nodal mass
      double mass = 100.;
      // Update mass to 100.5
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(mass).epsilon(Tolerance));

      // Check internal force
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10. * i;
      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_internal_force(false, Nphase, force));
      // Internal force
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(force(i)).epsilon(Tolerance));

      // External force
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 5. * i;
      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(force(i)).epsilon(Tolerance));

      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      for (unsigned i = 0; i < force.size(); ++i) force(i) = 15. * i;

      // Check acceleration
      Eigen::Matrix<double, Dim, 1> acceleration = force / mass;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Check velocity
      Eigen::Matrix<double, Dim, 1> velocity = force / mass * dt;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Apply friction constraints
      REQUIRE(node->assign_friction_constraint(0, 1., 0.5) == true);
      // Apply friction constraints
      REQUIRE(node->assign_friction_constraint(-1, 1., 0.5) == false);
      REQUIRE(node->assign_friction_constraint(3, 1., 0.5) == false);

      // Apply cohesion constraints
      REQUIRE(node->assign_cohesion_constraint(0, -1., 1000, 0.25, 2) == true);
      // Apply cohesion constraints
      REQUIRE(node->assign_cohesion_constraint(-1, -1., 1000, 0.25, 2) ==
              false);
      REQUIRE(node->assign_cohesion_constraint(3, -1., 1000, 0.25, 2) == false);

      // Test acceleration with constraints
      acceleration[0] = 0.5 * acceleration[0];
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Apply cundall damping when calculating acceleration
      REQUIRE(node->compute_acceleration_velocity_cundall(Nphase, dt, 0.05) ==
              true);

      // Test acceleration with cundall damping
      acceleration[0] = 0.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Apply velocity constraints
      REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      // Test velocity with constraints
      velocity[0] = 10.5;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Test acceleration with constraints
      acceleration[0] = 0.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Exception check when mass is zero
      mass = 0.;
      // Update mass to 0.
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(mass).epsilon(Tolerance));
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == false);
    }

    SECTION("Check momentum and velocity") {
      // Check momentum
      Eigen::Matrix<double, Dim, 1> momentum;
      for (unsigned i = 0; i < momentum.size(); ++i) momentum(i) = 10.;

      // Check initial momentum
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      // Check update momentum to 10
      REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(10.).epsilon(Tolerance));

      // Check update momentum to 20
      REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(20.).epsilon(Tolerance));

      // Check assign momentum to 10
      REQUIRE_NOTHROW(node->update_momentum(false, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(10.).epsilon(Tolerance));

      // Check zero velocity
      for (unsigned i = 0; i < Dim; ++i)
        REQUIRE(node->velocity(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      // Check mass
      double mass = 0.;
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
      // Compute and check velocity this should throw zero mass
      node->compute_velocity();

      mass = 100.;
      // Update mass to 100
      REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(100.).epsilon(Tolerance));

      // Compute and check velocity
      node->compute_velocity();
      for (unsigned i = 0; i < Dim; ++i)
        REQUIRE(node->velocity(Nphase)(i) == Approx(0.1).epsilon(Tolerance));

      // Apply velocity constraints
      REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
      // Check out of bounds condition
      REQUIRE(node->assign_velocity_constraint(1, 0) == false);

      // Check velocity before constraints
      Eigen::Matrix<double, Dim, 1> velocity;
      velocity << 0.1;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Apply constraints
      node->apply_velocity_constraints();

      // Check apply constraints
      velocity << 10.5;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));
    }

    SECTION("Check acceleration") {
      // Check acceleration
      Eigen::Matrix<double, Dim, 1> acceleration;
      for (unsigned i = 0; i < acceleration.size(); ++i) acceleration(i) = 5.;

      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      REQUIRE_NOTHROW(node->update_acceleration(true, Nphase, acceleration));
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) == Approx(5.).epsilon(Tolerance));

      // Apply velocity constraints
      REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
      // Check out of bounds condition
      REQUIRE(node->assign_velocity_constraint(1, 12.5) == false);

      // Check acceleration before constraints
      acceleration.resize(Dim);
      acceleration << 5.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Apply constraints
      node->apply_velocity_constraints();

      // Check apply constraints
      acceleration << 0.0;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));
    }

    SECTION("Check node material ids") {
      // Add material to nodes
      node->append_material_id(0);
      node->append_material_id(1);
      node->append_material_id(4);
      node->append_material_id(0);
      node->append_material_id(2);

      // Check size of material_ids
      REQUIRE(node->material_ids().size() == 4);

      // Check elements of material_ids
      std::vector<unsigned> material_ids = {0, 1, 2, 4};
      auto mat_ids = node->material_ids();
      unsigned i = 0;
      for (auto mitr = mat_ids.begin(); mitr != mat_ids.end(); ++mitr, ++i)
        REQUIRE(*mitr == material_ids.at(i));
    }
  }
}

// \brief Check node class for 2D case
TEST_CASE("Node is checked for 2D case", "[node][2D]") {
  const unsigned Dim = 2;
  const unsigned Dof = 2;
  const unsigned Nphases = 1;
  const unsigned Nphase = 0;
  Eigen::Vector2d coords;
  coords.setZero();

  // Check for id = 0
  SECTION("Node id is zero") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == 0);
  }

  // Check for id is a positive value
  SECTION("Node id is positive") {
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == std::numeric_limits<mpm::Index>::max());
  }

  // Check for degrees of freedom
  SECTION("Check degrees of freedom") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->dof() == 2);
  }

  // Check status
  SECTION("Check status") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->status() == false);
    node->assign_status(true);
    REQUIRE(node->status() == true);
  }

  SECTION("Boundary ghost id") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    node->ghost_id(5);
    REQUIRE(node->ghost_id() == 5);
  }

  // Check MPI Rank
  SECTION("Check MPI Rank") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == 0);

    // Assign MPI ranks
    node->mpi_rank(0);
    node->mpi_rank(0);
    node->mpi_rank(1);

    std::set<unsigned> ranks = node->mpi_ranks();
    REQUIRE(ranks.size() == 2);
    std::vector<unsigned> mpi_ranks = {0, 1};
    unsigned i = 0;
    for (auto it = ranks.begin(); it != ranks.end(); ++it, ++i)
      REQUIRE(*it == mpi_ranks.at(i));
  }

  // Test coordinates function
  SECTION("coordinates function is checked") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;

    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);

    // Check for coordinates being zero
    auto coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    // Check for negative value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = -1. * std::numeric_limits<double>::max();
    node->assign_coordinates(coords);
    coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);

    // Check for positive value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = std::numeric_limits<double>::max();
    node->assign_coordinates(coords);
    coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);
  }

  SECTION("Check nodal properties") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);

    // Check mass
    REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
    double mass = 100.5;
    // Update mass to 100.5
    REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.5).epsilon(Tolerance));
    // Update mass to 201
    REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(201.0).epsilon(Tolerance));
    // Assign mass to 100
    mass = 100.;
    REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.0).epsilon(Tolerance));

    SECTION("Check nodal pressure") {
      // Check pressure
      REQUIRE(node->pressure(Nphase) == Approx(0.0).epsilon(Tolerance));
      double pressure = 1000.7;
      // Update pressure to 1000.7
      REQUIRE_NOTHROW(node->update_mass_pressure(Nphase, mass * pressure));
      REQUIRE(node->pressure(Nphase) == Approx(1000.7).epsilon(Tolerance));
      // Update pressure to 2001.4
      REQUIRE_NOTHROW(node->update_mass_pressure(Nphase, mass * pressure));
      REQUIRE(node->pressure(Nphase) == Approx(2001.4).epsilon(Tolerance));
      // Assign pressure to 1000
      pressure = 1000.;
      node->assign_pressure(Nphase, pressure);
      REQUIRE(node->pressure(Nphase) == Approx(1000.0).epsilon(Tolerance));
      // Assign mass to 0
      mass = 0.;
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
      // Try to update pressure to 2000, should throw and keep to 1000.
      node->assign_pressure(Nphase, pressure);
      REQUIRE(node->pressure(Nphase) == Approx(1000.0).epsilon(Tolerance));
      // Check pressure constraints
      SECTION("Check nodal pressure constraints") {
        // Check assign pressure constraint
        REQUIRE(node->assign_pressure_constraint(mpm::NodePhase::NSolid, 8000,
                                                 nullptr) == true);
        // Check apply pressure constraint
        REQUIRE_NOTHROW(
            node->apply_pressure_constraint(mpm::NodePhase::NSolid));
        // Check pressure
        REQUIRE(node->pressure(mpm::NodePhase::NSolid) ==
                Approx(8000).epsilon(Tolerance));
      }
    }

    SECTION("Check volume") {
      // Check volume
      REQUIRE(node->volume(Nphase) == Approx(0.0).epsilon(Tolerance));
      double volume = 100.5;
      // Update volume to 100.5
      REQUIRE_NOTHROW(node->update_volume(true, Nphase, volume));
      REQUIRE(node->volume(Nphase) == Approx(100.5).epsilon(Tolerance));
      // Update volume to 201
      REQUIRE_NOTHROW(node->update_volume(true, Nphase, volume));
      REQUIRE(node->volume(Nphase) == Approx(201.0).epsilon(Tolerance));
      // Assign volume to 100
      volume = 100.;
      REQUIRE_NOTHROW(node->update_volume(false, Nphase, volume));
      REQUIRE(node->volume(Nphase) == Approx(100.0).epsilon(Tolerance));
    }

    SECTION("Check external force") {
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10.;

      // Check current external force is zero
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));

      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_external_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Update force to 20.0
      REQUIRE_NOTHROW(node->update_external_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(20.).epsilon(Tolerance));

      // Assign force as 10.0
      REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Check if exception is handled
      Eigen::Matrix<double, Dim, 1> force_bad;
      for (unsigned i = 0; i < force_bad.size(); ++i) force_bad(i) = 10.;

      // Exception handling invalid force dimension
      // TODO Assert:    REQUIRE_NOTHROW(node->update_external_force(true, 1,
      // force_bad));

      // Exception handling invalid force dimension
      // TODO Assert: REQUIRE_NOTHROW(node->update_external_force(false, 1,
      // force_bad));

      SECTION("Check concentrated force") {
        // Set external force to zero
        force.setZero();
        REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));

        // Concentrated force
        std::shared_ptr<mpm::FunctionBase> ffunction = nullptr;
        double concentrated_force = 65.32;
        const unsigned Direction = 0;
        // Check traction
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->external_force(Nphase)(i) ==
                  Approx(0.).epsilon(Tolerance));

        REQUIRE(node->assign_concentrated_force(
                    Nphase, Direction, concentrated_force, ffunction) == true);
        double current_time = 0.0;
        node->apply_concentrated_force(Nphase, current_time);

        for (unsigned i = 0; i < Dim; ++i) {
          if (i == Direction)
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(concentrated_force).epsilon(Tolerance));

          else
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(0.).epsilon(Tolerance));
        }

        // Check for incorrect direction / phase
        const unsigned wrong_dir = 4;
        REQUIRE(node->assign_concentrated_force(
                    Nphase, wrong_dir, concentrated_force, ffunction) == false);

        // Check again to ensure value hasn't been updated
        for (unsigned i = 0; i < Dim; ++i) {
          if (i == Direction)
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(concentrated_force).epsilon(Tolerance));
          else
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(0.).epsilon(Tolerance));
        }
      }
    }

    SECTION("Check internal force") {
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10.;

      // Check current internal force is zero
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));

      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_internal_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Update force to 20.0
      REQUIRE_NOTHROW(node->update_internal_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(20.).epsilon(Tolerance));

      // Assign force as 10.0
      REQUIRE_NOTHROW(node->update_internal_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));
    }

    SECTION("Check compute acceleration and velocity") {
      // Time step
      const double dt = 0.1;

      // Nodal mass
      double mass = 100.;
      // Update mass to 100.5
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(mass).epsilon(Tolerance));

      // Check internal force
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10. * i;
      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_internal_force(false, Nphase, force));
      // Internal force
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(force(i)).epsilon(Tolerance));

      // External force
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 5. * i;
      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(force(i)).epsilon(Tolerance));

      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      for (unsigned i = 0; i < force.size(); ++i) force(i) = 15. * i;

      // Check acceleration
      Eigen::Matrix<double, Dim, 1> acceleration = force / mass;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Check velocity
      Eigen::Matrix<double, Dim, 1> velocity = force / mass * dt;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Apply velocity constraints
      REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      // Test velocity with constraints
      velocity << 10.5, 0.03;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Test acceleration with constraints
      acceleration[0] = 0.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Apply cundall damping when calculating acceleration
      REQUIRE(node->compute_acceleration_velocity_cundall(Nphase, dt, 0.05) ==
              true);

      // Test acceleration with cundall damping
      acceleration << 0., 0.1425;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Exception check when mass is zero
      mass = 0.;
      // Update mass to 0.
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(mass).epsilon(Tolerance));
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == false);
    }

    SECTION("Check momentum, velocity and acceleration") {
      // Time step
      const double dt = 0.1;

      // Check momentum
      Eigen::Matrix<double, Dim, 1> momentum;
      for (unsigned i = 0; i < momentum.size(); ++i) momentum(i) = 10.;

      // Check initial momentum
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      // Check update momentum to 10
      REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(10.).epsilon(Tolerance));

      // Check update momentum to 20
      REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(20.).epsilon(Tolerance));

      // Check assign momentum to 10
      REQUIRE_NOTHROW(node->update_momentum(false, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(10.).epsilon(Tolerance));

      // Check zero velocity
      for (unsigned i = 0; i < Dim; ++i)
        REQUIRE(node->velocity(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      // Check mass
      double mass = 0.;
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
      // Compute and check velocity this should throw zero mass
      node->compute_velocity();

      mass = 100.;
      // Update mass to 100.
      REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(100.).epsilon(Tolerance));

      // Compute and check velocity
      node->compute_velocity();
      for (unsigned i = 0; i < Dim; ++i)
        REQUIRE(node->velocity(Nphase)(i) == Approx(0.1).epsilon(Tolerance));

      // Check acceleration
      Eigen::Matrix<double, Dim, 1> acceleration;
      for (unsigned i = 0; i < acceleration.size(); ++i) acceleration(i) = 5.;

      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      REQUIRE_NOTHROW(node->update_acceleration(true, Nphase, acceleration));
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) == Approx(5.).epsilon(Tolerance));

      // Check if exception is handled
      Eigen::Matrix<double, Dim, 1> acceleration_bad;
      for (unsigned i = 0; i < acceleration_bad.size(); ++i)
        acceleration_bad(i) = 10.;

      // Check velocity before constraints
      Eigen::Matrix<double, Dim, 1> velocity;
      velocity << 0.1, 0.1;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Check acceleration before constraints
      acceleration << 5., 5.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      SECTION("Check Cartesian velocity constraints") {
        // Apply velocity constraints
        REQUIRE(node->assign_velocity_constraint(0, -12.5) == true);
        // Check out of bounds condition
        REQUIRE(node->assign_velocity_constraint(2, 0.) == false);

        // Apply constraints
        node->apply_velocity_constraints();

        // Check apply constraints
        velocity << -12.5, 0.1;
        for (unsigned i = 0; i < velocity.size(); ++i)
          REQUIRE(node->velocity(Nphase)(i) ==
                  Approx(velocity(i)).epsilon(Tolerance));

        acceleration << 0., 5.;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check general velocity constraints in 1 direction") {
        // Apply velocity constraints
        REQUIRE(node->assign_velocity_constraint(0, -12.5) == true);

        // Apply rotation matrix with Euler angles alpha = 10 deg, beta = 30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << 10. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply inclined velocity constraints
        node->apply_velocity_constraints();

        // Check applied velocity constraints in the global coordinates
        velocity << -9.583478335521184, -8.025403099849004;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->velocity(Nphase)(i) ==
                  Approx(velocity(i)).epsilon(Tolerance));

        // Check that the velocity is as specified in local coordinate
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(0) ==
                Approx(-12.5).epsilon(Tolerance));

        // Check applied constraints on acceleration in the global coordinates
        acceleration << -0.396139826697847, 0.472101061636807;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));

        // Check that the acceleration is 0 in local coordinate
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(0) ==
                Approx(0).epsilon(Tolerance));
      }

      SECTION("Check general velocity constraints in all directions") {
        // Apply velocity constraints
        REQUIRE(node->assign_velocity_constraint(0, -12.5) == true);
        REQUIRE(node->assign_velocity_constraint(1, 7.5) == true);

        // Apply rotation matrix with Euler angles alpha = -10 deg, beta = 30
        // deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << -10. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply inclined velocity constraints
        node->apply_velocity_constraints();

        // Check applied velocity constraints in the global coordinates
        velocity << -14.311308834766370, 2.772442864323454;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->velocity(Nphase)(i) ==
                  Approx(velocity(i)).epsilon(Tolerance));

        // Check that the velocity is as specified in local coordinate
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(0) ==
                Approx(-12.5).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(1) ==
                Approx(7.5).epsilon(Tolerance));

        // Check applied constraints on acceleration in the global coordinates
        acceleration << 0, 0;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));

        // Check that the acceleration is 0 in local coordinate
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(0) ==
                Approx(0).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(1) ==
                Approx(0).epsilon(Tolerance));
      }

      SECTION("Check Cartesian friction constraints") {
        // Apply friction constraints
        REQUIRE(node->assign_friction_constraint(1, 1, 0.2) == true);
        // Check out of bounds condition
        REQUIRE(node->assign_friction_constraint(2, 1, 0.2) == false);

        // Apply friction constraints
        node->apply_friction_constraints(dt);

        // Check apply constraints
        acceleration << 4., 5.;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check general friction constraints in 1 direction") {
        // Apply friction constraints
        REQUIRE(node->assign_friction_constraint(1, 1, 0.2) == true);

        // Apply rotation matrix with Euler angles alpha = 10 deg, beta = 30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << 10. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply general friction constraints
        node->apply_friction_constraints(dt);

        // Check applied constraints on acceleration in the global coordinates
        acceleration << 4.905579787672637, 4.920772034660430;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));

        // Check the acceleration in local coordinate
        acceleration << 6.920903430595146, 0.616284167162194;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check cartesian cohesion constraints") {
        // Case: static, cohesion fully mobilized, edge
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 0.);
        node->assign_velocity_constraint(1, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 10., -6.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 2) ==
                true);
        // Check out of bounds condition
        REQUIRE(node->assign_cohesion_constraint(2, -1., 1000, 0.25, 2) ==
                false);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        acceleration << 7.5, -6.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check failing cohesion constraint case") {
        // Apply cohesion constraint with incorrect nposition
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 3) ==
                false);
        // Should throw: invalid cohesion boundary nposition
        node->apply_cohesion_constraints(dt);
      }

      SECTION("Check additional cohesion constraint cases") {
        // Case: static, cohesion not fully mobilized, corner
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 0.);
        node->assign_velocity_constraint(1, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 1., -6.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 1) ==
                true);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        acceleration << 0., -6.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check additional cohesion constraint cases") {
        // Case: kinetic, cohesion fully mobilized, edge
        // Time step
        const double dt = 0.1;
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 4.);
        node->assign_velocity_constraint(1, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 10., -6.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 2) ==
                true);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        // vel_net=5., vel_cohesional=0.25
        acceleration << 7.5, -6.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check additional cohesion constraint cases") {
        // Case: kinetic, cohesion not fully mobilized, edge
        // Time step
        const double dt = 0.1;
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 1.);
        node->assign_velocity_constraint(1, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 10., -6.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 10000, 0.25, 2) ==
                true);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        // vel_net=2., vel_cohesional=2.5
        acceleration << -10., -6.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check general cohesion constraints in 1 direction") {
        // Case: static, cohesion fully mobilized, edge
        // Assign mass
        mass = 1000.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 0.);
        node->assign_velocity_constraint(1, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 0., -9.81;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 2) ==
                true);

        // Apply rotation matrix with Euler angles alpha = -30 deg, beta = 0 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << -30. * M_PI / 180, 0. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply general cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check applied constraints on acceleration in the global coordinates
        acceleration << -0.2165063509, -9.685;
        for (unsigned i = 0; i < Dim; ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }

        // Check the acceleration in local coordinates
        acceleration << 4.655, -8.49571;
        for (unsigned i = 0; i < Dim; ++i) {
          REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check Cartesian acceleration constraints") {
        // Apply acceleration constraint
        REQUIRE(node->assign_acceleration_constraint(0, 0.) == true);
        REQUIRE(node->assign_acceleration_constraint(1, 1.) == true);

        node->apply_acceleration_constraints();

        // Check apply constraints
        acceleration << 0., 1.;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check general acceleration constraints") {
        // Apply rotation matrix with Euler angles alpha = 10 deg, beta = 30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << 10. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply acceleration constraint
        REQUIRE(node->assign_acceleration_constraint(0, 0.) == true);
        REQUIRE(node->assign_acceleration_constraint(1, 1.) == true);

        node->apply_acceleration_constraints();

        // Check apply constraints
        acceleration << -0.642787609686539, 0.766044443118978;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }
    }

    SECTION("Check node material ids") {
      // Add material to nodes
      node->append_material_id(0);
      node->append_material_id(1);
      node->append_material_id(4);
      node->append_material_id(0);
      node->append_material_id(2);

      // Check size of material_ids
      REQUIRE(node->material_ids().size() == 4);

      // Check elements of material_ids
      std::vector<unsigned> material_ids = {0, 1, 2, 4};
      auto mat_ids = node->material_ids();
      unsigned i = 0;
      for (auto mitr = mat_ids.begin(); mitr != mat_ids.end(); ++mitr, ++i)
        REQUIRE(*mitr == material_ids.at(i));
    }
  }
}

// \brief Check node class for 3D case
TEST_CASE("Node is checked for 3D case", "[node][3D]") {
  const unsigned Dim = 3;
  const unsigned Dof = 3;
  const unsigned Nphases = 1;
  const unsigned Nphase = 0;

  Eigen::Vector3d coords;
  coords.setZero();

  // Check for id = 0
  SECTION("Node id is zero") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == 0);
  }

  // Check for id is a positive value
  SECTION("Node id is positive") {
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == std::numeric_limits<mpm::Index>::max());
  }

  // Check for degrees of freedom
  SECTION("Check degrees of freedom") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->dof() == 3);
  }

  // Check status
  SECTION("Check status") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->status() == false);
    node->assign_status(true);
    REQUIRE(node->status() == true);
  }

  SECTION("Boundary ghost id") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    node->ghost_id(5);
    REQUIRE(node->ghost_id() == 5);
  }

  // Check MPI Rank
  SECTION("Check MPI Rank") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
    REQUIRE(node->id() == 0);

    // Assign MPI ranks
    node->mpi_rank(0);
    node->mpi_rank(0);
    node->mpi_rank(1);

    std::set<unsigned> ranks = node->mpi_ranks();
    REQUIRE(ranks.size() == 2);
    std::vector<unsigned> mpi_ranks = {0, 1};
    unsigned i = 0;
    for (auto it = ranks.begin(); it != ranks.end(); ++it, ++i)
      REQUIRE(*it == mpi_ranks.at(i));
  }

  // Test coordinates function
  SECTION("coordinates function is checked") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;

    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);

    // Check for coordinates being zero
    auto coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    // Check for negative value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = -1. * std::numeric_limits<double>::max();
    node->assign_coordinates(coords);
    coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);

    // Check for positive value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = std::numeric_limits<double>::max();
    node->assign_coordinates(coords);
    coordinates = node->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);
  }

  SECTION("Check nodal properties") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::NodeBase<Dim>> node =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);

    // Check mass
    REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
    double mass = 100.5;
    // Update mass to 100.5
    REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.5).epsilon(Tolerance));
    // Update mass to 201
    REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(201.0).epsilon(Tolerance));
    // Assign mass to 100
    mass = 100.;
    REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.0).epsilon(Tolerance));

    SECTION("Check nodal pressure") {
      // Check pressure
      REQUIRE(node->pressure(Nphase) == Approx(0.0).epsilon(Tolerance));
      double pressure = 1000.7;
      // Update pressure to 1000.7
      REQUIRE_NOTHROW(node->update_mass_pressure(Nphase, mass * pressure));
      REQUIRE(node->pressure(Nphase) == Approx(1000.7).epsilon(Tolerance));
      // Update pressure to 2001.4
      REQUIRE_NOTHROW(node->update_mass_pressure(Nphase, mass * pressure));
      REQUIRE(node->pressure(Nphase) == Approx(2001.4).epsilon(Tolerance));
      // Assign pressure to 1000
      pressure = 1000.;
      node->assign_pressure(Nphase, pressure);
      REQUIRE(node->pressure(Nphase) == Approx(1000.0).epsilon(Tolerance));
      // Assign mass to 0
      mass = 0.;
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
      // Try to update pressure to 2000, should throw and keep to 1000.
      node->assign_pressure(Nphase, pressure);
      REQUIRE(node->pressure(Nphase) == Approx(1000.0).epsilon(Tolerance));
      // Check pressure constraints
      SECTION("Check nodal pressure constraints") {
        // Check assign pressure constraint
        REQUIRE(node->assign_pressure_constraint(mpm::NodePhase::NSolid, 8000,
                                                 nullptr) == true);
        // Check apply pressure constraint
        REQUIRE_NOTHROW(
            node->apply_pressure_constraint(mpm::NodePhase::NSolid));
        // Check pressure
        REQUIRE(node->pressure(mpm::NodePhase::NSolid) ==
                Approx(8000).epsilon(Tolerance));
      }
    }

    SECTION("Check external force") {
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10.;

      // Check current external force
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));

      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_external_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Update force to 2.0
      REQUIRE_NOTHROW(node->update_external_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(20.).epsilon(Tolerance));

      // Assign force as 1.0
      REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      SECTION("Check concentrated force") {
        // Set external force to zero
        force.setZero();
        REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));

        // Concentrated froce
        std::shared_ptr<mpm::FunctionBase> ffunction = nullptr;
        double concentrated_force = 65.32;
        const unsigned Direction = 0;
        // Check external force
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->external_force(Nphase)(i) ==
                  Approx(0.).epsilon(Tolerance));

        REQUIRE(node->assign_concentrated_force(
                    Nphase, Direction, concentrated_force, ffunction) == true);
        double current_time = 0.0;
        node->apply_concentrated_force(Nphase, current_time);

        for (unsigned i = 0; i < Dim; ++i) {
          if (i == Direction)
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(concentrated_force).epsilon(Tolerance));
          else
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(0.).epsilon(Tolerance));
        }

        // Check for incorrect direction / phase
        const unsigned wrong_dir = 4;
        REQUIRE(node->assign_concentrated_force(
                    Nphase, wrong_dir, concentrated_force, ffunction) == false);

        // Check again to ensure value hasn't been updated
        for (unsigned i = 0; i < Dim; ++i) {
          if (i == Direction)
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(concentrated_force).epsilon(Tolerance));
          else
            REQUIRE(node->external_force(Nphase)(i) ==
                    Approx(0.).epsilon(Tolerance));
        }
      }
    }

    SECTION("Check internal force") {
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10.;

      // Check current internal force is zero
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(0.).epsilon(Tolerance));

      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_internal_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));

      // Update force to 20.0
      REQUIRE_NOTHROW(node->update_internal_force(true, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(20.).epsilon(Tolerance));

      // Assign force as 10.0
      REQUIRE_NOTHROW(node->update_internal_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(10.).epsilon(Tolerance));
    }

    SECTION("Check compute acceleration and velocity") {
      // Time step
      const double dt = 0.1;

      // Nodal mass
      double mass = 100.;
      // Update mass to 100.5
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(mass).epsilon(Tolerance));

      // Check internal force
      // Create a force vector
      Eigen::Matrix<double, Dim, 1> force;
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 10. * i;
      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_internal_force(false, Nphase, force));
      // Internal force
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->internal_force(Nphase)(i) ==
                Approx(force(i)).epsilon(Tolerance));

      // External force
      for (unsigned i = 0; i < force.size(); ++i) force(i) = 5. * i;
      // Update force to 10.0
      REQUIRE_NOTHROW(node->update_external_force(false, Nphase, force));
      for (unsigned i = 0; i < force.size(); ++i)
        REQUIRE(node->external_force(Nphase)(i) ==
                Approx(force(i)).epsilon(Tolerance));

      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      for (unsigned i = 0; i < force.size(); ++i) force(i) = 15. * i;

      // Check acceleration
      Eigen::Matrix<double, Dim, 1> acceleration = force / mass;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Check velocity
      Eigen::Matrix<double, Dim, 1> velocity = force / mass * dt;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Apply velocity constraints
      REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      // Test velocity with constraints
      velocity << 10.5, 0.03, 0.06;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Test acceleration with constraints
      acceleration[0] = 0.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Apply cundall damping when calculating acceleration
      REQUIRE(node->compute_acceleration_velocity_cundall(Nphase, dt, 0.05) ==
              true);

      // Test acceleration with cundall damping
      acceleration << 0.0, 0.13322949016875, 0.28322949016875;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      // Apply velocity constraints
      REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == true);

      // Exception check when mass is zero
      mass = 0.;
      // Update mass to 0.
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(mass).epsilon(Tolerance));
      REQUIRE(node->compute_acceleration_velocity(Nphase, dt) == false);
    }

    SECTION("Check momentum, velocity and acceleration") {
      // Time step
      const double dt = 0.1;

      // Check momentum
      Eigen::Matrix<double, Dim, 1> momentum;
      for (unsigned i = 0; i < momentum.size(); ++i) momentum(i) = 10.;

      // Check initial momentum
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      // Check update momentum to 10
      REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(10.).epsilon(Tolerance));

      // Check update momentum to 20
      REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(20.).epsilon(Tolerance));

      // Check assign momentum to 10
      REQUIRE_NOTHROW(node->update_momentum(false, Nphase, momentum));
      for (unsigned i = 0; i < momentum.size(); ++i)
        REQUIRE(node->momentum(Nphase)(i) == Approx(10.).epsilon(Tolerance));

      // Check mass
      double mass = 0.;
      REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(0.0).epsilon(Tolerance));
      // Compute and check velocity this should throw zero mass
      node->compute_velocity();

      mass = 100.;
      // Update mass to 100.5
      REQUIRE_NOTHROW(node->update_mass(true, Nphase, mass));
      REQUIRE(node->mass(Nphase) == Approx(100.).epsilon(Tolerance));

      // Check zero velocity
      for (unsigned i = 0; i < Dim; ++i)
        REQUIRE(node->velocity(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      // Compute and check velocity
      node->compute_velocity();
      for (unsigned i = 0; i < Dim; ++i)
        REQUIRE(node->velocity(Nphase)(i) == Approx(0.1).epsilon(Tolerance));

      // Check acceleration
      Eigen::Matrix<double, Dim, 1> acceleration;
      for (unsigned i = 0; i < acceleration.size(); ++i) acceleration(i) = 5.;

      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) == Approx(0.).epsilon(Tolerance));

      REQUIRE_NOTHROW(node->update_acceleration(true, Nphase, acceleration));
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) == Approx(5.).epsilon(Tolerance));

      // Check velocity before constraints
      Eigen::Matrix<double, Dim, 1> velocity;
      velocity << 0.1, 0.1, 0.1;
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(node->velocity(Nphase)(i) ==
                Approx(velocity(i)).epsilon(Tolerance));

      // Check acceleration before constraints
      acceleration.resize(Dim);
      acceleration << 5., 5., 5.;
      for (unsigned i = 0; i < acceleration.size(); ++i)
        REQUIRE(node->acceleration(Nphase)(i) ==
                Approx(acceleration(i)).epsilon(Tolerance));

      SECTION("Check Cartesian velocity constraints") {
        // Apply velocity constraints
        REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
        REQUIRE(node->assign_velocity_constraint(1, -12.5) == true);
        // Check out of bounds condition
        REQUIRE(node->assign_velocity_constraint(4, 0.) == false);

        // Apply constraints
        node->apply_velocity_constraints();

        // Check apply constraints
        velocity << 10.5, -12.5, 0.1;
        for (unsigned i = 0; i < velocity.size(); ++i)
          REQUIRE(node->velocity(Nphase)(i) ==
                  Approx(velocity(i)).epsilon(Tolerance));

        acceleration << 0.0, 0.0, 5.;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check general velocity constraints in 2 directions") {
        // Apply velocity constraints
        REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
        REQUIRE(node->assign_velocity_constraint(2, -12.5) == true);

        // Apply rotation matrix with Euler angles alpha = 10 deg, beta = 20 deg
        // and gamma = 30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << 10. * M_PI / 180, 20. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply constraints
        node->apply_velocity_constraints();

        // Check apply constraints
        velocity << 8.056429712589052, 10.985674281018227, -8.995236782890776;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->velocity(Nphase)(i) ==
                  Approx(velocity(i)).epsilon(Tolerance));

        // Check that the velocity is as specified in local coordinate
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(0) ==
                Approx(10.5).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(2) ==
                Approx(-12.5).epsilon(Tolerance));

        // Check apply constraints
        acceleration << -1.754172871235121, 2.722373665593709,
            1.723750597747385;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));

        // Check that the acceleration is 0 in local coordinate
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(0) ==
                Approx(0).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(2) ==
                Approx(0).epsilon(Tolerance));
      }

      SECTION("Check general velocity constraints in all directions") {
        // Apply velocity constraints
        REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);
        REQUIRE(node->assign_velocity_constraint(1, -12.5) == true);
        REQUIRE(node->assign_velocity_constraint(2, 7.5) == true);

        // Apply rotation matrix with Euler angles alpha = -10 deg, beta = 20
        // deg and gamma = -30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << -10. * M_PI / 180, 20. * M_PI / 180, -30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply constraints
        node->apply_velocity_constraints();

        // Check apply constraints
        velocity << 13.351984588153375, -5.717804716697730, 10.572663655835457;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->velocity(Nphase)(i) ==
                  Approx(velocity(i)).epsilon(Tolerance));

        // Check that the velocity is as specified in local coordinate
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(0) ==
                Approx(10.5).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(1) ==
                Approx(-12.5).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->velocity(Nphase))(2) ==
                Approx(7.5).epsilon(Tolerance));

        // Check apply constraints
        acceleration << 0, 0, 0;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));

        // Check that the acceleration is 0 in local coordinate
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(0) ==
                Approx(0).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(1) ==
                Approx(0).epsilon(Tolerance));
        REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(2) ==
                Approx(0).epsilon(Tolerance));
      }

      SECTION("Check Cartesian friction constraints") {
        // Apply friction constraints
        REQUIRE(node->assign_friction_constraint(2, 2, 0.3) == true);
        // Check out of bounds condition
        REQUIRE(node->assign_friction_constraint(4, 1, 0.2) == false);

        // Apply constraints
        node->apply_friction_constraints(dt);

        // Check apply constraints
        acceleration << 3.939339828220179, 3.939339828220179, 5.;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check general friction constraints in 1 direction") {
        // Apply friction constraints
        REQUIRE(node->assign_friction_constraint(2, 2, 0.3) == true);

        // Apply rotation matrix with Euler angles alpha = 10 deg, beta = 20 deg
        // and gamma = 30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << 10. * M_PI / 180, 20. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply inclined velocity constraints
        node->apply_friction_constraints(dt);

        // Check applied constraints on acceleration in the global coordinates
        acceleration << 4.602895052828914, 4.492575657560740, 4.751301246937935;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));

        // Check the acceleration in local coordinate
        acceleration << 6.878925666702865, 3.365244416454818, 2.302228080558999;
        for (unsigned i = 0; i < Dim; ++i)
          REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check cartesian cohesion constraints") {
        // Case: static, cohesion fully mobilized, face
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 0.);
        node->assign_velocity_constraint(1, 0.);
        node->assign_velocity_constraint(2, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 10., -6., 0.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 3) ==
                true);
        // Check out of bounds condition
        REQUIRE(node->assign_cohesion_constraint(3, -1., 1000, 0.25, 3) ==
                false);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        // 10-0.625*(10/(10-0.625))), -6., 0.
        acceleration << 9.3333333333, -6., 0.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check failing cohesion constraint case") {
        // Apply cohesion constraint with incorrect nposition
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 0) ==
                false);
        // Should throw: invalid cohesion boundary nposition
        node->apply_cohesion_constraints(dt);
      }

      SECTION("Check additional cohesion constraint cases") {
        // Case: static, cohesion not fully mobilized, edge
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 0.);
        node->assign_velocity_constraint(1, 0.);
        node->assign_velocity_constraint(2, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 3., -6., 0.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 10000, 0.25, 2) ==
                true);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        acceleration << 0., -6., 0.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check additional cohesion constraint cases") {
        // Case: kinetic, cohesion fully mobilized, corner
        // Time step
        const double dt = 0.1;
        // Assign mass
        mass = 100.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 4.);
        node->assign_velocity_constraint(1, 0.);
        node->assign_velocity_constraint(2, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 10., -6., 0.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 1) ==
                true);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        // vel_net_t=5., vel_cohesion=0.015625
        acceleration << 9.84375, -6., 0.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check additional cohesion constraint cases") {
        // Case: kinetic, cohesion not fully mobilized, edge
        // Time step
        const double dt = 0.1;
        // Assign mass
        mass = 10.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 1);
        node->assign_velocity_constraint(1, 0.);
        node->assign_velocity_constraint(2, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 10., -6., 0.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 10000, 0.25, 2) ==
                true);

        // Apply cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check apply constraints
        // vel_net_t=2., vel_cohesion=3.125
        acceleration << -10., -6., 0.;
        for (unsigned i = 0; i < acceleration.size(); ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check general cohesion constraints in 1 direction") {
        // Case: static, cohesion fully mobilized, face
        // Assign mass
        mass = 2000.;
        node->update_mass(false, Nphase, mass);
        // Reset velocity
        node->assign_velocity_constraint(0, 0.);
        node->assign_velocity_constraint(1, 0.);
        node->assign_velocity_constraint(2, 0.);
        node->apply_velocity_constraints();
        // Assign acceleration
        acceleration << 0., -9.81, 0.;
        node->update_acceleration(false, Nphase, acceleration);
        // Apply cohesion constraints
        REQUIRE(node->assign_cohesion_constraint(1, -1., 1000, 0.25, 3) ==
                true);

        // Apply rotation matrix with Euler angles alpha = -30 deg, beta = 0 deg
        // and gamma = 0 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << -30. * M_PI / 180, 0. * M_PI / 180, 0. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply general cohesion constraints
        node->apply_cohesion_constraints(dt);

        // Check applied constraints on acceleration in the global coordinates
        // x=x'*cos(30)+y'*sin(30), y=y'*cos(30)-x'*sin(30)
        acceleration << -0.027236821, -9.7942748, 0.;
        for (unsigned i = 0; i < Dim; ++i) {
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }

        // Check the acceleration in local coordinates
        // x'=4.905-0.03125*(4.905/(4.905-0.03125)), y'=-9.81*cos(30), z'=0.
        acceleration << 4.873549628, -8.495709211, 0.;
        for (unsigned i = 0; i < Dim; ++i) {
          REQUIRE((inverse_rotation_matrix * node->acceleration(Nphase))(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
        }
      }

      SECTION("Check Cartesian acceleration constraints") {
        // Apply acceleration constraint
        REQUIRE(node->assign_acceleration_constraint(0, 0.) == true);
        REQUIRE(node->assign_acceleration_constraint(1, 1.) == true);
        REQUIRE(node->assign_acceleration_constraint(2, 2.) == true);

        node->apply_acceleration_constraints();

        // Check apply constraints
        acceleration << 0., 1., 2.;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }

      SECTION("Check general acceleration constraints") {
        // Apply rotation matrix with Euler angles alpha = 10 deg, beta = 20 deg
        // and gamma = 30 deg
        Eigen::Matrix<double, Dim, 1> euler_angles;
        euler_angles << 10. * M_PI / 180, 20. * M_PI / 180, 30. * M_PI / 180;
        const auto rotation_matrix =
            mpm::geometry::rotation_matrix(euler_angles);
        node->assign_rotation_matrix(rotation_matrix);
        const auto inverse_rotation_matrix = rotation_matrix.inverse();

        // Apply acceleration constraint
        REQUIRE(node->assign_acceleration_constraint(0, 0.) == true);
        REQUIRE(node->assign_acceleration_constraint(1, 1.) == true);
        REQUIRE(node->assign_acceleration_constraint(2, 2.) == true);

        node->apply_acceleration_constraints();

        // Check apply constraints
        acceleration << -0.304490395522427, -0.242764661649871,
            2.20189711796183;
        for (unsigned i = 0; i < acceleration.size(); ++i)
          REQUIRE(node->acceleration(Nphase)(i) ==
                  Approx(acceleration(i)).epsilon(Tolerance));
      }
    }

    SECTION("Check node material ids") {
      // Add material to nodes
      node->append_material_id(0);
      node->append_material_id(1);
      node->append_material_id(4);
      node->append_material_id(0);
      node->append_material_id(2);

      // Check size of material_ids
      REQUIRE(node->material_ids().size() == 4);

      // Check elements of material_ids
      std::vector<unsigned> material_ids = {0, 1, 2, 4};
      auto mat_ids = node->material_ids();
      unsigned i = 0;
      for (auto mitr = mat_ids.begin(); mitr != mat_ids.end(); ++mitr, ++i)
        REQUIRE(*mitr == material_ids.at(i));
    }
  }
}
