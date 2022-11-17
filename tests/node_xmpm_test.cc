#include <Eigen/Dense>
#include <memory>

#include "catch.hpp"

#include "node.h"

//! \brief Check nodal functions for xmpm
TEST_CASE("Node_xmpm is checked for 3D case", "[node][XMPM][3D]") {

  const unsigned Dim = 3;
  const unsigned Dof = 3;
  const unsigned Nphases = 1;
  const unsigned Nphase = 0;

  Eigen::Vector3d coords;
  coords.setZero();
  mpm::Index id = 0;
  std::shared_ptr<mpm::NodeBase<Dim>> node =
      std::make_shared<mpm::Node<Dim, Dof, Nphases>>(id, coords);
  // Number of nodes
  unsigned nnodes = 1;
  // Number of materials
  unsigned nmaterials = 1;
  // Tolerance
  const double Tolerance = 1.E-7;
  // Declare nodal properties
  std::shared_ptr<mpm::NodalProperties> property_handle =
      std::make_shared<mpm::NodalProperties>();
  // Define properties to be created
  std::string property = "momentum";
  unsigned prop_id = 0;
  REQUIRE(property_handle->create_property(property, nnodes * Dim, nmaterials));
  // clang-format off
    Eigen::Matrix<double,Dim,1> data;
    data<< 0.0, 
           0.5,
          1.0;
  // clang-format on
  REQUIRE_NOTHROW(property_handle->assign_property(property, 0, 0, data, Dim));
  node->initialise_discontinuity_property_handle(prop_id, property_handle);
  // Check discontinuous property
  SECTION("Check discontinuous properties") {
    // REQUIRE_NOTHROW(node->assign_discontinuity_enrich(true));
    // // initialise discontinuity property handle

    REQUIRE(node->discontinuity_property(property, Dim)(0, 0) ==
            Approx(0.0).epsilon(Tolerance));
    REQUIRE(node->discontinuity_property(property, Dim)(1, 0) ==
            Approx(0.5).epsilon(Tolerance));
    REQUIRE(node->discontinuity_property(property, Dim)(2, 0) ==
            Approx(1.0).epsilon(Tolerance));

    node->update_discontinuity_property(true, property, data, prop_id, Dim);
    REQUIRE(node->discontinuity_property(property, Dim)(0, 0) ==
            Approx(0.0).epsilon(Tolerance));
    REQUIRE(node->discontinuity_property(property, Dim)(1, 0) ==
            Approx(1.0).epsilon(Tolerance));
    REQUIRE(node->discontinuity_property(property, Dim)(2, 0) ==
            Approx(2.0).epsilon(Tolerance));
  }
  SECTION("Check momentum and velocity constrains") {
    // Time step
    const double dt = 0.1;
    // Check momentum
    Eigen::Matrix<double, Dim, 1> momentum;
    for (unsigned i = 0; i < momentum.size(); ++i) momentum(i) = 1.;

    // Check initial momentum
    for (unsigned i = 0; i < momentum.size(); ++i)
      REQUIRE(node->momentum(Nphase)(i) == Approx(0.).epsilon(Tolerance));

    // update momentum to 1
    REQUIRE_NOTHROW(node->update_momentum(true, Nphase, momentum));

    // clang-format off
    Eigen::Matrix<double,Dim,1> momentum_enrich;
    momentum_enrich<< 1.0, 
                      1.0,
                      1.0;
    // clang-format on
    REQUIRE(property_handle->create_property("momenta_enrich", nnodes * Dim,
                                             nmaterials));
    REQUIRE_NOTHROW(property_handle->assign_property("momenta_enrich", prop_id,
                                                     0, momentum_enrich, Dim));

    // clang-format off
    Eigen::Matrix<double,Dim,1> force_enrich;
    force_enrich<< 10, 
                   10,
                   10;
    // clang-format on

    REQUIRE(property_handle->create_property("internal_force_enrich",
                                             nnodes * Dim, nmaterials));
    REQUIRE_NOTHROW(property_handle->assign_property(
        "internal_force_enrich", prop_id, 0, force_enrich, Dim));

    REQUIRE(property_handle->create_property("external_force_enrich",
                                             nnodes * Dim, nmaterials));
    REQUIRE_NOTHROW(property_handle->assign_property(
        "external_force_enrich", prop_id, 0, force_enrich, Dim));

    // clang-format off
    Eigen::Matrix<double,Dim,1> normal_enrich;
    normal_enrich<< 0, 
                   0,
                   1;
    // clang-format on
    REQUIRE(property_handle->create_property(
        "normal_unit_vectors_discontinuity", nnodes * Dim, nmaterials));
    REQUIRE_NOTHROW(property_handle->assign_property(
        "normal_unit_vectors_discontinuity", prop_id, 0, normal_enrich, Dim));

    Eigen::Matrix<double, 1, 1> coef;
    coef << -1.0;
    REQUIRE(
        property_handle->create_property("friction_coef", nnodes, nmaterials));
    REQUIRE_NOTHROW(
        property_handle->assign_property("friction_coef", prop_id, 0, coef, 1));

    node->compute_momentum_discontinuity(Nphase, dt);
    for (unsigned i = 0; i < momentum.size(); ++i)
      REQUIRE(node->discontinuity_property("momenta_enrich", Dim)(i, 0) ==
              Approx(3.0).epsilon(Tolerance));

    for (unsigned i = 0; i < momentum.size(); ++i) momentum(i) = 10.;

    // Check assign momentum to 10
    REQUIRE_NOTHROW(node->update_momentum(false, Nphase, momentum));

    // check velocity and constrain
    double mass = 100.;
    // Update mass to 100.0
    REQUIRE_NOTHROW(node->update_mass(false, Nphase, mass));
    REQUIRE(node->mass(Nphase) == Approx(100.).epsilon(Tolerance));

    // Compute and check velocity this should throw zero mass
    node->compute_velocity();
    // Apply velocity constraints
    REQUIRE(node->assign_velocity_constraint(0, 10.5) == true);

    // Check velocity before constraints
    Eigen::Matrix<double, Dim, 1> velocity;
    for (unsigned i = 0; i < momentum.size(); ++i) velocity(i) = 0.1;
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(node->velocity(Nphase)(i) ==
              Approx(velocity(i)).epsilon(Tolerance));

    Eigen::Matrix<double, 1, 1> mass_enrich;
    mass_enrich << 1.0;

    REQUIRE(property_handle->create_property("mass_enrich", nnodes * 1,
                                             nmaterials));
    REQUIRE_NOTHROW(property_handle->assign_property("mass_enrich", prop_id, 0,
                                                     mass_enrich, 1));
    // Apply constraints
    node->apply_velocity_constraints();

    velocity << 10.5;

    for (unsigned i = 0; i < 1; ++i)
      REQUIRE((node->momentum(Nphase)(i) +
               node->discontinuity_property("momenta_enrich", Dim)(i, 0)) /
                  ((node->mass(Nphase) +
                    node->discontinuity_property("mass_enrich", 1)(i, 0))) ==
              Approx(10.5).epsilon(Tolerance));

    for (unsigned i = 0; i < 1; ++i)
      REQUIRE((node->momentum(Nphase)(i) -
               node->discontinuity_property("momenta_enrich", Dim)(i, 0)) /
                  ((node->mass(Nphase) -
                    node->discontinuity_property("mass_enrich", 1)(i, 0))) ==
              Approx(10.5).epsilon(Tolerance));
  }
}
