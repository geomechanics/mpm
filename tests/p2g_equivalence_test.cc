#include <cmath>
#include <memory>
#include <vector>

#include "catch.hpp"

#include "cell.h"
#include "element.h"
#include "factory.h"
#include "hexahedron_element.h"
#include "mesh.h"
#include "node.h"
#include "particle.h"

//! Direct equivalence test: locked P2G vs. thread-local P2G.
TEST_CASE("P2G locked vs thread-local equivalence", "[p2g][equivalence]") {
  const unsigned Dim = 3;
  const unsigned Dof = 3;
  const unsigned Nphases = 1;
  const unsigned Nnodes = 8;
  const unsigned Phase = 0;
  const double Tolerance = 1.E-9;

  auto element = Factory<mpm::Element<Dim>>::instance()->create("ED3H8");
  auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);

  const double dx = 1.0;
  for (unsigned k = 0; k < 3; ++k) {
    for (unsigned j = 0; j < 3; ++j) {
      for (unsigned i = 0; i < 3; ++i) {
        Eigen::Matrix<double, 3, 1> coords;
        coords << i * dx, j * dx, k * dx;
        auto node = std::make_shared<mpm::Node<Dim, Dof, Nphases>>(
            k * 9 + j * 3 + i, coords);
        mesh->add_node(node, false);
      }
    }
  }

  auto node_id = [](unsigned i, unsigned j, unsigned k) {
    return k * 9 + j * 3 + i;
  };

  std::vector<std::shared_ptr<mpm::Cell<Dim>>> cells;
  cells.reserve(8);
  for (unsigned k = 0; k < 2; ++k) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned i = 0; i < 2; ++i) {
        auto cell =
            std::make_shared<mpm::Cell<Dim>>(cells.size(), Nnodes, element);
        cell->add_node(0, mesh->node(node_id(i, j, k)));
        cell->add_node(1, mesh->node(node_id(i + 1, j, k)));
        cell->add_node(2, mesh->node(node_id(i + 1, j + 1, k)));
        cell->add_node(3, mesh->node(node_id(i, j + 1, k)));
        cell->add_node(4, mesh->node(node_id(i, j, k + 1)));
        cell->add_node(5, mesh->node(node_id(i + 1, j, k + 1)));
        cell->add_node(6, mesh->node(node_id(i + 1, j + 1, k + 1)));
        cell->add_node(7, mesh->node(node_id(i, j + 1, k + 1)));
        cell->initialise();
        mesh->add_cell(cell, false);
        cells.emplace_back(cell);
      }
    }
  }

  const double off = dx * 0.25;
  mpm::Index pid = 0;
  for (unsigned k = 0; k < 2; ++k) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned i = 0; i < 2; ++i) {
        for (unsigned pz = 0; pz < 2; ++pz) {
          for (unsigned py = 0; py < 2; ++py) {
            for (unsigned px = 0; px < 2; ++px) {
              Eigen::Matrix<double, 3, 1> coords;
              coords << i * dx + off + px * dx * 0.5,
                  j * dx + off + py * dx * 0.5,
                  k * dx + off + pz * dx * 0.5;
              auto particle = std::make_shared<mpm::Particle<Dim>>(pid++,
                                                                    coords);
              Eigen::Matrix<double, 3, 1> xi;
              xi << -0.5 + px, -0.5 + py, -0.5 + pz;
              particle->assign_cell_xi(cells[(k * 2 + j) * 2 + i], xi);
              particle->assign_mass(1.0 + 0.01 * static_cast<double>(pid));
              particle->assign_volume(0.125);
              particle->assign_velocity(
                  (Eigen::Matrix<double, 3, 1>() << 1.0, 2.0, 3.0)
                      .finished());
              Eigen::Matrix<double, 6, 1> stress;
              stress << 2.0, 3.0, 4.0, 0.5, 0.25, -0.125;
              particle->initial_stress(stress);
              mesh->add_particle(particle, false);
            }
          }
        }
      }
    }
  }

  mesh->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Dim>::compute_shapefn,
                std::placeholders::_1));

  const Eigen::Matrix<double, Dim, 1> gravity =
      (Eigen::Matrix<double, Dim, 1>() << 0.0, 0.0, -9.81).finished();

  mesh->iterate_over_nodes(
      std::bind(&mpm::NodeBase<Dim>::initialise, std::placeholders::_1));
  mesh->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Dim>::map_mass_momentum_to_nodes,
      std::placeholders::_1, mpm::VelocityUpdate::FLIP));
  mesh->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Dim>::map_body_force, std::placeholders::_1,
      gravity));
  mesh->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Dim>::map_internal_force, std::placeholders::_1));

  const unsigned nnodes = mesh->nnodes();
  std::vector<double> old_mass(nnodes);
  std::vector<Eigen::Matrix<double, Dim, 1>> old_momentum(nnodes);
  std::vector<Eigen::Matrix<double, Dim, 1>> old_external(nnodes);
  std::vector<Eigen::Matrix<double, Dim, 1>> old_internal(nnodes);
  for (unsigned n = 0; n < nnodes; ++n) {
    auto node = mesh->node(n);
    old_mass[n] = node->mass(Phase);
    old_momentum[n] = node->momentum(Phase);
    old_external[n] = node->external_force(Phase);
    old_internal[n] = node->internal_force(Phase);
  }

  mesh->iterate_over_nodes(
      std::bind(&mpm::NodeBase<Dim>::initialise, std::placeholders::_1));
  mesh->map_mass_momentum_to_nodes_thread_local(Phase);
  mesh->map_body_internal_force_thread_local(gravity, Phase);

  for (unsigned n = 0; n < nnodes; ++n) {
    auto node = mesh->node(n);
    REQUIRE(node->mass(Phase) == Approx(old_mass[n]).epsilon(Tolerance));
    for (unsigned d = 0; d < Dim; ++d) {
      REQUIRE(node->momentum(Phase)(d) ==
              Approx(old_momentum[n](d)).epsilon(Tolerance));
      REQUIRE(node->external_force(Phase)(d) ==
              Approx(old_external[n](d)).epsilon(Tolerance));
      REQUIRE(node->internal_force(Phase)(d) ==
              Approx(old_internal[n](d)).epsilon(Tolerance));
    }
  }
}
