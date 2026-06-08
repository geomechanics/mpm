#include <chrono>
#include <iomanip>
#include <iostream>
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

//! Hidden micro-benchmark for P2G mass/momentum and force mapping.
TEST_CASE("P2G thread-local performance benchmark", "[benchmark][p2g][.]") {
  const unsigned Dim = 3;
  const unsigned Dof = 3;
  const unsigned Nphases = 1;
  const unsigned Nnodes = 8;
  const unsigned Phase = 0;
  const unsigned N = 32;
  const unsigned n_iterations = 3;

  auto element = Factory<mpm::Element<Dim>>::instance()->create("ED3H8");
  auto mesh = std::make_shared<mpm::Mesh<Dim>>(0);
  std::vector<std::shared_ptr<mpm::Cell<Dim>>> cells;
  cells.reserve(N * N * N);

  const double dx = 1.0;
  const unsigned nx = N + 1;
  const unsigned ny = N + 1;
  const unsigned nz = N + 1;

  for (unsigned k = 0; k < nz; ++k) {
    for (unsigned j = 0; j < ny; ++j) {
      for (unsigned i = 0; i < nx; ++i) {
        Eigen::Matrix<double, 3, 1> coords;
        coords << i * dx, j * dx, k * dx;
        auto node = std::make_shared<mpm::Node<Dim, Dof, Nphases>>(
            k * nx * ny + j * nx + i, coords);
        mesh->add_node(node, false);
      }
    }
  }

  auto node_id = [nx, ny](unsigned i, unsigned j, unsigned k) {
    return k * nx * ny + j * nx + i;
  };

  for (unsigned k = 0; k < N; ++k) {
    for (unsigned j = 0; j < N; ++j) {
      for (unsigned i = 0; i < N; ++i) {
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
  for (unsigned k = 0; k < N; ++k) {
    for (unsigned j = 0; j < N; ++j) {
      for (unsigned i = 0; i < N; ++i) {
        for (unsigned pz = 0; pz < 2; ++pz) {
          for (unsigned py = 0; py < 2; ++py) {
            for (unsigned px = 0; px < 2; ++px) {
              Eigen::Matrix<double, 3, 1> coords;
              coords << i * dx + off + px * dx * 0.5,
                  j * dx + off + py * dx * 0.5,
                  k * dx + off + pz * dx * 0.5;
              const mpm::Index pid =
                  ((k * N + j) * N + i) * 8 + (pz * 4 + py * 2 + px);
              auto particle = std::make_shared<mpm::Particle<Dim>>(pid,
                                                                    coords);
              Eigen::Matrix<double, 3, 1> xi;
              xi << -0.5 + px, -0.5 + py, -0.5 + pz;
              particle->assign_cell_xi(cells[(k * N + j) * N + i], xi);
              particle->assign_mass(1.0);
              particle->assign_volume(0.125);
              particle->assign_velocity(Eigen::Matrix<double, 3, 1>::Ones());
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

  Eigen::Matrix<double, Dim, 1> gravity = Eigen::Matrix<double, Dim, 1>::Zero();
  gravity(2) = -9.81;

  for (int warmup = 0; warmup < 2; ++warmup) {
    mesh->iterate_over_nodes(
        std::bind(&mpm::NodeBase<Dim>::initialise, std::placeholders::_1));
    mesh->map_mass_momentum_to_nodes_thread_local(Phase);
    mesh->map_body_internal_force_thread_local(gravity, Phase);
  }

  unsigned nthreads = 1;
#ifdef _OPENMP
  nthreads = static_cast<unsigned>(omp_get_max_threads());
#endif

  auto t_new_start = std::chrono::steady_clock::now();
  for (unsigned iter = 0; iter < n_iterations; ++iter) {
    mesh->iterate_over_nodes(
        std::bind(&mpm::NodeBase<Dim>::initialise, std::placeholders::_1));
    mesh->map_mass_momentum_to_nodes_thread_local(Phase);
    mesh->map_body_internal_force_thread_local(gravity, Phase);
  }
  auto t_new_end = std::chrono::steady_clock::now();
  const double new_ms =
      std::chrono::duration<double, std::milli>(t_new_end - t_new_start)
          .count();

  std::cout << "\n=== P2G Performance Benchmark ===\n";
  std::cout << "Grid: " << N << "x" << N << "x" << N << " cells\n";
  std::cout << "Particles: " << mesh->nparticles() << "\n";
  std::cout << "Nodes:     " << mesh->nnodes() << "\n";
  std::cout << "Threads:   " << nthreads << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Thread-local path: " << (new_ms / n_iterations)
            << " ms/iter\n";

  if (nthreads == 1) {
    auto t_old_start = std::chrono::steady_clock::now();
    for (unsigned iter = 0; iter < n_iterations; ++iter) {
      mesh->iterate_over_nodes(
          std::bind(&mpm::NodeBase<Dim>::initialise, std::placeholders::_1));
      mesh->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Dim>::map_mass_momentum_to_nodes,
          std::placeholders::_1, mpm::VelocityUpdate::FLIP));
      mesh->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Dim>::map_body_force, std::placeholders::_1,
          gravity));
      mesh->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Dim>::map_internal_force,
          std::placeholders::_1));
    }
    auto t_old_end = std::chrono::steady_clock::now();
    const double old_ms =
        std::chrono::duration<double, std::milli>(t_old_end - t_old_start)
            .count();
    std::cout << "Locked path:       " << (old_ms / n_iterations)
              << " ms/iter\n";
    std::cout << "Speedup:           " << (old_ms / new_ms) << "x\n";
  } else {
    std::cout << "Locked path:       skipped for nthreads > 1\n";
  }

  REQUIRE(true);
}
