#include <limits>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "catch.hpp"
#include "convergence_criterion_base.h"
#include "convergence_criterion_residual.h"
#include "convergence_criterion_solution.h"

// Generate random RHS vector with specified dimension and seed
Eigen::VectorXd CreateRandomRHSVectorWithMagnitude(unsigned dim,
                                                   double magnitude,
                                                   unsigned seed = 0) {
  std::srand(seed);
  Eigen::VectorXd vector = magnitude * Eigen::VectorXd::Random(dim);
  return vector;
}

TEST_CASE("Convergence criteria test", "[convergence_criteria]") {
  // Allowed relative tolerance
  double rel_tolerance = 1.E-12;
  // Tolerance
  const double tolerance = 1.E-12;
  // verbosity
  unsigned verbosity = 2;

  SECTION("Residual criterion") {
    // Construct residual criterion
    std::shared_ptr<mpm::ConvergenceCriterionBase> residual_criterion =
        std::make_shared<mpm::ConvergenceCriterionResidual>(
            rel_tolerance, tolerance, verbosity);

    // Convergence false
    auto random_vec = CreateRandomRHSVectorWithMagnitude(16, 10);
    REQUIRE_FALSE(residual_criterion->check_convergence(random_vec, true));

    // Check relative tolerance
    REQUIRE_NOTHROW(residual_criterion->set_tolerance(1.E-5));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-5);
    REQUIRE(residual_criterion->check_convergence(random_vec));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-7);
    REQUIRE_FALSE(residual_criterion->check_convergence(random_vec, true));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-8);
    REQUIRE_FALSE(residual_criterion->check_convergence(random_vec));

    // Check abs tolerance
    REQUIRE_NOTHROW(residual_criterion->set_abs_tolerance(1.E-6));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-7);
    REQUIRE(residual_criterion->check_convergence(random_vec, true));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-3);
    REQUIRE_FALSE(residual_criterion->check_convergence(random_vec));

    // Constructor
    std::shared_ptr<mpm::ConvergenceCriterionBase> residual_criterion2 =
        std::make_shared<mpm::ConvergenceCriterionResidual>(rel_tolerance,
                                                            verbosity);
    std::shared_ptr<mpm::ConvergenceCriterionBase> residual_criterion3(
        residual_criterion);
    residual_criterion3 = residual_criterion2;
    residual_criterion3 = std::move(residual_criterion2);

    // Others
    residual_criterion3->set_verbosity(0);
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-15);
    REQUIRE(residual_criterion3->check_convergence(random_vec, true));
  }

  SECTION("Solution criterion") {
    // Construct solution criterion
    std::shared_ptr<mpm::ConvergenceCriterionBase> solution_criterion =
        std::make_shared<mpm::ConvergenceCriterionSolution>(rel_tolerance,
                                                            verbosity);

    // Convergence false
    auto random_vec = CreateRandomRHSVectorWithMagnitude(16, 10);
    REQUIRE_FALSE(solution_criterion->check_convergence(random_vec));

    // Check tolerance
    REQUIRE_NOTHROW(solution_criterion->set_tolerance(1.E-5));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-6);
    REQUIRE(solution_criterion->check_convergence(random_vec));
    random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-4);
    REQUIRE_FALSE(solution_criterion->check_convergence(random_vec));
  }
}

TEST_CASE("Convergence criteria test MPI", "[convergence_criteria][mpi]") {
  // Allowed relative tolerance
  double rel_tolerance = 1.E-12;
  // Tolerance
  const double tolerance = 1.E-12;
  // verbosity
  unsigned verbosity = 2;

  // Get number of MPI ranks
  int mpi_size, mpi_rank;
#ifdef USE_PETSC
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  SECTION("Parallel residual criterion") {
    // Construct residual criterion
    std::shared_ptr<mpm::ConvergenceCriterionBase> residual_criterion =
        std::make_shared<mpm::ConvergenceCriterionResidual>(
            rel_tolerance, tolerance, verbosity);

    if (mpi_size == 4) {
      // Assign MPI attributes
      residual_criterion->assign_global_active_dof(16);
      std::vector<int> rgm(4);
      for (int i = 0; i < rgm.size(); i++) rgm[i] = mpi_rank * mpi_size + i;
      residual_criterion->assign_rank_global_mapper(rgm);

      // Convergence false
      auto random_vec = CreateRandomRHSVectorWithMagnitude(16, 10);
      Eigen::VectorXd my_vec(4);
      for (unsigned i = 0; i < my_vec.size(); i++)
        my_vec(i) = random_vec(mpi_rank * mpi_size + i);
      REQUIRE_FALSE(residual_criterion->check_convergence(my_vec, true));

      // Check relative tolerance
      REQUIRE_NOTHROW(residual_criterion->set_tolerance(1.E-5));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-5);
      for (unsigned i = 0; i < my_vec.size(); i++)
        my_vec(i) = random_vec(mpi_rank * mpi_size + i);
      REQUIRE(residual_criterion->check_convergence(random_vec));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-7);
      for (unsigned i = 0; i < my_vec.size(); i++)
        my_vec(i) = random_vec(mpi_rank * mpi_size + i);
      REQUIRE_FALSE(residual_criterion->check_convergence(random_vec, true));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-8);
      for (unsigned i = 0; i < my_vec.size(); i++)
        my_vec(i) = random_vec(mpi_rank * mpi_size + i);
      REQUIRE_FALSE(residual_criterion->check_convergence(random_vec));

      // Check abs tolerance
      REQUIRE_NOTHROW(residual_criterion->set_abs_tolerance(1.E-6));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-7);
      for (unsigned i = 0; i < my_vec.size(); i++)
        my_vec(i) = random_vec(mpi_rank * mpi_size + i);
      REQUIRE(residual_criterion->check_convergence(random_vec, true));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-3);
      for (unsigned i = 0; i < my_vec.size(); i++)
        my_vec(i) = random_vec(mpi_rank * mpi_size + i);
      REQUIRE_FALSE(residual_criterion->check_convergence(random_vec));
    }
  }

  SECTION("Solution criterion") {
    // Construct solution criterion
    std::shared_ptr<mpm::ConvergenceCriterionBase> solution_criterion =
        std::make_shared<mpm::ConvergenceCriterionSolution>(rel_tolerance,
                                                            verbosity);

    if (mpi_size == 4) {
      // Assign MPI attributes
      solution_criterion->assign_global_active_dof(16);
      std::vector<int> rgm(16);
      for (int i = 0; i < rgm.size(); i++) rgm[i] = i;
      solution_criterion->assign_rank_global_mapper(rgm);

      // Convergence false
      auto random_vec = CreateRandomRHSVectorWithMagnitude(16, 10);
      REQUIRE_FALSE(solution_criterion->check_convergence(random_vec));

      // Check tolerance
      REQUIRE_NOTHROW(solution_criterion->set_tolerance(1.E-5));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-6);
      REQUIRE(solution_criterion->check_convergence(random_vec));
      random_vec = CreateRandomRHSVectorWithMagnitude(16, 1.e-4);
      REQUIRE_FALSE(solution_criterion->check_convergence(random_vec));
    }
  }
#endif
}