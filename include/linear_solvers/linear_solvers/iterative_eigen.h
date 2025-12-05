#ifndef MPM_ITERATIVE_EIGEN_H_
#define MPM_ITERATIVE_EIGEN_H_

#include <cmath>

#include "factory.h"
#include "solver_base.h"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
// Additional includes for unsupported solvers
#include <unsupported/Eigen/IterativeSolvers>

namespace mpm {

//! MPM Iterative Eigen solver class
//! \brief Iterative linear sparse matrix solver class using Eigen library
template <typename Traits>
class IterativeEigen : public SolverBase<Traits> {
 public:
  //! Constructor
  //! \param[in] max_iter Maximum number of iterations
  //! \param[in] tolerance Tolerance for solver to achieve convergence
  IterativeEigen(unsigned max_iter, double tolerance)
      : mpm::SolverBase<Traits>(max_iter, tolerance) {
    //! Logger
    std::string logger = "EigenIterativeSolver::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
    //! Default sub solver type
    sub_solver_type_ = "cg";
  };

  //! Destructor
  ~IterativeEigen(){};

  //! Matrix solver with default initial guess
  Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A,
                        const Eigen::VectorXd& b) override;

  //! Return the type of solver
  std::string solver_type() const { return "Eigen"; }

  //! Assign global active dof
  void assign_global_active_dof(unsigned global_active_dof) override {}

  //! Assign rank to global mapper
  void assign_rank_global_mapper(
      const std::vector<int>& rank_global_mapper) override {}

 private:
  // ===== Private solver methods for different algorithms =====
  
  //! Conjugate Gradient solver with preconditioner support
  Eigen::VectorXd solveConjugateGradient(
      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);
  
  //! BiCGSTAB solver with preconditioner support
  Eigen::VectorXd solveBiCGSTAB(
      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);
  
  //! Least Squares Conjugate Gradient solver with preconditioner support
  Eigen::VectorXd solveLeastSquaresCG(
      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);
  
  //! GMRES solver with preconditioner support
  Eigen::VectorXd solveGMRES(
      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);
  
  //! MINRES solver for symmetric matrices
  Eigen::VectorXd solveMINRES(
      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b);

  // ===== Helper template methods =====
  
  //! Configure solver parameters
  template <typename SolverType>
  void configureSolver(SolverType& solver);
  
  //! Solve with initial guess support
  template <typename SolverType>
  Eigen::VectorXd solveWithInitialGuess(
      SolverType& solver, const Eigen::VectorXd& b);
  
  //! Report solver status and convergence information
  template <typename SolverType>
  void reportSolverStatus(
      const SolverType& solver, const std::string& solver_name);
  
  //! Verify solution quality
  void verifySolution(
      const Eigen::SparseMatrix<double>& A,
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& b);

 protected:
  //! Solver type
  using SolverBase<Traits>::sub_solver_type_;
  //! Preconditioner type
  using SolverBase<Traits>::preconditioner_type_;
  //! Maximum number of iterations
  using SolverBase<Traits>::max_iter_;
  //! Tolerance
  using SolverBase<Traits>::tolerance_;
  //! Verbosity
  using SolverBase<Traits>::verbosity_;
  //! Drop tolerance for ILUT
  using SolverBase<Traits>::drop_tolerance_;
  //! Fill factor for ILUT
  using SolverBase<Traits>::fill_factor_;
  //! Restart iterations for GMRES
  using SolverBase<Traits>::restart_iterations_;
  //! Use initial guess flag
  using SolverBase<Traits>::use_initial_guess_;
  //! Use last solution flag
  using SolverBase<Traits>::use_last_solution_;
  //! Initial guess vector
  using SolverBase<Traits>::initial_guess_;
  //! Last solution vector
  using SolverBase<Traits>::last_solution_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
};
}  // namespace mpm

#include "iterative_eigen.tcc"

#endif  // MPM_ITERATIVE_EIGEN_