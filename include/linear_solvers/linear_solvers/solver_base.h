// ========================================
// File: solver_base.h
// ========================================
#ifndef MPM_SOLVER_BASE_H_
#define MPM_SOLVER_BASE_H_

#include "data_types.h"
#include "logger.h"
#include <chrono>

namespace mpm {
template <typename Traits>
class SolverBase {
 public:
  //! Constructor with min and max iterations and tolerance
  //! \param[in] max_iter Maximum number of iterations
  //! \param[in] tolerance Tolerance for solver to achieve convergence
  SolverBase(unsigned max_iter, double tolerance) {
    max_iter_ = max_iter;
    tolerance_ = tolerance;
    //! Logger
    std::string logger = "SolverBase::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  };

  //! Destructor
  virtual ~SolverBase(){};

  //! Matrix solver with default initial guess
  virtual Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A,
                                const Eigen::VectorXd& b) = 0;

  //! Assign global active dof
  virtual void assign_global_active_dof(unsigned global_active_dof) = 0;

  //! Assign rank to global mapper
  virtual void assign_rank_global_mapper(
      const std::vector<int>& rank_global_mapper) = 0;

  //! Set sub solver type
  void set_sub_solver_type(const std::string& type) noexcept {
    sub_solver_type_ = type;
  }

  //! Set preconditioner type
  void set_preconditioner_type(const std::string& type) noexcept {
    preconditioner_type_ = type;
  }

  //! Set maximum number of iterations
  void set_max_iteration(unsigned max_iter) noexcept { max_iter_ = max_iter; }

  //! Set relative iteration tolerance
  void set_tolerance(double tol) noexcept { tolerance_ = tol; }

  //! Set absolute iteration tolerance
  void set_abs_tolerance(double tol) noexcept { abs_tolerance_ = tol; }

  //! Set divergence iteration tolerance
  void set_div_tolerance(double tol) noexcept { div_tolerance_ = tol; }

  //! Set verbosity
  void set_verbosity(unsigned v) noexcept { verbosity_ = v; }

  //! Set drop tolerance for ILUT preconditioner
  void set_drop_tolerance(double tol) noexcept { drop_tolerance_ = tol; }

  //! Set fill factor for ILUT preconditioner
  void set_fill_factor(int factor) noexcept { fill_factor_ = factor; }

  //! Set restart iterations for GMRES
  void set_restart_iterations(int restart) noexcept { 
    restart_iterations_ = restart; 
  }

  //! Enable/disable use of initial guess
  void set_use_initial_guess(bool use) noexcept { 
    use_initial_guess_ = use; 
  }

  //! Enable/disable use of last solution as initial guess
  void set_use_last_solution(bool use) noexcept { 
    use_last_solution_ = use; 
  }

  //! Set initial guess vector
  void set_initial_guess(const Eigen::VectorXd& guess) {
    initial_guess_ = guess;
    use_initial_guess_ = true;
  }

  //! Clear stored last solution
  void clear_last_solution() { last_solution_.resize(0); }

 protected:
  //! Solver type
  std::string sub_solver_type_;
  //! Preconditioner type
  std::string preconditioner_type_{"none"};
  //! Maximum number of iterations
  unsigned max_iter_;
  //! Relative tolerance
  double tolerance_;
  //! Absolute tolerance
  double abs_tolerance_{1e-10};
  //! Divergence tolerance
  double div_tolerance_{1e10};
  //! Verbosity
  unsigned verbosity_{0};
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  
  // ===== Extended parameters for preconditioners and solvers =====
  //! Drop tolerance for ILUT preconditioner
  double drop_tolerance_{1e-5};
  //! Fill factor for ILUT preconditioner
  int fill_factor_{30};
  //! Restart iterations for GMRES
  int restart_iterations_{1000};
  //! Use initial guess flag
  bool use_initial_guess_{false};
  //! Use last solution as initial guess
  bool use_last_solution_{false};
  //! User-provided initial guess
  Eigen::VectorXd initial_guess_;
  //! Stored last solution for reuse
  Eigen::VectorXd last_solution_;
};
}  // namespace mpm

#endif  // MPM_SOLVER_BASE_H_