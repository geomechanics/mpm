#ifndef MPM_CONVERGENCE_CRITERION_BASE_H_
#define MPM_CONVERGENCE_CRITERION_BASE_H_

#include "logger.h"
#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

// MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

// Eigen
#include "Eigen/Dense"
#include <Eigen/Sparse>

namespace mpm {

// Convergence criterion base class
//! \brief Class which perform check of convergence
//! \details Given a certain tolerance, the class will return boolean to check
//! convergence of an iteration.
class ConvergenceCriterionBase {
 public:
  //! Constructor with two arguments
  ConvergenceCriterionBase(double tolerance, unsigned verbosity) {
    tolerance_ = tolerance;
    verbosity_ = verbosity;
    //! Logger
    std::string logger = "ConvergenceCriterionBase::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Constructor with three arguments
  ConvergenceCriterionBase(double tolerance, double abs_tolerance,
                           unsigned verbosity) {
    tolerance_ = tolerance;
    abs_tolerance_ = abs_tolerance;
    verbosity_ = verbosity;
    //! Logger
    std::string logger = "ConvergenceCriterionBase::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  // Virtual destructor
  virtual ~ConvergenceCriterionBase() = default;

  //! Copy constructor
  ConvergenceCriterionBase(const ConvergenceCriterionBase&) = default;

  //! Assignment operator
  ConvergenceCriterionBase& operator=(const ConvergenceCriterionBase&) =
      default;

  //! Move constructor
  ConvergenceCriterionBase(ConvergenceCriterionBase&&) = default;

  //! Set verbosity
  void set_verbosity(unsigned v) noexcept { verbosity_ = v; }

  //! Set relative iteration tolerance
  void set_tolerance(double tol) noexcept { tolerance_ = tol; }

  //! Set absolute iteration tolerance
  void set_abs_tolerance(double tol) noexcept { abs_tolerance_ = tol; }

  //! Assign global active dof for parallel vector
  void assign_global_active_dof(unsigned global_active_dof) noexcept {
    global_active_dof_ = global_active_dof;
  };

  //! Assign rank to global mapper for parallel vector
  //! \param[in] rank_global_mapper maps of local to global vector indices
  void assign_rank_global_mapper(
      const std::vector<int>& rank_global_mapper) noexcept {
    rank_global_mapper_ = rank_global_mapper;
  };

  //! Function to check convergence
  //! \param[in] vector vector of interest
  //! \param[in] save_settings boolean which determine if any saving of
  //! variables is necessary
  virtual inline bool check_convergence(const Eigen::VectorXd& vector,
                                        bool save_settings = false) = 0;

 protected:
  //! Logger
  std::shared_ptr<spdlog::logger> console_;
  //! Relative tolerance
  double tolerance_;
  //! Absolute tolerance
  double abs_tolerance_;
  //! Verbosity
  unsigned verbosity_{0};
  //! Global active dof
  unsigned global_active_dof_;
  //! Rank to Global mapper
  std::vector<int> rank_global_mapper_;
};
}  // namespace mpm

#endif  // MPM_CONVERGENCE_CRITERION_BASE_H_