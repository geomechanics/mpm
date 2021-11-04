#ifndef MPM_CONVERGENCE_CRITERION_SOLUTION_H_
#define MPM_CONVERGENCE_CRITERION_SOLUTION_H_

#include "convergence_criterion_base.h"

#ifdef USE_PETSC
#include <petscksp.h>
#endif

namespace mpm {

// Convergence criteria of solution class
//! \brief Class which perform check of convergence of nonlinear iteration
//! solution
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ConvergenceCriterionSolution
    : public mpm::ConvergenceCriterionBase<Tdim> {
 public:
  //! Constructor
  ConvergenceCriterionSolution(double tolerance, unsigned verbosity)
      : mpm::ConvergenceCriterionBase<Tdim>(tolerance, verbosity) {
    //! Logger
    std::string logger = "ConvergenceCriterionSolution::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  // Virtual destructor
  virtual ~ConvergenceCriterionSolution() = default;

  //! Copy constructor
  ConvergenceCriterionSolution(const ConvergenceCriterionSolution<Tdim>&) =
      default;

  //! Assignment operator
  ConvergenceCriterionSolution& operator=(
      const ConvergenceCriterionSolution<Tdim>&) = default;

  //! Move constructor
  ConvergenceCriterionSolution(ConvergenceCriterionSolution<Tdim>&&) = default;

  //! Function to check convergence
  //! \param[in] solution_vector Solution vector of interest
  //! \param[in] save_settings Unused boolean settings
  bool check_convergence(const Eigen::VectorXd& solution_vector,
                         bool save_settings = false) override;

 protected:
  //! Logger
  std::shared_ptr<spdlog::logger> console_;
  //! Relative tolerance
  using ConvergenceCriterionBase<Tdim>::tolerance_;
  //! Absolute tolerance
  using ConvergenceCriterionBase<Tdim>::abs_tolerance_;
  //! Verbosity
  using ConvergenceCriterionBase<Tdim>::verbosity_;
  //! Global active dof
  using ConvergenceCriterionBase<Tdim>::global_active_dof_;
  //! Rank to Global mapper
  using ConvergenceCriterionBase<Tdim>::rank_global_mapper_;
};
}  // namespace mpm

#include "convergence_criterion_solution.tcc"
#endif  // MPM_CONVERGENCE_CRITERION_SOLUTION_H_