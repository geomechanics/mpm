#ifndef MPM_CONVERGENCE_CRITERION_RESIDUAL_H_
#define MPM_CONVERGENCE_CRITERION_RESIDUAL_H_

#include "convergence_criterion_base.h"

#ifdef USE_PETSC
#include <petscksp.h>
#endif

namespace mpm {

// Convergence criterion of residual class
//! \brief Class which perform check of convergence of nonlinear iteration
//! residuals
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ConvergenceCriterionResidual
    : public mpm::ConvergenceCriterionBase<Tdim> {
 public:
  //! Constructor with two arguments
  ConvergenceCriterionResidual(double tolerance, unsigned verbosity)
      : mpm::ConvergenceCriterionBase<Tdim>(tolerance, verbosity) {
    abs_tolerance_ = tolerance;
    //! Logger
    std::string logger = "ConvergenceCriterionBase::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Constructor with three arguments
  ConvergenceCriterionResidual(double tolerance, double abs_tolerance,
                               unsigned verbosity)
      : mpm::ConvergenceCriterionBase<Tdim>(tolerance, abs_tolerance,
                                            verbosity) {
    //! Logger
    std::string logger = "ConvergenceCriterionResidual::";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  // Virtual destructor
  virtual ~ConvergenceCriterionResidual() = default;

  //! Copy constructor
  ConvergenceCriterionResidual(const ConvergenceCriterionResidual<Tdim>&) =
      default;

  //! Assignment operator
  ConvergenceCriterionResidual& operator=(
      const ConvergenceCriterionResidual<Tdim>&) = default;

  //! Move constructor
  ConvergenceCriterionResidual(ConvergenceCriterionResidual<Tdim>&&) = default;

  //! Function to check convergence
  //! \param[in] residual_vector Residual vector of interest
  //! \param[in] initial Boolean to indentify 1st (true) iteration
  bool check_convergence(const Eigen::VectorXd& residual_vector,
                         bool initial = false) override;

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
  //! Initial residuals
  double initial_residual_norm_;
};
}  // namespace mpm

#include "convergence_criterion_residual.tcc"
#endif  // MPM_CONVERGENCE_CRITERION_RESIDUAL_H_