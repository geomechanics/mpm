#ifndef MPM_MATERIAL_MOHR_COULOMB_SFD_H_
#define MPM_MATERIAL_MOHR_COULOMB_SFD_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

namespace mohrcoulombsfd {
//! Failure state
enum FailureState { Elastic, Tensile, Shear };
}  // namespace mohrcoulombsfd

//! MohrCoulombSFD class
//! \brief Mohr Coulomb material model
//! \details Mohr Coulomb material model with softening
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MohrCoulombSFD : public Material<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  MohrCoulombSFD(unsigned id, const Json& material_properties);

  //! Destructor
  ~MohrCoulombSFD() override{};

  //! Delete copy constructor
  MohrCoulombSFD(const MohrCoulombSFD&) = delete;

  //! Delete assignement operator
  MohrCoulombSFD& operator=(const MohrCoulombSFD&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override;

  //! State variables
  std::vector<std::string> state_variables() const override;

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars) override;

  //! Compute stress invariants (j2, j3, rho, theta, and epsilon)
  //! \param[in] stress Stress
  //! \param[in] state_vars History-dependent state variables
  //! \retval status of computation of stress invariants
  bool compute_stress_invariants(const Vector6d& stress,
                                 mpm::dense_map* state_vars);

  //! Compute yield function and yield state
  //! \param[in] state_vars History-dependent state variables
  //! \retval yield_type Yield type (elastic, shear or tensile)
  mpm::mohrcoulombsfd::FailureState compute_yield_state(
      Eigen::Matrix<double, 2, 1>* yield_function,
      const mpm::dense_map& state_vars);

  //! Compute dF/dSigma and dP/dSigma
  //! \param[in] yield_type Yield type (elastic, shear or tensile)
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] stress Stress
  //! \param[in] df_dsigma dF/dSigma
  //! \param[in] dp_dsigma dP/dSigma
  //! \param[in] dp_dq dP / dq
  //! \param[in] softening Softening parameter
  void compute_df_dp(mpm::mohrcoulombsfd::FailureState yield_type,
                     const mpm::dense_map* state_vars, const Vector6d& stress,
                     Vector6d* df_dsigma, Vector6d* dp_dsigma, double* dp_dq,
                     double* softening);

 protected:
  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

 private:
  //! Compute elastic tensor
  bool compute_elastic_tensor();

  //! Elastic stiffness matrix
  Matrix6x6 de_;
  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Youngs modulus
  double youngs_modulus_{std::numeric_limits<double>::max()};
  //! Bulk modulus
  double bulk_modulus_{std::numeric_limits<double>::max()};
  //! Shear modulus
  double shear_modulus_{std::numeric_limits<double>::max()};
  //! Poisson ratio
  double poisson_ratio_{std::numeric_limits<double>::max()};
  //! Maximum friction angle phi
  double phi_peak_{std::numeric_limits<double>::max()};
  //! Maximum dilation angle psi
  double psi_peak_{std::numeric_limits<double>::max()};
  //! Undrained friction angle from drained strength
  double phi_undrained_{std::numeric_limits<double>::max()};
  //! Maximum cohesion
  double cohesion_peak_{std::numeric_limits<double>::max()};
  //! Residual friction angle phi
  double phi_residual_{std::numeric_limits<double>::max()};
  //! Residual dilation angle psi
  double psi_residual_{std::numeric_limits<double>::max()};
  //! Residual cohesion
  double cohesion_residual_{std::numeric_limits<double>::max()};
  //! Peak plastic deviatoric strain
  double pdstrain_peak_{std::numeric_limits<double>::max()};
  //! Residual plastic deviatoric strain
  double pdstrain_residual_{std::numeric_limits<double>::max()};
  //! SPT-N bool
  bool sptn_bool_{false};
  //! SPT-N
  double sptn_{std::numeric_limits<double>::max()};
  //! su_over_p_bool
  bool su_over_p_bool_{false};
  //! Maximum su_over_p
  double su_over_pi_peak_{std::numeric_limits<double>::max()};
  //! Residual su_over_p
  double su_over_pi_residual_{std::numeric_limits<double>::max()};
  //! mc_to_tresca_ bool
  bool mc_to_tresca_bool_{false};
  //! Tension cutoff
  double tension_cutoff_{std::numeric_limits<double>::max()};
  //! softening
  bool softening_{false};
};  // MohrCoulombSFD class
}  // namespace mpm

#include "mohr_coulomb_sfd.tcc"

#endif  // MPM_MATERIAL_MOHR_COULOMB_SFD_H_
