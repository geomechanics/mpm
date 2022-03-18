#ifndef MPM_MATERIAL_MOHR_COULOMB_H_
#define MPM_MATERIAL_MOHR_COULOMB_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

namespace mohrcoulomb {
//! Failure state
enum FailureState { Elastic = 0, Shear = 1, Tensile = 2 };
}  // namespace mohrcoulomb

//! MohrCoulomb class
//! \brief Mohr Coulomb material model
//! \details Mohr Coulomb material model with softening
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MohrCoulomb : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  MohrCoulomb(unsigned id, const Json& material_properties);

  //! Destructor
  ~MohrCoulomb() override{};

  //! Delete copy constructor
  MohrCoulomb(const MohrCoulomb&) = delete;

  //! Delete assignement operator
  MohrCoulomb& operator=(const MohrCoulomb&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override;

  //! State variables
  std::vector<std::string> state_variables() const override;

  //! Initialise material
  //! \brief Function that initialise material to be called at the beginning of
  //! time step
  void initialise(mpm::dense_map* state_vars) override {
    (*state_vars).at("yield_state") = 0;
  };

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
  mpm::mohrcoulomb::FailureState compute_yield_state(
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
  void compute_df_dp(mpm::mohrcoulomb::FailureState yield_type,
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
  //! \param[in] state_vars History-dependent state variables
  Matrix6x6 compute_elastic_tensor(mpm::dense_map* state_vars);

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_elasto_plastic_tensor(const Vector6d& stress,
                                          const Vector6d& dstrain,
                                          const ParticleBase<Tdim>* ptr,
                                          mpm::dense_map* state_vars,
                                          bool hardening = true) override;

  //! Inline ternary function to check negative or zero numbers
  inline double check_low(double val) {
    return (val > 1.0e-15 ? val : 1.0e-15);
  }

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
  //! Tension cutoff
  double tension_cutoff_{std::numeric_limits<double>::max()};
  //! softening
  bool softening_{false};
  //! Failure state map
  std::map<int, mpm::mohrcoulomb::FailureState> yield_type_ = {
      {0, mpm::mohrcoulomb::FailureState::Elastic},
      {1, mpm::mohrcoulomb::FailureState::Shear},
      {2, mpm::mohrcoulomb::FailureState::Tensile}};
};  // MohrCoulomb class
}  // namespace mpm

#include "mohr_coulomb.tcc"

#endif  // MPM_MATERIAL_MOHR_COULOMB_H_
