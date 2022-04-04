#ifndef MPM_MATERIAL_FINITE_MOHR_COULOMB_H_
#define MPM_MATERIAL_FINITE_MOHR_COULOMB_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "finite_elasto_plastic.h"
#include "mohr_coulomb.h"

namespace mpm {

//! MohrCoulomb class in Finite Strain framework
//! \brief Mohr Coulomb material model
//! \details Mohr Coulomb material model with softening
//! \tparam Tdim Dimension
template <unsigned Tdim>
class FiniteMohrCoulomb : public FiniteElastoPlastic<Tdim> {
 public:
  //! Define a vector of 3 dof
  using Vector3d = Eigen::Matrix<double, 3, 1>;
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 3 x 3
  using Matrix3x3 = Eigen::Matrix<double, 3, 3>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  FiniteMohrCoulomb(unsigned id, const Json& material_properties);

  //! Destructor
  ~FiniteMohrCoulomb() override{};

  //! Delete copy constructor
  FiniteMohrCoulomb(const FiniteMohrCoulomb&) = delete;

  //! Delete assignement operator
  FiniteMohrCoulomb& operator=(const FiniteMohrCoulomb&) = delete;

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

 private:
  //! Compute updated kirchhoff stress in principal coordinate system by
  //! performing elasto-plastic return mapping
  //! \param[in, out] principal_elastic_hencky_strain Elastic strain in
  //! principal coordinates
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of kirchhoff stress in principal
  //! coordinate
  Vector3d compute_return_mapping(Vector3d& principal_elastic_hencky_strain,
                                  const ParticleBase<Tdim>* ptr,
                                  mpm::dense_map* state_vars) override {
    return Vector3d::Zero();
  };

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] elastic_left_cauchy_green Updated elastic_left_cauchy_green
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_elasto_plastic_tensor(
      const Vector6d& stress, const Matrix3x3& elastic_left_cauchy_green,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars,
      bool hardening = true) override;

  //! Compute elastic tensor
  //! \param[in] state_vars History-dependent state variables
  Matrix6x6 compute_elastic_tensor(mpm::dense_map* state_vars);

  //! Inline ternary function to check negative or zero numbers
  inline double check_low(double val) {
    return (val > 1.0e-15 ? val : 1.0e-15);
  }

  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

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
};  // FiniteMohrCoulomb class
}  // namespace mpm

#include "finite_mohr_coulomb.tcc"

#endif  // MPM_MATERIAL_FINITE_MOHR_COULOMB_H_
