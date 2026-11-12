#ifndef MPM_MATERIAL_MOHR_COULOMB_H_
#define MPM_MATERIAL_MOHR_COULOMB_H_

#include <cmath>
#include <limits>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

namespace mohrcoulomb {
//! Failure state
enum FailureState { Elastic = 0, Shear = 1, Tensile = 2, Separated = 3 };
}  // namespace mohrcoulomb

//! MohrCoulomb class
//! \brief Mohr-Coulomb material model with tension cutoff and softening
//! \details Shear: Mohr-Coulomb yield surface with a hyperbolic
//! Menetrey-Willam plastic potential. Tension: Rankine cutoff with a
//! hyperbolic elliptic plastic potential. Stresses are integrated with an
//! implicit (closest-point) return mapping, with corner (two-surface) returns
//! and adaptive strain substepping.
//! Material properties (JSON):
//!   density, youngs_modulus, poisson_ratio,
//!   friction, dilation, cohesion (peak values, angles in degrees),
//!   residual_friction, residual_dilation, residual_cohesion,
//!   peak_pdstrain, residual_pdstrain, softening (bool),
//!   tension_cutoff,
//!   packing_fraction or porosity, packing_fraction_minimum (optional),
//!   apex_smoothing (optional, stress units, default 0)
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MohrCoulomb : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] id Material id
  //! \param[in] material_properties Material properties
  MohrCoulomb(unsigned id, const Json& material_properties);

  //! Destructor
  ~MohrCoulomb() override = default;

  //! Delete copy constructor
  MohrCoulomb(const MohrCoulomb&) = delete;

  //! Delete assignment operator
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

  //! Compute stress (with adaptive strain substepping)
  //! \param[in] stress Stress at the start of the step
  //! \param[in] dstrain Strain increment
  //! \param[in] ptr Constant pointer to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars, double dt) override;

  //! Compute stress invariants and store them in the state variables
  //! ("pressure" = -p, "tau" = sqrt(J2), "theta" = Lode angle)
  //! \param[in] stress Stress
  //! \param[in] state_vars History-dependent state variables
  //! \retval status of computation of stress invariants
  bool compute_stress_invariants(const Vector6d& stress,
                                 mpm::dense_map* state_vars);

  //! Compute yield functions and yield state
  //! \param[out] yield_function Yield functions (0: tension, 1: shear)
  //! \param[in] state_vars History-dependent state variables
  //! \retval yield_type Yield type (elastic, shear, tensile or shear-tensile)
  mpm::mohrcoulomb::FailureState compute_yield_state(
      Eigen::Matrix<double, 2, 1>* yield_function,
      const mpm::dense_map& state_vars);

  //! Compute yield and flow derivatives for one surface
  //! \param[in] yield_type Surface (shear or tensile)
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] stress Stress
  //! \param[out] df_dsigma dF/dSigma
  //! \param[out] dp_dsigma dP/dSigma
  //! \param[out] dp_dq dP/dq
  //! \param[out] d2p_dsigma2 d2P/dSigma2
  //! \param[out] softening Softening modulus H
  //! \param[in] pdstrain Accumulated plastic deviatoric strain (kappa)
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
  //! Stress update for a single strain increment (driven by compute_stress)
  //! \param[in] stress Stress at the start of the increment
  //! \param[in] dstrain Strain increment
  //! \param[in] ptr Constant pointer to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time increment
  //! \param[in] allow_apex Permit the direct apex return as a last resort
  //! \param[out] success False if the return mapping did not converge
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress_increment(const Vector6d& stress,
                                    const Vector6d& dstrain,
                                    const ParticleBase<Tdim>* ptr,
                                    mpm::dense_map* state_vars, double dt,
                                    bool allow_apex, bool* success);

  //! Compute elastic tensor
  //! \param[in] state_vars History-dependent state variables
  Matrix6x6 compute_elastic_tensor(mpm::dense_map* state_vars);

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] ptr Constant pointer to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations matrix
  Matrix6x6 compute_elasto_plastic_tensor(const Vector6d& stress,
                                          const Vector6d& dstrain,
                                          const ParticleBase<Tdim>* ptr,
                                          mpm::dense_map* state_vars, double dt,
                                          bool hardening = true) override;

  //! Inline ternary function to check negative or zero numbers
  inline double check_low(double val) {
    return (val > 1.0e-15 ? val : 1.0e-15);
  }

  //! Set the per-particle cohesion from the initial stress state ("initial"
  //! cohesion type); runs once, on the particle's first update
  //! \param[in] stress Initial (start-of-step) stress
  //! \param[in] ptr Particle (for the coordinates if a phreatic level is set)
  //! \param[in] state_vars History-dependent state variables
  void initialise_cohesion(const Vector6d& stress,
                           const ParticleBase<Tdim>* ptr,
                           mpm::dense_map* state_vars);

  //! Peak cohesion of a particle (falls back to the material value until
  //! captured)
  double peak_cohesion(const mpm::dense_map& state_vars) const {
    if (!cohesion_initial_) return cohesion_peak_;
    const double c = state_vars.at("cohesion_peak");
    return (c < 0.) ? cohesion_peak_ : c;
  }

  //! Residual cohesion of a particle
  double residual_cohesion(const mpm::dense_map& state_vars) const {
    return cohesion_initial_
               ? residual_cohesion_ratio_ * peak_cohesion(state_vars)
               : cohesion_residual_;
  }

  void update_softening_parameters(mpm::dense_map* state_vars);
  std::tuple<Vector6d, double, bool> compute_single_surface_return(
      mpm::mohrcoulomb::FailureState yield_type,
      const Eigen::Matrix<double, 2, 1>& yield_function,
      const Vector6d& current_stress, const Matrix6x6& de,
      mpm::dense_map* state_vars);

  //! Compute corner return mapping for multi-surface failure
  std::tuple<Vector6d, double, bool> compute_corner_return(
      const Vector6d& current_stress, const Matrix6x6& de,
      mpm::dense_map* state_vars);

  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Grain density
  double grain_density_{std::numeric_limits<double>::max()};
  //! Minimum packing fraction
  double minimum_packing_fraction_{0.0};

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
  //! Cohesion set per particle from the initial mean effective stress
  bool cohesion_initial_{false};
  //! Cohesion at zero reference mean effective stress
  double cohesion_intercept_{0.};
  //! Increase of cohesion per unit reference mean effective stress
  double cohesion_slope_{0.};
  //! Lower and upper bounds of the per-particle cohesion
  double cohesion_min_{0.};
  double cohesion_max_{std::numeric_limits<double>::max()};
  //! Residual cohesion as a fraction of the per-particle peak cohesion
  double residual_cohesion_ratio_{1.};
  //! Subtract hydrostatic pore pressure below a phreatic level
  bool pore_pressure_correction_{false};
  //! Phreatic points (x, y coordinates)
  std::vector<std::pair<double, double>> phreatic_points_;
  //! Phreatic level (coordinate along vertical_direction_)
  double phreatic_level_{0.};
  //! Unit weight of water (stress units per length)
  double water_unit_weight_{9810.};
  //! Coordinate index of the vertical (upward) direction
  unsigned vertical_direction_{Tdim - 1};
  //! Peak plastic deviatoric strain
  double pdstrain_peak_{std::numeric_limits<double>::max()};
  //! Residual plastic deviatoric strain
  double pdstrain_residual_{std::numeric_limits<double>::max()};
  //! Tension cutoff
  double tension_cutoff_{std::numeric_limits<double>::max()};
  //! Minimum hyperbolic smoothing of the plastic potentials near the apex
  //! (stress units); needed for low or zero cohesion / tension cutoff
  double apex_smoothing_{0.};
  //! softening
  bool softening_{false};
  //! Failure state map
  std::map<int, mpm::mohrcoulomb::FailureState> yield_type_ = {
      {0, mpm::mohrcoulomb::FailureState::Elastic},
      {1, mpm::mohrcoulomb::FailureState::Shear},
      {2, mpm::mohrcoulomb::FailureState::Tensile},
      {3, mpm::mohrcoulomb::FailureState::Separated}};
};  // MohrCoulomb class
}  // namespace mpm

#include "mohr_coulomb.tcc"

#endif  // MPM_MATERIAL_MOHR_COULOMB_H_