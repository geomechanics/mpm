#ifndef MPM_MATERIAL_NORSAND_H_
#define MPM_MATERIAL_NORSAND_H_

#include <cmath>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

namespace norsand {
//! Failure state
enum class FailureState { Elastic = 0, Yield = 1 };
}  // namespace norsand

//! NorSand class
//! \brief NorSand material model
//! \details NorSand material model with softening
//! \tparam Tdim Dimension
template <unsigned Tdim>
class NorSand : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  NorSand(unsigned id, const Json& material_properties);

  //! Destructor
  ~NorSand() override = default;

  //! Delete copy constructor
  NorSand(const NorSand&) = delete;

  //! Delete assignement operator
  NorSand& operator=(const NorSand&) = delete;

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

 protected:
  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

 private:
  //! Compute elastic tensor
  //! \param[in] stress Stress
  //! \param[in] state_vars History-dependent state variables
  Eigen::Matrix<double, 6, 6> compute_elastic_tensor(
      const Vector6d& stress, mpm::dense_map* state_vars);

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

  //! Compute stress invariants (p, q, lode_angle and M_theta)
  //! \param[in] stress Stress
  //! \param[in|out] p Mean stress
  //! \param[in|out] q Deviatoric stress
  //! \param[in|out] lode_angle Lode angle
  //! \param[in|out] M_theta Critical state M lode angle
  void compute_stress_invariants(const Vector6d& stress, double* p, double* q,
                                 double* lode_angle, double* M_theta);

  //! Compute image parameters (psi_image, chi_image, M_image, M_image_tc)
  //! \param[in] state_vars History-dependent state variables
  //! \retval computation of image parameters
  void compute_image_parameters(mpm::dense_map* state_vars);

  //! Compute state variables (void ratio, p_image, e_image, etc)
  //! \param[in] stress Stress
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] yield_type Yild type (elastic or yield)
  //! \retval status of computation of stress invariants
  void compute_state_variables(const Vector6d& stress, const Vector6d& dstrain,
                               mpm::dense_map* state_vars,
                               mpm::norsand::FailureState yield_type);

  //! Compute yield function and yield state
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] stress Stress
  //! \retval yield_type Yield type (elastic or yield)
  mpm::norsand::FailureState compute_yield_state(double* yield_function,
                                                 const Vector6d& stress,
                                                 mpm::dense_map* state_vars);

  //! Compute p_cohesion and p_dilation
  //! \param[in] state_vars History-dependent state variables
  //! \retval status of computation of stress invariants
  void compute_p_bond(mpm::dense_map* state_vars);

  //! Inline ternary function to check negative or zero numbers
  inline double check_low(double val) {
    return (val > 1.0e-15 ? val : 1.0e-15);
  }

  //! Inline ternary function to check number not greater than one
  inline double check_one(double val) { return (val < 1.0 ? val : 1.0); }

  //! Maximum reasonable (e_max - e_min) is 0.5 for M_image reduction
  double cap_{0.5};
  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Poisson ratio
  double poisson_ratio_{std::numeric_limits<double>::max()};
  //! Reference pressure pref
  double reference_pressure_{std::numeric_limits<double>::max()};
  //! Critical state friction angle
  double friction_cs_{std::numeric_limits<double>::max()};
  //! Critical state coefficient M in triaxial compression
  double Mtc_{std::numeric_limits<double>::max()};
  //! Critical state coefficient M in triaxial extension
  double Mte_{std::numeric_limits<double>::max()};
  //! Use bolton CSL line
  bool use_bolton_csl_{false};
  //! Volumetric coupling (dilatancy) parameter N
  double N_{std::numeric_limits<double>::max()};
  //! Minimum void ratio
  double e_min_{std::numeric_limits<double>::max()};
  //! Maximum void ratio
  double e_max_{std::numeric_limits<double>::max()};
  //! Crushing pressure
  double crushing_pressure_{std::numeric_limits<double>::max()};
  //! Lambda volumetric
  double lambda_{std::numeric_limits<double>::max()};
  //! Kappa swelling volumetric
  double kappa_{std::numeric_limits<double>::max()};
  //! Gamma void ratio at reference pressure
  double gamma_{std::numeric_limits<double>::max()};
  //! Dilatancy coefficient
  double chi_{std::numeric_limits<double>::max()};
  //! Dilatancy coefficient image
  double chi_image_{std::numeric_limits<double>::max()};
  //! Hardening modulus
  double hardening_modulus_{std::numeric_limits<double>::max()};
  //! Initial void ratio
  double void_ratio_initial_{std::numeric_limits<double>::max()};
  //! Initial image pressure
  double p_image_initial_{std::numeric_limits<double>::max()};
  //! Flag for bonded model
  bool bond_model_{false};
  //! Initial p_cohesion
  double p_cohesion_initial_{0.};
  //! Initial p_dilation
  double p_dilation_initial_{0.};
  //! Cohesion degradation parameter m upon shearing
  double m_cohesion_{0.};
  //! Dilation degradation parameter m upon shearing
  double m_dilation_{0.};
  //! Parameter for modulus
  double m_modulus_{0.};
  //! Flag to force stress ratio to converge to critical state
  bool force_critical_state_{false};
  //! Default tolerance
  double tolerance_{std::numeric_limits<double>::epsilon()};
  //! Failure state map
  std::map<int, mpm::norsand::FailureState> yield_type_ = {
      {0, mpm::norsand::FailureState::Elastic},
      {1, mpm::norsand::FailureState::Yield}};

};  // NorSand class
}  // namespace mpm

#include "norsand.tcc"

#endif  // MPM_MATERIAL_NORSAND_H_
