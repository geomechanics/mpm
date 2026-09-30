#ifndef MPM_MATERIAL_DPM_HAMAKER_H_
#define MPM_MATERIAL_DPM_HAMAKER_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "Eigen/Dense"

#include "dpm.h"
#include "infinitesimal_elasto_plastic.h"

namespace mpm {
namespace materials {

//! Nearest-neighbour surface-to-surface gap between grains from the packing
//! fraction, assuming uniform dilation from the contact packing fraction
//! \tparam Tdim Dimension
//! \param[in] phi Current packing fraction, solved for in dpmhamaker.tcc
//! \param[in] phi_contact Packing fraction at which grains are in contact
//! (DPM packing_fraction_dilation)
//! \param[in] grain_diameter Mean grain diameter
//! \retval gap Gap (>= 0), or max double if phi or phi_contact is
//! empty/negative
template <unsigned Tdim>
inline double hamaker_gap(double phi, double phi_contact,
                          double grain_diameter);

//! Hamaker (van der Waals) attractive force between two grains
//! \param[in] gap Surface-to-surface gap
//! \param[in] grain_diameter Mean grain diameter
//! \param[in] hamaker_const Hamaker constant A
//! \param[in] delta_min Minimum separation (contact cutoff, must be > 0)
//! \param[in] delta_max Maximum gap beyond which the force is zero
//! \retval force Attractive force (>= 0)
inline double hamaker_force(double gap, double grain_diameter,
                            double hamaker_const, double delta_min,
                            double delta_max);

}  // namespace materials
}  // namespace mpm

namespace mpm {

//! Dilatant plasticity model with Hamaker cohesion class
//! \brief DPM Hamaker material model
//! \details Dilatant plasticity model with Drucker-Prager yield criterion,
//! Mu(I)-Phi(I) rheology, restitution-derived viscoelasticity and Reynolds
//! dilatancy, with Hamaker (van der Waals) cohesion between grains.
//! \tparam Tdim Dimension
template <unsigned Tdim>
class DilatantPlasticHamakerModel : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  DilatantPlasticHamakerModel(unsigned id, const Json& material_properties);

  //! Destructor
  ~DilatantPlasticHamakerModel() override {};

  //! Delete copy constructor
  DilatantPlasticHamakerModel(const DilatantPlasticHamakerModel&) = delete;

  //! Delete assignement operator
  DilatantPlasticHamakerModel& operator=(const DilatantPlasticHamakerModel&) = delete;

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
  //! \param[in] dt Time step increment
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars, double dt) override;

 protected:
  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

  //! Compute elastic tensor
  Matrix6x6 compute_elastic_tensor();

  //! Compute visco-elastic tensor
  //! \param[in] dt Time step increment
  Matrix6x6 compute_visco_tensor(double dt);

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \param[in] lin_v Scheme dependent kinematic linearization parameter -
  //! velocity
  //! \param[in] lin_a Scheme dependent kinematic linearization parameter -
  //! acceleration
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_elasto_plastic_tensor(
      const Vector6d& stress, const Vector6d& dstrain,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt,
      double lin_v = 0.0, double lin_a = 0.0, bool hardening = true) override;

  //! Inline ternary function to check negative or zero numbers
  inline double check_low(double val) {
    return (val > 1.0e-15 ? val : 1.0e-15);
  }

  //! Initial packing fraction
  double initial_packing_fraction_{std::numeric_limits<double>::max()};
  //! Solid grain Density
  double grain_density_{std::numeric_limits<double>::max()};
  //! Youngs modulus
  double youngs_modulus_{std::numeric_limits<double>::max()};
  //! Bulk modulus
  double bulk_modulus_{std::numeric_limits<double>::max()};
  //! Shear modulus
  double shear_modulus_{std::numeric_limits<double>::max()};
  //! Poisson ratio
  double poisson_ratio_{std::numeric_limits<double>::max()};
  //! Bulk viscosity
  double bulk_viscosity_{0.0};
  //! Shear viscosity
  double shear_viscosity_{0.0};

  //! Friction parameter at zero strain rate
  double mu_s_{std::numeric_limits<double>::max()};
  //! Critical packing fraction
  double critical_packing_fraction_{std::numeric_limits<double>::max()};
  //! Minimum packing fraction
  double minimum_packing_fraction_{std::numeric_limits<double>::max()};
  //! Scaling constant (chi)
  double dilation_scaling_{std::numeric_limits<double>::max()};

  //! Rate dependency flag
  bool rate_dependent_{false};
  //! Mean particle size
  double d_{std::numeric_limits<double>::max()};
  //! Friction parameter at high strain rate
  double mu_2_{std::numeric_limits<double>::max()};
  // Friction I0 parameter
  double I0_{std::numeric_limits<double>::max()};
  // Dilation parameter a
  double dilation_a_{std::numeric_limits<double>::max()};

  //! Hamaker parameters
  //! Hamaker constant A (material specific, 0 means no cohesion)
  double hamaker_constant_{0.0};
  //! Minimum separation at contact (default 0.4 nm)
  double hamaker_delta_min_{4.0e-10};
  //! Maximum gap beyond which the Hamaker force is zero (default 100 nm)
  double hamaker_delta_max_{1.0e-7};

  //! Return mapping parameters
  //! Absolute tolerance
  double abs_tol_{1.e-10};
  //! Relative tolerance
  double rel_tol_{1.e-8};
  //! Maximum number of iterations
  unsigned max_iter_{15};

  //! Discrete tolerance
  double tolerance_{1.0e-10};

  //! Failure state map
  std::map<int, mpm::dpm::FailureState> yield_type_ = {
      {0, mpm::dpm::FailureState::Elastic},
      {1, mpm::dpm::FailureState::Shear},
      {2, mpm::dpm::FailureState::Apex},
      {3, mpm::dpm::FailureState::PartialPlastic},
      {4, mpm::dpm::FailureState::Separated}};
};  // DilatantPlasticHamakerModel class
}  // namespace mpm

#include "dpmhamaker.tcc"

#endif  // MPM_MATERIAL_DPM_HAMAKER_H_
