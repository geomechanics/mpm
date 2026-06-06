#ifndef MPM_MATERIAL_HB_VISCOPLASTIC_H_
#define MPM_MATERIAL_HB_VISCOPLASTIC_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

namespace hb_viscoplastic {
//! Failure state
enum FailureState { Elastic = 0, Shear = 1 };
}  // namespace hb_viscoplastic

//! Herschel-Bulkley elasto-viscoplastic model class
//! \brief Herschel-Bulkley elasto-viscoplastic rheology for Non-newtonian
//! visco-plastic material
//! \details Herschel-Bulkley elasto-viscoplasticity
//! \tparam Tdim Dimension
template <unsigned Tdim>
class HerschelBulkleyViscoPlastic : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  HerschelBulkleyViscoPlastic(unsigned id, const Json& material_properties);

  //! Destructor
  ~HerschelBulkleyViscoPlastic() override {};

  //! Delete copy constructor
  HerschelBulkleyViscoPlastic(const HerschelBulkleyViscoPlastic&) = delete;

  //! Delete assignement operator
  HerschelBulkleyViscoPlastic& operator=(const HerschelBulkleyViscoPlastic&) =
      delete;

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

 private:
  //! Compute elastic tensor
  //! \param[in] vol_strain Total volumetric strain
  Matrix6x6 compute_elastic_tensor(double vol_strain);

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

  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Bulk modulus
  double bulk_modulus_{std::numeric_limits<double>::max()};
  //! Shear modulus
  double shear_modulus_{std::numeric_limits<double>::max()};

  //! Volumetric wave speed
  double c_{std::numeric_limits<double>::max()};
  //! Gamma
  double gamma_{std::numeric_limits<double>::max()};
  //! Consistency parameter for Herschel-Bulkley model
  double k_{std::numeric_limits<double>::max()};
  //! Flow index for Herschel-Bulkley model
  double n_{std::numeric_limits<double>::max()};
  //! peak tau for Herschel-Bulkley model
  double tau_peak_{std::numeric_limits<double>::max()};
  //! residual tau for Herschel-Bulkley model
  double tau_residual_{std::numeric_limits<double>::max()};
  //! Exponential degradation parameter
  double beta_{std::numeric_limits<double>::max()};

  //! Return mapping parameters
  //! Absolute tolerance
  double abs_tol_{1.e-10};
  //! Relative tolerance
  double rel_tol_{1.e-8};
  //! Maximum number of iterations
  unsigned max_iter_{15};
  //! Tolerance for flow rule
  double tolerance_{1.e-10};

  //! Failure state map
  std::map<int, mpm::hb_viscoplastic::FailureState> yield_type_ = {
      {0, mpm::hb_viscoplastic::FailureState::Elastic},
      {1, mpm::hb_viscoplastic::FailureState::Shear}};
};  // HerschelBulkleyViscoPlastic class
}  // namespace mpm

#include "herschel_bulkley_viscoplastic.tcc"

#endif  // MPM_MATERIAL_HB_VISCOPLASTIC_H_