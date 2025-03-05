#ifndef MPM_MATERIAL_BINGHAM_VISCOPLASTIC_H_
#define MPM_MATERIAL_BINGHAM_VISCOPLASTIC_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

namespace bingham_viscoplastic {
//! Failure state
enum FailureState { Elastic = 0, Shear = 1 };
}  // namespace bingham_viscoplastic

//! Bingham elasto-viscoplastic model class
//! \brief Bingham elasto-viscoplastic rheology for Non-newtonian visco-plastic
//! material
//! \details Bingham elasto-viscoplasticity
//! \tparam Tdim Dimension
template <unsigned Tdim>
class BinghamViscoPlastic : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  BinghamViscoPlastic(unsigned id, const Json& material_properties);

  //! Destructor
  ~BinghamViscoPlastic() override{};

  //! Delete copy constructor
  BinghamViscoPlastic(const BinghamViscoPlastic&) = delete;

  //! Delete assignement operator
  BinghamViscoPlastic& operator=(const BinghamViscoPlastic&) = delete;

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
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_elasto_plastic_tensor(const Vector6d& stress,
                                          const Vector6d& dstrain,
                                          const ParticleBase<Tdim>* ptr,
                                          mpm::dense_map* state_vars, double dt,
                                          bool hardening = true) override;

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
  //! Dynamic viscosity
  double dynamic_viscosity_{std::numeric_limits<double>::max()};
  //! Tau0
  double tau0_{std::numeric_limits<double>::max()};
  //! Initial flocculation state
  double lambda0_{std::numeric_limits<double>::max()};
  //! Flocullation parameter
  double athix_{std::numeric_limits<double>::max()};
  //! Thixotropy deflocculation rate
  double alpha_{std::numeric_limits<double>::max()};

  //! Return mapping parameters
  //! Absolute tolerance
  double abs_tol_{1.e-10};
  //! Relative tolerance
  double rel_tol_{1.e-8};
  //! Maximum number of iterations
  unsigned max_iter_{15};

  //! Discrete tolerance
  double tolerance_{1.0e-15};

  //! Failure state map
  std::map<int, mpm::bingham_viscoplastic::FailureState> yield_type_ = {
      {0, mpm::bingham_viscoplastic::FailureState::Elastic},
      {1, mpm::bingham_viscoplastic::FailureState::Shear}};
};  // BinghamViscoPlastic class
}  // namespace mpm

#include "bingham_viscoplastic.tcc"

#endif  // MPM_MATERIAL_BINGHAM_VISCOPLASTIC_H_