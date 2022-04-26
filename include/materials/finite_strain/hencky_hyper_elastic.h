#ifndef MPM_MATERIAL_HENCKY_HYPER_ELASTIC_H_
#define MPM_MATERIAL_HENCKY_HYPER_ELASTIC_H_

#include <limits>

#include "Eigen/Dense"

#include "material.h"
#include "math_utility.h"

namespace mpm {

//! HenckyHyperElastic class
//! \brief Hencky Hyper Elastic material model
//! \details HenckyHyperElastic class stresses and strains
//! \tparam Tdim Dimension
template <unsigned Tdim>
class HenckyHyperElastic : public Material<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id
  //! \param[in] material_properties Material properties
  HenckyHyperElastic(unsigned id, const Json& material_properties);

  //! Destructor
  ~HenckyHyperElastic() override{};

  //! Delete copy constructor
  HenckyHyperElastic(const HenckyHyperElastic&) = delete;

  //! Delete assignement operator
  HenckyHyperElastic& operator=(const HenckyHyperElastic&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override {
    mpm::dense_map state_vars;
    return state_vars;
  }

  //! State variables
  std::vector<std::string> state_variables() const override { return {}; }

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] deformation_gradient Deformation gradient at the current step
  //! \param[in] deformation_gradient_increment Deformation gradient increment
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(
      const Vector6d& stress,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient_increment,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) override;

  //! Compute consistent tangent matrix
  //! \param[in] stress Updated stress
  //! \param[in] prev_stress Stress at the current step
  //! \param[in] deformation_gradient Deformation gradient at the current step
  //! \param[in] deformation_gradient_increment Deformation gradient increment
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_consistent_tangent_matrix(
      const Vector6d& stress, const Vector6d& prev_stress,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient_increment,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) override;

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

 private:
  //! Elastic stiffness matrix
  Matrix6x6 de_;
  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Youngs modulus
  double youngs_modulus_{std::numeric_limits<double>::max()};
  //! Poisson ratio
  double poisson_ratio_{std::numeric_limits<double>::max()};
  //! mu of Lame's constants
  double mu_{std::numeric_limits<double>::max()};
  //! lambda of Lame's constants
  double lambda_{std::numeric_limits<double>::max()};
};  // HenckyHyperElastic class
}  // namespace mpm

#include "hencky_hyper_elastic.tcc"

#endif  // MPM_MATERIAL_HENCKY_HYPER_ELASTIC_H