#ifndef MPM_MATERIAL_LINEAR_ELASTIC_PML_H_
#define MPM_MATERIAL_LINEAR_ELASTIC_PML_H_

#include <limits>

#include "Eigen/Dense"

#include "material.h"

namespace mpm {

//! LinearElasticPML class for Perfectly Matched Layer (PML) particle
//! \brief Linear Elastic material model for Perfectly Matched Layer (PML)
//! particle
//! \details LinearElasticPML class stresses and strains for Perfectly Matched
//! Layer (PML)
//! \tparam Tdim Dimension
template <unsigned Tdim>
class LinearElasticPML : public LinearElastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id
  //! \param[in] material_properties Material properties
  LinearElasticPML(unsigned id, const Json& material_properties);

  //! Destructor
  ~LinearElasticPML() override{};

  //! Delete copy constructor
  LinearElasticPML(const LinearElasticPML&) = delete;

  //! Delete assignement operator
  LinearElasticPML& operator=(const LinearElasticPML&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override;

  //! State variables
  std::vector<std::string> state_variables() const override;

 protected:
  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

 private:
  //! Density
  double density_{std::numeric_limits<double>::max()};
  //! Youngs modulus
  double youngs_modulus_{std::numeric_limits<double>::max()};
  //! Poisson ratio
  double poisson_ratio_{std::numeric_limits<double>::max()};
  //! Lame's lambda
  double lambda_{std::numeric_limits<double>::max()};
  //! Shear modulus
  double shear_modulus_{std::numeric_limits<double>::max()};
  //! Maximum damping ratio
  double alpha_{std::numeric_limits<double>::max()};
  //! Damping power
  double dpower_{0.};
};  // LinearElasticPML class
}  // namespace mpm

#include "linear_elastic_pml.tcc"

#endif  // MPM_MATERIAL_LINEAR_ELASTIC_PML_H_
