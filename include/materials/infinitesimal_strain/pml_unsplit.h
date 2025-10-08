#ifndef MPM_MATERIAL_PML_UNSPLIT_H_
#define MPM_MATERIAL_PML_UNSPLIT_H_

#include <limits>

#include "Eigen/Dense"

#include "material.h"

namespace mpm {

//! UnsplitPML class for Perfectly Matched Layer (PML) particle
//! \brief Linear Elastic material model for Perfectly Matched Layer (PML)
//! particle
//! \details UnsplitPML class stresses and strains for Perfectly Matched
//! Layer (PML)
//! \tparam Tdim Dimension
template <unsigned Tdim>
class UnsplitPML : public LinearElastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id
  //! \param[in] material_properties Material properties
  UnsplitPML(unsigned id, const Json& material_properties);

  //! Destructor
  ~UnsplitPML() override{};

  //! Delete copy constructor
  UnsplitPML(const UnsplitPML&) = delete;

  //! Delete assignement operator
  UnsplitPML& operator=(const UnsplitPML&) = delete;

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
  //! P wave velocity
  double vp_{std::numeric_limits<double>::max()};
  //! Damping ratio
  double damping_ratio_{std::numeric_limits<double>::max()};
  //! Reflection coefficient
  double reflec_coeff_{std::numeric_limits<double>::max()};
  //! Characteristic element length
  double h_char_{std::numeric_limits<double>::max()};
  //! Damping power
  double dpower_{0.};
};  // UnsplitPML class
}  // namespace mpm

#include "pml_unsplit.tcc"

#endif  // MPM_MATERIAL_PML_UNSPLIT_H_
