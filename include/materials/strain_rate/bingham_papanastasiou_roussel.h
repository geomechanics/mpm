#ifndef MPM_MATERIAL_BINGHAM_PapanastasiouRousselPAPANASTASIOUS_ROUSSEL_H_
#define MPM_MATERIAL_BINGHAM_PAPANASTASIOUS_ROUSSEL_H_

#include <limits>

#include "Eigen/Dense"

#include "material.h"

namespace mpm {

//! Papanastasiou-Roussel Bingham class
//! \brief Papanastasiou-Roussel Bingham fluid material model
//! \details Papanastasiou-Roussel Bingham class stresses and strains
//! \tparam Tdim Dimension
template <unsigned Tdim>
class BinghamPapanastasiouRoussel : public Material<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] id Material ID
  //! \param[in] material_properties Material properties
  BinghamPapanastasiouRoussel(unsigned id, const Json& material_properties);

  //! Destructor
  ~BinghamPapanastasiouRoussel() override{};

  //! Delete copy constructor
  BinghamPapanastasiouRoussel(const BinghamPapanastasiouRoussel&) = delete;

  //! Delete assignement operator
  BinghamPapanastasiouRoussel& operator = 
                            (const BinghamPapanastasiouRoussel&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override;

  //! State variables
  std::vector<std::string> state_variables() const override;

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
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
  //! Dirac delta function in Voigt notation
  Eigen::Matrix<double, 6, 1> dirac_delta() const;


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
  //! Rest time
  double rt_{std::numeric_limits<double>::max()};
  //! Regularization shape factor m
  double m_{std::numeric_limits<double>::max()};
  //! Critical yielding shear rate
  double critical_shear_rate_{std::numeric_limits<double>::max()};
  //! Compressibility multiplier
  double compressibility_multiplier_{1.0};    

};  // Papanastasiou-Roussel Bingham class
}  // namespace mpm

#include "bingham_papanastasiou_roussel.tcc"

#endif  // MPM_MATERIAL_BINGHAM_PAPANASTASIOUS_ROUSSEL_H_
