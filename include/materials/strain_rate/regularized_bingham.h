#ifndef MPM_MATERIAL_REGULARIZED_BINGHAM_H_
#define MPM_MATERIAL_REGULARIZED_BINGHAM_H_

#include <limits>

#include "Eigen/Dense"

#include "material.h"

namespace mpm {

//! Regularized Bingham class
//! \brief Regularized Bingham fluid material model
//! \details Regularized Bingham class stresses and strains
//! \tparam Tdim Dimension
template <unsigned Tdim>
class RegularizedBingham : public Material<Tdim> {
    public:
    //! Define a vector of 6 dof
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    //! Define a Matrix of 6 x 6
    using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

    //! Constructor with id and material properties
    //! \param[in] id Material ID
    //! \param[in] material_properties Material properties
    RegularizedBingham(unsigned id, const Json& material_properties);

    //! Destructor
    ~RegularizedBingham() override{};

    //! Delete copy constructor
    RegularizedBingham(const RegularizedBingham&) = delete;

    //! Delete assignement operator
    RegularizedBingham& operator=(const RegularizedBingham&) = delete;

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
    //! Equation of state parameter: lambda
    double lambda_{std::numeric_limits<double>::max()};
    //! Tau0 - shear yield stress in unit of [Pa]
    double tau0_{std::numeric_limits<double>::max()};
    //! mu - constant plastic viscosity [N s / m^2 or kg / m / s]
    double mu_{std::numeric_limits<double>::max()};
    //! Regularization shape factor m
    double m_{std::numeric_limits<double>::max()};
    //! Thixotropy parameter lambda0
    double lambda0{std::numeric_limits<double>::max()};
    //! Thixotropy parameter theta
    double theta{std::numeric_limits<double>::max()};
    //! Critical yielding shear rate
    double critical_shear_rate_{std::numeric_limits<double>::max()};
    //! Compressibility multiplier
    double compressibility_multiplier_{1.0};    

};  // Regularized Bingham class
}  // namespace mpm

#include "regularized_bingham.tcc"

#endif  // MPM_MATERIAL_REGULARIZED_BINGHAM_H_
