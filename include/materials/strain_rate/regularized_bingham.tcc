//! Constructor with material properties
template <unsigned Tdim>
mpm::RegularizedBingham<Tdim>::RegularizedBingham(unsigned id, 
    const Json& material_properties)
    : Material<Tdim>(id, material_properties) {
    try {
        // Fluid density
        density_ = material_properties.at("density").template get<double>();
        // Equation of state parameter: lambda
        lambda_ = material_properties.at("lambda").template get<double>();
        // Bulk modulus
        bulk_modulus_ =
            material_properties.at("bulk_modulus").template get<double>();
        // Yield stress
        tau0_ = material_properties.at("tau0").template get<double>();
        // Viscosity
        mu_ = material_properties.at("mu").template get<double>();
        // Regularization shape factor m
        m_ = material_properties.at("m").template get<double>();
        // Thixotropy parameter lambda0
        lambda0 = material_properties.at("lambda0").template get<double>();
        // Thixotropy parameter theta
        theta = material_properties.at("theta").template get<double>();
        critical_shear_rate_ =
            material_properties["critical_shear_rate"].template get<double>();

        // Special material properties
        if (material_properties.contains("incompressible")) {
        bool incompressible =
            material_properties.at("incompressible").template get<bool>();
        if (incompressible) compressibility_multiplier_ = 0.0;
        }
        properties_ = material_properties;
    } catch (Json::exception& except) {
        console_->error("Material parameter not set: {} {}\n", except.what(),
                        except.id);
    }
    }

//! Initialise history variables
template <unsigned Tdim>
mpm::dense_map mpm::RegularizedBingham<Tdim>::initialise_state_variables() {
    mpm::dense_map state_vars = { {"pressure", 0.0},
                                    {"current_time", 0.0}};

    return state_vars;
}

//! State variables
template <unsigned Tdim>
std::vector<std::string> mpm::RegularizedBingham<Tdim>::state_variables() const {
    const std::vector<std::string> state_vars = {"pressure", "current_time"};
    return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::RegularizedBingham<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

    // Get strain rate
    auto strain_rate = ptr->strain_rate();
    // Convert strain rate to rate of deformation tensor
    strain_rate.tail(3) *= 0.5;

    // Set threshold for minimum critical shear rate
    const double shear_rate_threshold = 1.0E-15;
    if (critical_shear_rate_ < shear_rate_threshold)
        critical_shear_rate_ = shear_rate_threshold;

    // Rate of shear = sqrt(2 * D_ij * D_ij)
    // Since D (D_ij) is in Voigt notation (D_i), and the definition above is in
    // matrix, the last 3 components have to be doubled D_ij * D_ij = D_0^2 +
    // D_1^2 + D_2^2 + 2*D_3^2 + 2*D_4^2 + 2*D_5^2 Yielding is defined: rate of
    // shear > critical_shear_rate_^2 Checking yielding from strain rate vs
    // critical yielding shear rate
    double shear_rate =
        std::sqrt(2. * (strain_rate.dot(strain_rate) +
                        strain_rate.tail(3).dot(strain_rate.tail(3))));

    // Compute apparent viscosity
    double apparent_viscosity = 0.;
    if (shear_rate * shear_rate > critical_shear_rate_ * critical_shear_rate_)
        apparent_viscosity = mu_ + (tau0_ / shear_rate) 
                        * (1. - std::exp(-m_ * shear_rate));

    // Compute shear change to volumetric
    // tau deviatoric part of cauchy stress tensor
    Eigen::Matrix<double, 6, 1> tau = 2 * apparent_viscosity * strain_rate;

    //   // von Mises criterion
    //   // trace of second invariant J2 of deviatoric stress in matrix form
    //   // Since tau is in Voigt notation, only the first three numbers matter
    //   // yield condition trace of the invariant > tau0^2
      const double trace_invariant2 = 0.5 * (tau.head(3)).dot(tau.head(3));
      if (trace_invariant2 < (tau0_ * tau0_)) tau.setZero();

    // Update the bulk modulus
    double bulk_modulus = bulk_modulus_ + lambda_ * (*state_vars).at("pressure"); 
    // Update pressure
    (*state_vars).at("pressure") += compressibility_multiplier_ * (
                                -bulk_modulus * ptr->dvolumetric_strain());

    // Update volumetric and deviatoric stress
    // thermodynamic pressure is from material point
    // stress = -thermodynamic_pressure I + tau, where I is identity matrix or
    // direc_delta in Voigt notation
    const Eigen::Matrix<double, 6, 1> updated_stress =
        -(*state_vars).at("pressure") * this->dirac_delta() *
            compressibility_multiplier_ +
        tau;

    return updated_stress;
    }

//! Dirac delta 2D
template <>
inline Eigen::Matrix<double, 6, 1> mpm::RegularizedBingham<2>::dirac_delta() const {

return (Eigen::Matrix<double, 6, 1>() << 1.f, 1.f, 0.f, 0.f, 0.f, 0.f)
    .finished();
}

//! Dirac delta 3D
template <>
inline Eigen::Matrix<double, 6, 1> mpm::RegularizedBingham<3>::dirac_delta() const {

return (Eigen::Matrix<double, 6, 1>() << 1.f, 1.f, 1.f, 0.f, 0.f, 0.f)
    .finished();
}
