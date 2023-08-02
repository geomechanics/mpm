//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleFiniteStrain<Tdim>::ParticleFiniteStrain(Index id,
                                                      const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  // Logger
  std::string logger = "particle_finite_strain" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleFiniteStrain<Tdim>::ParticleFiniteStrain(Index id,
                                                      const VectorDim& coord,
                                                      bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  //! Logger
  std::string logger = "particle_finite_strain" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Compute shape functions and gradients
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_shapefn() noexcept {
  mpm::Particle<Tdim>::compute_shapefn();
  this->reference_dn_dx_ = this->dn_dx_;
}

// Compute stress
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_stress(
    double dt, mpm::StressRate stress_rate) noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Calculate stress
  this->stress_ =
      (this->material())
          ->compute_stress(stress_, deformation_gradient_,
                           deformation_gradient_increment_, this,
                           &state_variables_[mpm::ParticlePhase::Solid]);

  // Update deformation gradient
  this->deformation_gradient_ =
      this->deformation_gradient_increment_ * this->deformation_gradient_;
}

// Compute strain of the particle
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_strain(double dt) noexcept {
  // Compute deformation gradient increment
  // Note: Deformation gradient must be updated after compute_stress
  deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(
          dn_dx_, mpm::ParticlePhase::Solid, dt);

  // Update volume and mass density
  const double deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ *= deltaJ;
  this->mass_density_ /= deltaJ;
}

//! Function to reinitialise material to be run at the beginning of each time
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::initialise_constitutive_law() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);

  // Reset material to be Elastic
  material_[mpm::ParticlePhase::Solid]->initialise(
      &state_variables_[mpm::ParticlePhase::Solid]);

  // Compute initial consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, deformation_gradient_,
          deformation_gradient_increment_, this,
          &state_variables_[mpm::ParticlePhase::Solid]);
}

//! Map mass and material stiffness matrix to cell
//! (used in poisson equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticleFiniteStrain<Tdim>::map_stiffness_matrix_to_cell(
    double newmark_beta, double dt, bool quasi_static) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute material stiffness matrix
    this->map_material_stiffness_matrix_to_cell();

    // Compute mass matrix
    if (!quasi_static) this->map_mass_matrix_to_cell(newmark_beta, dt);

    // Compute geometric stiffness matrix
    this->map_geometric_stiffness_matrix_to_cell();

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map geometric stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool
    mpm::ParticleFiniteStrain<Tdim>::map_geometric_stiffness_matrix_to_cell() {
  bool status = true;
  try {
    // Stress tensor in suitable dimension
    const Eigen::Matrix<double, Tdim, Tdim>& stress_matrix =
        mpm::math::matrix_form<Tdim>(this->stress_);

    const auto& reduced_stiffness = dn_dx_ * stress_matrix * dn_dx_.transpose();

    const Eigen::Matrix<double, Tdim, Tdim> Idim =
        Eigen::Matrix<double, Tdim, Tdim>::Identity();

    Eigen::MatrixXd geometric_stiffness(Idim.rows() * reduced_stiffness.rows(),
                                        Idim.cols() * reduced_stiffness.cols());
    for (int i = 0; i < reduced_stiffness.cols(); i++)
      for (int j = 0; j < reduced_stiffness.rows(); j++)
        geometric_stiffness.block(i * Idim.rows(), j * Idim.cols(), Idim.rows(),
                                  Idim.cols()) = reduced_stiffness(i, j) * Idim;

    // Compute local geometric stiffness matrix
    cell_->compute_local_geometric_stiffness_matrix(geometric_stiffness,
                                                    volume_);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute stress using implicit updating scheme
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_stress_newmark() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Clone state variables
  auto temp_state_variables = state_variables_[mpm::ParticlePhase::Solid];
  // Calculate stress
  this->stress_ = (this->material())
                      ->compute_stress(previous_stress_, deformation_gradient_,
                                       deformation_gradient_increment_, this,
                                       &temp_state_variables);

  // Compute current consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, deformation_gradient_,
          deformation_gradient_increment_, this, &temp_state_variables);
}

// Compute deformation gradient increment and volume of the particle
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_strain_volume_newmark() noexcept {
  // Compute volume and mass density at the previous time step
  double deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ /= deltaJ;
  this->mass_density_ *= deltaJ;

  // Compute deformation gradient increment from previous time step
  this->deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(this->reference_dn_dx_,
                                                   mpm::ParticlePhase::Solid);

  // Update dn_dx to the intermediate/iterative configuration
  const auto& def_grad_inc_inverse =
      this->deformation_gradient_increment_.inverse().block(0, 0, Tdim, Tdim);
  for (unsigned i = 0; i < dn_dx_.rows(); i++)
    dn_dx_.row(i) = reference_dn_dx_.row(i) * def_grad_inc_inverse;

  // Update volume and mass density
  deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ *= deltaJ;
  this->mass_density_ /= deltaJ;
}

// Compute Hencky strain
template <unsigned Tdim>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleFiniteStrain<Tdim>::compute_hencky_strain() const {

  // Left Cauchy-Green strain
  const Eigen::Matrix<double, 3, 3> left_cauchy_green_tensor =
      deformation_gradient_ * deformation_gradient_.transpose();

  // Principal values of left Cauchy-Green strain
  Eigen::Matrix<double, 3, 3> directors = Eigen::Matrix<double, 3, 3>::Zero();
  const Eigen::Matrix<double, 3, 1> principal_left_cauchy_green_strain =
      mpm::math::principal_tensor(left_cauchy_green_tensor, directors);

  // Principal value of Hencky (logarithmic) strain
  Eigen::Matrix<double, 3, 3> principal_hencky_strain =
      Eigen::Matrix<double, 3, 3>::Zero();
  principal_hencky_strain.diagonal() =
      0.5 * principal_left_cauchy_green_strain.array().log();

  // Hencky strain tensor and vector
  const Eigen::Matrix<double, 3, 3> hencky_strain =
      directors * principal_hencky_strain * directors.transpose();
  Eigen::Matrix<double, 6, 1> hencky_strain_vector;
  hencky_strain_vector(0) = hencky_strain(0, 0);
  hencky_strain_vector(1) = hencky_strain(1, 1);
  hencky_strain_vector(2) = hencky_strain(2, 2);
  hencky_strain_vector(3) = 2. * hencky_strain(0, 1);
  hencky_strain_vector(4) = 2. * hencky_strain(1, 2);
  hencky_strain_vector(5) = 2. * hencky_strain(2, 0);

  return hencky_strain_vector;
}

// Update stress and strain after convergence of Newton-Raphson iteration
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::update_stress_strain() noexcept {
  // Update converged stress
  this->stress_ =
      (this->material())
          ->compute_stress(this->previous_stress_, this->deformation_gradient_,
                           this->deformation_gradient_increment_, this,
                           &state_variables_[mpm::ParticlePhase::Solid]);

  // Update initial stress of the time step
  this->previous_stress_ = this->stress_;

  // Update deformation gradient
  this->deformation_gradient_ =
      this->deformation_gradient_increment_ * this->deformation_gradient_;

  // Volumetric strain increment
  this->dvolumetric_strain_ =
      (this->deformation_gradient_increment_.determinant() - 1.0);

  // Reset deformation gradient increment
  this->deformation_gradient_increment_.setIdentity();
}
