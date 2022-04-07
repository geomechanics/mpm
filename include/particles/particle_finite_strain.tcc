//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleFiniteStrain<Tdim>::ParticleFiniteStrain(Index id,
                                                      const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
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

//! Return particle data as POD
template <unsigned Tdim>
// cppcheck-suppress *
std::shared_ptr<void> mpm::ParticleFiniteStrain<Tdim>::pod() const {
  // Initialise particle_data
  auto particle_data = std::make_shared<mpm::PODParticleFiniteStrain>();

  Eigen::Vector3d coordinates;
  coordinates.setZero();
  for (unsigned j = 0; j < Tdim; ++j) coordinates[j] = this->coordinates_[j];

  Eigen::Vector3d displacement;
  displacement.setZero();
  for (unsigned j = 0; j < Tdim; ++j) displacement[j] = this->displacement_[j];

  Eigen::Vector3d velocity;
  velocity.setZero();
  for (unsigned j = 0; j < Tdim; ++j) velocity[j] = this->velocity_[j];

  Eigen::Vector3d acceleration;
  acceleration.setZero();
  for (unsigned j = 0; j < Tdim; ++j) acceleration[j] = this->acceleration_[j];

  // Particle local size
  Eigen::Vector3d nsize;
  nsize.setZero();
  Eigen::VectorXd size = this->natural_size();
  for (unsigned j = 0; j < Tdim; ++j) nsize[j] = size[j];

  Eigen::Matrix<double, 6, 1> stress = this->stress_;

  Eigen::Matrix<double, 6, 1> strain = this->strain_;

  particle_data->id = this->id();
  particle_data->mass = this->mass();
  particle_data->volume = this->volume();
  particle_data->pressure =
      (state_variables_[mpm::ParticlePhase::Solid].find("pressure") !=
       state_variables_[mpm::ParticlePhase::Solid].end())
          ? state_variables_[mpm::ParticlePhase::Solid].at("pressure")
          : 0.;

  particle_data->coord_x = coordinates[0];
  particle_data->coord_y = coordinates[1];
  particle_data->coord_z = coordinates[2];

  particle_data->displacement_x = displacement[0];
  particle_data->displacement_y = displacement[1];
  particle_data->displacement_z = displacement[2];

  particle_data->nsize_x = nsize[0];
  particle_data->nsize_y = nsize[1];
  particle_data->nsize_z = nsize[2];

  particle_data->velocity_x = velocity[0];
  particle_data->velocity_y = velocity[1];
  particle_data->velocity_z = velocity[2];

  particle_data->acceleration_x = acceleration[0];
  particle_data->acceleration_y = acceleration[1];
  particle_data->acceleration_z = acceleration[2];

  particle_data->stress_xx = stress[0];
  particle_data->stress_yy = stress[1];
  particle_data->stress_zz = stress[2];
  particle_data->tau_xy = stress[3];
  particle_data->tau_yz = stress[4];
  particle_data->tau_xz = stress[5];

  particle_data->strain_xx = strain[0];
  particle_data->strain_yy = strain[1];
  particle_data->strain_zz = strain[2];
  particle_data->gamma_xy = strain[3];
  particle_data->gamma_yz = strain[4];
  particle_data->gamma_xz = strain[5];

  particle_data->defgrad_00 = deformation_gradient_(0, 0);
  particle_data->defgrad_01 = deformation_gradient_(0, 1);
  particle_data->defgrad_02 = deformation_gradient_(0, 2);
  particle_data->defgrad_10 = deformation_gradient_(1, 0);
  particle_data->defgrad_11 = deformation_gradient_(1, 1);
  particle_data->defgrad_12 = deformation_gradient_(1, 2);
  particle_data->defgrad_20 = deformation_gradient_(2, 0);
  particle_data->defgrad_21 = deformation_gradient_(2, 1);
  particle_data->defgrad_22 = deformation_gradient_(2, 2);

  particle_data->epsilon_v = this->volumetric_strain_centroid_;

  particle_data->status = this->status();

  particle_data->cell_id = this->cell_id();

  particle_data->material_id = this->material_id();

  // Write state variables
  if (this->material() != nullptr) {
    particle_data->nstate_vars =
        state_variables_[mpm::ParticlePhase::Solid].size();
    if (state_variables_[mpm::ParticlePhase::Solid].size() > 20)
      throw std::runtime_error("# of state variables cannot be more than 20");
    unsigned i = 0;
    auto state_variables = (this->material())->state_variables();
    for (const auto& state_var : state_variables) {
      particle_data->svars[i] =
          state_variables_[mpm::ParticlePhase::Solid].at(state_var);
      ++i;
    }
  }

  return particle_data;
}

//! Initialise particle data from POD
template <unsigned Tdim>
bool mpm::ParticleFiniteStrain<Tdim>::initialise_particle(
    PODParticle& particle) {
  // Initialise solid phase
  bool status = mpm::Particle<Tdim>::initialise_particle(particle);
  auto particle_finite_strain =
      reinterpret_cast<PODParticleFiniteStrain*>(&particle);

  // Deformation gradient
  this->deformation_gradient_(0, 0) = particle_finite_strain->defgrad_00;
  this->deformation_gradient_(0, 1) = particle_finite_strain->defgrad_01;
  this->deformation_gradient_(0, 2) = particle_finite_strain->defgrad_02;
  this->deformation_gradient_(1, 0) = particle_finite_strain->defgrad_10;
  this->deformation_gradient_(1, 1) = particle_finite_strain->defgrad_11;
  this->deformation_gradient_(1, 2) = particle_finite_strain->defgrad_12;
  this->deformation_gradient_(2, 0) = particle_finite_strain->defgrad_20;
  this->deformation_gradient_(2, 1) = particle_finite_strain->defgrad_21;
  this->deformation_gradient_(2, 2) = particle_finite_strain->defgrad_22;

  this->deformation_gradient_increment_.setIdentity();

  return status;
}

//! Initialise particle data from POD
template <unsigned Tdim>
bool mpm::ParticleFiniteStrain<Tdim>::initialise_particle(
    PODParticle& particle,
    const std::vector<std::shared_ptr<mpm::Material<Tdim>>>& materials) {
  auto particle_finite_strain =
      reinterpret_cast<PODParticleFiniteStrain*>(&particle);
  bool status = this->initialise_particle(*particle_finite_strain);

  assert(materials.size() == 1);

  if (materials.at(mpm::ParticlePhase::Solid) != nullptr) {
    if (this->material_id() == materials.at(mpm::ParticlePhase::Solid)->id() ||
        this->material_id() == std::numeric_limits<unsigned>::max()) {
      bool assign_mat =
          this->assign_material(materials.at(mpm::ParticlePhase::Solid));
      if (!assign_mat) throw std::runtime_error("Material assignment failed");
      // Reinitialize state variables
      auto mat_state_vars = (this->material())->initialise_state_variables();
      if (mat_state_vars.size() == particle.nstate_vars) {
        unsigned i = 0;
        auto state_variables = (this->material())->state_variables();
        for (const auto& state_var : state_variables) {
          this->state_variables_[mpm::ParticlePhase::Solid].at(state_var) =
              particle.svars[i];
          ++i;
        }
      }
    } else {
      status = false;
      throw std::runtime_error("Material is invalid to assign to particle!");
    }
  }

  return status;
}

// Initialise particle properties
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::initialise() {
  mpm::Particle<Tdim>::initialise();
  deformation_gradient_increment_.setIdentity();
}

// Compute stress
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_stress() noexcept {
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
    // Calculate G matrix
    const Eigen::MatrixXd gmatrix = this->compute_gmatrix();

    // Stress component matrix
    Eigen::MatrixXd stress_matrix = compute_stress_matrix();

    // Compute local geometric stiffness matrix
    cell_->compute_local_material_stiffness_matrix(gmatrix, stress_matrix,
                                                   volume_);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute G matrix for geometric stiffness
template <>
inline Eigen::MatrixXd
    mpm::ParticleFiniteStrain<1>::compute_gmatrix() noexcept {
  Eigen::MatrixXd gmatrix;
  gmatrix.resize(1, this->nodes_.size());
  gmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    gmatrix(0, i) = dn_dx_(i, 0);
  }
  return gmatrix;
}

// Compute G matrix for geometric stiffness
template <>
inline Eigen::MatrixXd
    mpm::ParticleFiniteStrain<2>::compute_gmatrix() noexcept {
  Eigen::MatrixXd gmatrix;
  gmatrix.resize(4, 2 * this->nodes_.size());
  gmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    gmatrix(0, 2 * i) = dn_dx_(i, 0);
    gmatrix(1, 2 * i) = dn_dx_(i, 1);
    gmatrix(2, 2 * i + 1) = dn_dx_(i, 0);
    gmatrix(3, 2 * i + 1) = dn_dx_(i, 1);
  }
  return gmatrix;
}

// Compute G matrix for geometric stiffness
template <>
inline Eigen::MatrixXd
    mpm::ParticleFiniteStrain<3>::compute_gmatrix() noexcept {
  Eigen::MatrixXd gmatrix;
  gmatrix.resize(9, 3 * this->nodes_.size());
  gmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    gmatrix(0, 3 * i) = dn_dx_(i, 0);
    gmatrix(1, 3 * i) = dn_dx_(i, 1);
    gmatrix(2, 3 * i) = dn_dx_(i, 2);
    gmatrix(3, 3 * i + 1) = dn_dx_(i, 0);
    gmatrix(4, 3 * i + 1) = dn_dx_(i, 1);
    gmatrix(5, 3 * i + 1) = dn_dx_(i, 2);
    gmatrix(6, 3 * i + 2) = dn_dx_(i, 0);
    gmatrix(7, 3 * i + 2) = dn_dx_(i, 1);
    gmatrix(8, 3 * i + 2) = dn_dx_(i, 2);
  }
  return gmatrix;
}

// Compute stress component matrix for geometric stiffness
template <>
inline Eigen::MatrixXd
    mpm::ParticleFiniteStrain<1>::compute_stress_matrix() noexcept {
  Eigen::MatrixXd stress_matrix;
  stress_matrix.resize(1, 1);
  stress_matrix.setZero();

  stress_matrix(0, 0) = this->stress_(0);

  return stress_matrix;
}

// Compute stress component matrix for geometric stiffness
template <>
inline Eigen::MatrixXd
    mpm::ParticleFiniteStrain<2>::compute_stress_matrix() noexcept {
  Eigen::MatrixXd stress_matrix;
  stress_matrix.resize(4, 4);
  stress_matrix.setZero();

  for (unsigned i = 0; i <= 1; i++) {
    stress_matrix(2 * i + 0, 2 * i + 0) = this->stress_(0);
    stress_matrix(2 * i + 0, 2 * i + 1) = this->stress_(3);
    stress_matrix(2 * i + 1, 2 * i + 0) = this->stress_(3);
    stress_matrix(2 * i + 1, 2 * i + 1) = this->stress_(1);
  }

  return stress_matrix;
}

// Compute stress component matrix for geometric stiffness
template <>
inline Eigen::MatrixXd
    mpm::ParticleFiniteStrain<3>::compute_stress_matrix() noexcept {
  Eigen::MatrixXd stress_matrix;
  stress_matrix.resize(9, 9);
  stress_matrix.setZero();

  for (unsigned i = 0; i <= 2; i++) {
    stress_matrix(3 * i + 0, 3 * i + 0) = this->stress_(0);
    stress_matrix(3 * i + 0, 3 * i + 1) = this->stress_(3);
    stress_matrix(3 * i + 0, 3 * i + 2) = this->stress_(5);
    stress_matrix(3 * i + 1, 3 * i + 0) = this->stress_(3);
    stress_matrix(3 * i + 1, 3 * i + 1) = this->stress_(1);
    stress_matrix(3 * i + 1, 3 * i + 2) = this->stress_(4);
    stress_matrix(3 * i + 2, 3 * i + 0) = this->stress_(5);
    stress_matrix(3 * i + 2, 3 * i + 1) = this->stress_(4);
    stress_matrix(3 * i + 2, 3 * i + 2) = this->stress_(2);
  }

  return stress_matrix;
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
      this->compute_deformation_gradient_increment(this->dn_dx_,
                                                   mpm::ParticlePhase::Solid);

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
      mpm::materials::principal_tensor(left_cauchy_green_tensor, directors);

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

  // Update volumetric strain at particle position (not at centroid)
  this->volumetric_strain_centroid_ += this->dvolumetric_strain_;

  // Reset deformation gradient increment
  this->deformation_gradient_increment_.setIdentity();
}

//! Compute size of serialized particle data
template <unsigned Tdim>
int mpm::ParticleFiniteStrain<Tdim>::compute_pack_size() const {
  int total_size = mpm::Particle<Tdim>::compute_pack_size();
  int partial_size;
#ifdef USE_MPI
  // deformation gradient
  // Note: only need 3 extra size considering strain is not transferred
  MPI_Pack_size(3, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;
#endif
  return total_size;
}

//! Serialize particle data
template <unsigned Tdim>
std::vector<uint8_t> mpm::ParticleFiniteStrain<Tdim>::serialize() {
  // Compute pack size
  if (pack_size_ == 0) pack_size_ = this->compute_pack_size();
  // Initialize data buffer
  std::vector<uint8_t> data;
  data.resize(pack_size_);
  uint8_t* data_ptr = &data[0];
  int position = 0;

#ifdef USE_MPI
  // Type
  int type = ParticleType.at(this->type());
  MPI_Pack(&type, 1, MPI_INT, data_ptr, data.size(), &position, MPI_COMM_WORLD);

  // Material id
  unsigned nmaterials = material_id_.size();
  MPI_Pack(&nmaterials, 1, MPI_UNSIGNED, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  MPI_Pack(&material_id_[mpm::ParticlePhase::Solid], 1, MPI_UNSIGNED, data_ptr,
           data.size(), &position, MPI_COMM_WORLD);

  // ID
  MPI_Pack(&id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Mass
  MPI_Pack(&mass_, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Volume
  MPI_Pack(&volume_, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Pressure
  double pressure =
      (state_variables_[mpm::ParticlePhase::Solid].find("pressure") !=
       state_variables_[mpm::ParticlePhase::Solid].end())
          ? state_variables_[mpm::ParticlePhase::Solid].at("pressure")
          : 0.;
  MPI_Pack(&pressure, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // Coordinates
  MPI_Pack(coordinates_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);
  // Displacement
  MPI_Pack(displacement_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);
  // Natural size
  MPI_Pack(natural_size_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);
  // Velocity
  MPI_Pack(velocity_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Acceleration
  MPI_Pack(acceleration_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);
  // Stress
  MPI_Pack(stress_.data(), 6, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Deformation Gradient
  MPI_Pack(deformation_gradient_.data(), 9, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // epsv
  MPI_Pack(&volumetric_strain_centroid_, 1, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Cell id
  MPI_Pack(&cell_id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Status
  MPI_Pack(&status_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // nstate variables
  unsigned nstate_vars = state_variables_[mpm::ParticlePhase::Solid].size();
  MPI_Pack(&nstate_vars, 1, MPI_UNSIGNED, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // state variables
  if (this->material(mpm::ParticlePhase::Solid) != nullptr) {
    std::vector<double> svars;
    auto state_variables =
        (this->material(mpm::ParticlePhase::Solid))->state_variables();
    for (const auto& state_var : state_variables)
      svars.emplace_back(
          state_variables_[mpm::ParticlePhase::Solid].at(state_var));

    // Write state vars
    MPI_Pack(&svars[0], nstate_vars, MPI_DOUBLE, data_ptr, data.size(),
             &position, MPI_COMM_WORLD);
  }
#endif
  return data;
}

//! Deserialize particle data
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::deserialize(
    const std::vector<uint8_t>& data,
    std::vector<std::shared_ptr<mpm::Material<Tdim>>>& materials) {
  uint8_t* data_ptr = const_cast<uint8_t*>(&data[0]);
  int position = 0;

#ifdef USE_MPI
  // Type
  int type;
  MPI_Unpack(data_ptr, data.size(), &position, &type, 1, MPI_INT,
             MPI_COMM_WORLD);
  assert(type == ParticleType.at(this->type()));
  // material id
  int nmaterials = 0;
  MPI_Unpack(data_ptr, data.size(), &position, &nmaterials, 1, MPI_UNSIGNED,
             MPI_COMM_WORLD);
  MPI_Unpack(data_ptr, data.size(), &position,
             &material_id_[mpm::ParticlePhase::Solid], 1, MPI_UNSIGNED,
             MPI_COMM_WORLD);

  assert(nmaterials == materials.size());
  // ID
  MPI_Unpack(data_ptr, data.size(), &position, &id_, 1, MPI_UNSIGNED_LONG_LONG,
             MPI_COMM_WORLD);
  // mass
  MPI_Unpack(data_ptr, data.size(), &position, &mass_, 1, MPI_DOUBLE,
             MPI_COMM_WORLD);
  // volume
  MPI_Unpack(data_ptr, data.size(), &position, &volume_, 1, MPI_DOUBLE,
             MPI_COMM_WORLD);
  // mass density
  this->mass_density_ = mass_ / volume_;

  // pressure
  double pressure;
  MPI_Unpack(data_ptr, data.size(), &position, &pressure, 1, MPI_DOUBLE,
             MPI_COMM_WORLD);

  // Coordinates
  MPI_Unpack(data_ptr, data.size(), &position, coordinates_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);
  // Displacement
  MPI_Unpack(data_ptr, data.size(), &position, displacement_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);
  // Natural size
  MPI_Unpack(data_ptr, data.size(), &position, natural_size_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);
  // Velocity
  MPI_Unpack(data_ptr, data.size(), &position, velocity_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);
  // Acceleration
  MPI_Unpack(data_ptr, data.size(), &position, acceleration_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);
  // Stress
  MPI_Unpack(data_ptr, data.size(), &position, stress_.data(), 6, MPI_DOUBLE,
             MPI_COMM_WORLD);
  this->previous_stress_ = stress_;
  // Deformation gradient
  MPI_Unpack(data_ptr, data.size(), &position, deformation_gradient_.data(), 9,
             MPI_DOUBLE, MPI_COMM_WORLD);
  deformation_gradient_increment_.setIdentity();

  // epsv
  MPI_Unpack(data_ptr, data.size(), &position, &volumetric_strain_centroid_, 1,
             MPI_DOUBLE, MPI_COMM_WORLD);
  // cell id
  MPI_Unpack(data_ptr, data.size(), &position, &cell_id_, 1,
             MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
  // status
  MPI_Unpack(data_ptr, data.size(), &position, &status_, 1, MPI_C_BOOL,
             MPI_COMM_WORLD);

  // Assign materials
  assert(material_id_[mpm::ParticlePhase::Solid] ==
         materials.at(mpm::ParticlePhase::Solid)->id());
  bool assign_mat =
      this->assign_material(materials.at(mpm::ParticlePhase::Solid));
  if (!assign_mat)
    throw std::runtime_error(
        "deserialize particle(): Material assignment failed");

  // nstate vars
  unsigned nstate_vars;
  MPI_Unpack(data_ptr, data.size(), &position, &nstate_vars, 1, MPI_UNSIGNED,
             MPI_COMM_WORLD);

  if (nstate_vars > 0) {
    std::vector<double> svars;
    svars.reserve(nstate_vars);
    MPI_Unpack(data_ptr, data.size(), &position, &svars[0], nstate_vars,
               MPI_DOUBLE, MPI_COMM_WORLD);

    // Reinitialize state variables
    auto mat_state_vars = (this->material())->initialise_state_variables();
    if (mat_state_vars.size() != nstate_vars)
      throw std::runtime_error(
          "Deserialize particle(): state_vars size mismatch");
    unsigned i = 0;
    auto state_variables = (this->material())->state_variables();
    for (const auto& state_var : state_variables) {
      this->state_variables_[mpm::ParticlePhase::Solid].at(state_var) =
          svars[i];
      ++i;
    }
  }
#endif
}