//! Construct a two phase particle with id and coordinates
template <unsigned Tdim>
mpm::FluidParticle<Tdim>::FluidParticle(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {

  // Logger
  std::string logger =
      "FluidParticle" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Compute stress
template <unsigned Tdim>
void mpm::FluidParticle<Tdim>::compute_stress(
    double dt, mpm::StressRate stress_rate) noexcept {
  // Run particle compute stress
  mpm::Particle<Tdim>::compute_stress(dt);

  // Calculate fluid turbulent stress
  this->stress_.noalias() += this->compute_turbulent_stress();
}

// Compute turbulent stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1>
    mpm::FluidParticle<Tdim>::compute_turbulent_stress() {
  // Compute turbulent stress depends on the model
  Eigen::Matrix<double, 6, 1> tstress;
  tstress.setZero();

  // Apply LES Smagorinsky closure
  const double smagorinsky_constant = 0.2;
  const double grid_spacing = std::pow(cell_->volume(), 1 / (double)Tdim);
  const auto strain_rate = this->strain_rate();
  double local_strain_rate = 0.0;
  local_strain_rate +=
      2 * (strain_rate[0] * strain_rate[0] + strain_rate[1] * strain_rate[1] +
           strain_rate[2] * strain_rate[2]) +
      strain_rate[3] * strain_rate[3] + strain_rate[4] * strain_rate[4] +
      strain_rate[5] * strain_rate[5];
  local_strain_rate = std::sqrt(local_strain_rate);

  // Turbulent Eddy Viscosity
  const double turbulent_viscosity =
      std::pow(smagorinsky_constant * grid_spacing, 2) * local_strain_rate;

  // Turbulent stress
  tstress(0) = 2. * turbulent_viscosity / mass_density_ * strain_rate(0);
  tstress(1) = 2. * turbulent_viscosity / mass_density_ * strain_rate(1);
  tstress(2) = 2. * turbulent_viscosity / mass_density_ * strain_rate(2);
  tstress(3) = turbulent_viscosity / mass_density_ * strain_rate(3);
  tstress(4) = turbulent_viscosity / mass_density_ * strain_rate(4);
  tstress(5) = turbulent_viscosity / mass_density_ * strain_rate(5);

  return tstress;
}

//! Map internal force
template <>
inline void mpm::FluidParticle<1>::map_internal_force() noexcept {
  // initialise a vector of total stress (deviatoric + turbulent - pressure)
  Eigen::Matrix<double, 6, 1> total_stress = this->stress_;
  total_stress(0) -=
      this->projection_param_ *
      this->state_variables(mpm::ParticlePhase::SinglePhase)["pressure"];
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 1, 1> force;
    force[0] = dn_dx_(i, 0) * total_stress[0];
    force *= -1 * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::SinglePhase,
                                     force);
  }
}

//! Map internal force
template <>
inline void mpm::FluidParticle<2>::map_internal_force() noexcept {
  // initialise a vector of total stress (deviatoric + turbulent - pressure)
  Eigen::Matrix<double, 6, 1> total_stress = this->stress_;
  total_stress(0) -=
      this->projection_param_ *
      this->state_variables(mpm::ParticlePhase::SinglePhase)["pressure"];
  total_stress(1) -=
      this->projection_param_ *
      this->state_variables(mpm::ParticlePhase::SinglePhase)["pressure"];

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 2, 1> force;
    force[0] = dn_dx_(i, 0) * total_stress[0] + dn_dx_(i, 1) * total_stress[3];
    force[1] = dn_dx_(i, 1) * total_stress[1] + dn_dx_(i, 0) * total_stress[3];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::SinglePhase,
                                     force);
  }
}

//! Map internal force
template <>
inline void mpm::FluidParticle<3>::map_internal_force() noexcept {
  // initialise a vector of total stress (deviatoric + turbulent - pressure)
  Eigen::Matrix<double, 6, 1> total_stress = this->stress_;
  total_stress(0) -=
      this->projection_param_ *
      this->state_variables(mpm::ParticlePhase::SinglePhase)["pressure"];
  total_stress(1) -=
      this->projection_param_ *
      this->state_variables(mpm::ParticlePhase::SinglePhase)["pressure"];
  total_stress(2) -=
      this->projection_param_ *
      this->state_variables(mpm::ParticlePhase::SinglePhase)["pressure"];

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 3, 1> force;
    force[0] = dn_dx_(i, 0) * total_stress[0] + dn_dx_(i, 1) * total_stress[3] +
               dn_dx_(i, 2) * total_stress[5];

    force[1] = dn_dx_(i, 1) * total_stress[1] + dn_dx_(i, 0) * total_stress[3] +
               dn_dx_(i, 2) * total_stress[4];

    force[2] = dn_dx_(i, 2) * total_stress[2] + dn_dx_(i, 1) * total_stress[4] +
               dn_dx_(i, 0) * total_stress[5];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::SinglePhase,
                                     force);
  }
}

//! Map laplacian element matrix to cell (used in poisson equation LHS)
template <unsigned Tdim>
bool mpm::FluidParticle<Tdim>::map_laplacian_to_cell() {
  bool status = true;
  try {
    // Compute local matrix of Laplacian
    cell_->compute_local_laplacian(dn_dx_, volume_);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map poisson rhs element matrix to cell (used in poisson equation RHS)
template <unsigned Tdim>
bool mpm::FluidParticle<Tdim>::map_poisson_right_to_cell() {
  bool status = true;
  try {
    // Compute local poisson rhs matrix
    cell_->compute_local_poisson_right(
        shapefn_, dn_dx_, volume_,
        this->material(mpm::ParticlePhase::SinglePhase)
            ->template property<double>(std::string("density")));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute updated pressure of the particle based on nodal pressure
template <unsigned Tdim>
bool mpm::FluidParticle<Tdim>::compute_updated_pressure() {
  bool status = true;
  try {
    double pressure_increment = 0;
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      pressure_increment += shapefn_(i) * nodes_[i]->pressure_increment();
    }

    // Get interpolated nodal pressure
    state_variables_[mpm::ParticlePhase::SinglePhase].at("pressure") =
        state_variables_[mpm::ParticlePhase::SinglePhase].at("pressure") *
            projection_param_ +
        pressure_increment;

    // Overwrite pressure if free surface
    if (this->free_surface())
      state_variables_[mpm::ParticlePhase::SinglePhase].at("pressure") = 0.0;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map correction matrix element matrix to cell (used to correct velocity)
template <unsigned Tdim>
bool mpm::FluidParticle<Tdim>::map_correction_matrix_to_cell() {
  bool status = true;
  try {
    cell_->compute_local_correction_matrix(shapefn_, dn_dx_, volume_);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

//! Compute size of serialized particle data
template <unsigned Tdim>
int mpm::FluidParticle<Tdim>::compute_pack_size() const {
  int total_size = mpm::Particle<Tdim>::compute_pack_size();
  int partial_size;
#ifdef USE_MPI
  // projection parameter - beta
  MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;
#endif
  return total_size;
}

//! Serialize particle data
template <unsigned Tdim>
std::vector<uint8_t> mpm::FluidParticle<Tdim>::serialize() {
  // Compute pack size
  if (pack_size_ == 0) pack_size_ = compute_pack_size();
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
  // Strain
  MPI_Pack(strain_.data(), 6, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Deformation Gradient
  MPI_Pack(deformation_gradient_.data(), 9, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Mapping matrix
  bool initialise_mapping = (this->mapping_matrix_.size() != 0);
  MPI_Pack(&initialise_mapping, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  if (initialise_mapping) {
    MPI_Pack(mapping_matrix_.data(), Tdim * Tdim, MPI_DOUBLE, data_ptr,
             data.size(), &position, MPI_COMM_WORLD);
  }

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

  // Projection parameter
  MPI_Pack(&projection_param_, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
#endif
  return data;
}

//! Deserialize particle data
template <unsigned Tdim>
void mpm::FluidParticle<Tdim>::deserialize(
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
  // Strain
  MPI_Unpack(data_ptr, data.size(), &position, strain_.data(), 6, MPI_DOUBLE,
             MPI_COMM_WORLD);
  // Deformation gradient
  MPI_Unpack(data_ptr, data.size(), &position, deformation_gradient_.data(), 9,
             MPI_DOUBLE, MPI_COMM_WORLD);

  // Mapping matrix
  bool initialise_mapping = false;
  MPI_Unpack(data_ptr, data.size(), &position, &initialise_mapping, 1,
             MPI_C_BOOL, MPI_COMM_WORLD);
  if (initialise_mapping) {
    if (mapping_matrix_.cols() != Tdim) mapping_matrix_.resize(Tdim, Tdim);
    MPI_Unpack(data_ptr, data.size(), &position, mapping_matrix_.data(),
               Tdim * Tdim, MPI_DOUBLE, MPI_COMM_WORLD);
  }

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

  // projection parameter
  MPI_Unpack(data_ptr, data.size(), &position, &projection_param_, 1,
             MPI_DOUBLE, MPI_COMM_WORLD);

#endif
}