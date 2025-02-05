// Map temperature variables to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_heat_to_nodes_newmark() {
  this->map_heat_to_nodes();
  // get the specific_heat 
  const double specific_heat = 
        this->material(mpm::ParticlePhase::Solid)
            ->template property<double>(std::string("specific_heat"));
  // Map heat capacity and heat to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_heat_rate(true, mpm::ParticlePhase::Solid,
                  mass_ * specific_heat * shapefn_[i] * temperature_rate_);
    nodes_[i]->update_heat_ddot(true, mpm::ParticlePhase::Solid,
                  mass_ * specific_heat * shapefn_[i] * temperature_ddot_);
  }
}

// Map particle heat_rate to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_heat_rate_to_nodes() {
  // get the specific_heat 
  const double specific_heat = 
        this->material(mpm::ParticlePhase::Solid)
            ->template property<double>(std::string("specific_heat"));
  // Map heat capacity and heat to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_internal_heat(true, mpm::ParticlePhase::Solid,
                  -mass_ * specific_heat * shapefn_[i] * 
                  nodes_[i]->temperature_rate(mpm::ParticlePhase::Solid));
  }
}

// Map heat capacity to cell
template <unsigned Tdim>
bool mpm::Particle<Tdim>::map_heat_capacity_to_cell(double dt, 
                            double newmark_beta, double newmark_gamma) {
  bool status = true;
  try {
    // get the specific_heat of solid  
    double specific_heat_capacity = 
          this->material(mpm::ParticlePhase::Solid)
              ->template property<double>(std::string("specific_heat"));

    double heat_capacity = mass_density_ * specific_heat_capacity;
    cell_->compute_local_heat_capacity_matrix(shapefn_, 
                volume_, heat_capacity * newmark_gamma / newmark_beta / dt); 
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Map heat conductivity matrix to cell
template <unsigned Tdim>
bool mpm::Particle<Tdim>::map_heat_conductivity_to_cell() {
  bool status = true;
  try {
    // get the thermal conductivity coefficient of solid
    const double thermal_conductivity = 
          this->material(mpm::ParticlePhase::Solid)
              ->template property<double>(std::string("thermal_conductivity"));  

    cell_->compute_local_thermal_conductivity_matrix(dn_dx_, 
                                            volume_, thermal_conductivity); 

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Map thermal stiffness matrix to cell
template <unsigned Tdim>
bool mpm::Particle<Tdim>::map_thermal_expansivity_to_cell() {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // get the thermal expansivity coefficient of solid
    const double thermal_expansivity = 
          this->material(mpm::ParticlePhase::Solid)
              ->template property<double>(std::string("thermal_expansivity"));  

    const Eigen::Matrix<double, Tdim, Tdim> beta_matrix = -thermal_expansivity *
                    Eigen::Matrix<double, Tdim, Tdim>::Identity();

    // Reduce constitutive relations matrix depending on the dimension
    const Eigen::MatrixXd reduced_dmatrix =
        this->reduce_dmatrix(constitutive_matrix_);

    // Calculate B matrix
    const Eigen::MatrixXd bmatrix = this->compute_bmatrix();

    // Compute the identity vector
    Eigen::VectorXd identity_vector;
    identity_vector.resize(Tdim * (Tdim + 1) / 2);
    identity_vector.setZero();
    for (unsigned i = 0; i < Tdim; ++i) identity_vector[i] = 1;

    cell_->compute_local_thermal_expansivity_matrix(shapefn_, bmatrix, 
              reduced_dmatrix, identity_vector, volume_, thermal_expansivity); 
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute strain and volume of the particle using nodal displacement and
// nodal temeprature increment
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_strain_volume_newmark_thermal() noexcept {
  // Compute the volume at the previous time step
  this->volume_ /= (1. + dvolumetric_strain_);
  this->mass_density_ *= (1. + dvolumetric_strain_);

  // get the thermal expansivity coefficient of solid
  const double beta = 
        this->material(mpm::ParticlePhase::Solid)
            ->template property<double>(std::string("thermal_expansivity"));  

  // Compute deformation gradient increment from previous time step
  this->deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment_thermal(this->dn_dx_, beta,
                                                    mpm::ParticlePhase::Solid);

  // Compute strain increment from previous time step
  this->dstrain_ =
      this->compute_strain_increment_thermal(dn_dx_, beta,
                                              mpm::ParticlePhase::Solid);

  // Updated volumetric strain increment
  this->dvolumetric_strain_ = this->dstrain_.head(Tdim).sum();

  // Update volume using volumetric strain increment
  this->volume_ *= (1. + dvolumetric_strain_);
  this->mass_density_ /= (1. + dvolumetric_strain_);
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<1>::
                                    compute_strain_increment_thermal(
    const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept {
  // Define strain rincrement
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 1, 1> displacement = nodes_[i]->displacement(phase);
    double temperature_increment = nodes_[i]->temperature_increment(phase);
    // Mechanical strain
    strain_increment[0] += dn_dx(i, 0) * displacement[0];

    strain_increment[0] += -temperature_increment * beta * shapefn_[i];
  }

  if (std::fabs(strain_increment(0)) < 1.E-15) strain_increment[0] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<2>::
                                    compute_strain_increment_thermal(
    const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero(); 

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> displacement = nodes_[i]->displacement(phase);
    double temperature_increment = nodes_[i]->temperature_increment(phase);    
    // Mechanical strain
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
    strain_increment[1] += dn_dx(i, 1) * displacement[1];
    strain_increment[3] +=
        dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
    // Thermal strain
    strain_increment[0] += -temperature_increment * beta * shapefn_[i];
    strain_increment[1] += -temperature_increment * beta * shapefn_[i];
  }

  if (std::fabs(strain_increment[0]) < 1.E-15) strain_increment[0] = 0.;
  if (std::fabs(strain_increment[1]) < 1.E-15) strain_increment[1] = 0.;
  if (std::fabs(strain_increment[3]) < 1.E-15) strain_increment[3] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<3>::
                                    compute_strain_increment_thermal(
    const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> displacement = nodes_[i]->displacement(phase);
    double temperature_increment = nodes_[i]->temperature_increment(phase);   
    // Mechanical strain
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
    strain_increment[1] += dn_dx(i, 1) * displacement[1];
    strain_increment[2] += dn_dx(i, 2) * displacement[2];
    strain_increment[3] +=
        dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
    strain_increment[4] +=
        dn_dx(i, 2) * displacement[1] + dn_dx(i, 1) * displacement[2];
    strain_increment[5] +=
        dn_dx(i, 2) * displacement[0] + dn_dx(i, 0) * displacement[2];
    // Thermal strain
    strain_increment[0] += -temperature_increment * beta * shapefn_[i];
    strain_increment[1] += -temperature_increment * beta * shapefn_[i];
    strain_increment[2] += -temperature_increment * beta * shapefn_[i];
  }

  for (unsigned i = 0; i < strain_increment.size(); ++i)
    if (std::fabs(strain_increment[i]) < 1.E-15) strain_increment[i] = 0.;
  return strain_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::Particle<1>::compute_deformation_gradient_increment_thermal(
        const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    const auto& temperature_increment = nodes_[i]->temperature_increment(phase);    
    deformation_gradient_increment(0, 0) += dn_dx(i, 0) * displacement[0];
    deformation_gradient_increment(0, 0) += 
                              -temperature_increment * beta * shapefn_[i];
  }

  if (std::fabs(deformation_gradient_increment(0, 0) - 1.) < 1.E-15)
    deformation_gradient_increment(0, 0) = 1.;
  return deformation_gradient_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::Particle<2>::compute_deformation_gradient_increment_thermal(
        const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    const auto& temperature_increment= nodes_[i]->temperature_increment(phase);  
    deformation_gradient_increment.block(0, 0, 2, 2).noalias() +=
        displacement * dn_dx.row(i);
    for (unsigned j = 0; j < 2; ++j)    
      deformation_gradient_increment(j, j) += 
                              -temperature_increment * beta * shapefn_[i];
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      if (i != j && std::fabs(deformation_gradient_increment(i, j)) < 1.E-15)
        deformation_gradient_increment(i, j) = 0.;
      if (i == j &&
          std::fabs(deformation_gradient_increment(i, j) - 1.) < 1.E-15)
        deformation_gradient_increment(i, j) = 1.;
    }
  }
  return deformation_gradient_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::Particle<3>::compute_deformation_gradient_increment_thermal(
        const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    const auto& temperature_increment = nodes_[i]->temperature_increment(phase);
    deformation_gradient_increment.noalias() += displacement * dn_dx.row(i);
    for (unsigned j = 0; j < 3; ++j)
      deformation_gradient_increment(j, j) += 
                                -temperature_increment * beta * shapefn_[i];
  }

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      if (i != j && std::fabs(deformation_gradient_increment(i, j)) < 1.E-15)
        deformation_gradient_increment(i, j) = 0.;
      if (i == j &&
          std::fabs(deformation_gradient_increment(i, j) - 1.) < 1.E-15)
        deformation_gradient_increment(i, j) = 1.;
    }
  }
  return deformation_gradient_increment;
}

// Compute updated temperature of the particle
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_updated_temperature_newmark(
                                            double dt) noexcept {

  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  // Get interpolated nodal displacement and acceleration
  double nodal_temperature_increment = 0;
  double nodal_temperature_rate = 0;
  double nodal_temperature_ddot = 0; 
  double nodal_temperature = 0;  
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodal_temperature_increment +=
      shapefn_[i] * nodes_[i]->temperature_increment(mpm::ParticlePhase::Solid);
    nodal_temperature_rate +=
      shapefn_[i] * nodes_[i]->temperature_rate(mpm::ParticlePhase::Solid);
    nodal_temperature_ddot +=
      shapefn_[i] * nodes_[i]->temperature_ddot(mpm::ParticlePhase::Solid);
    nodal_temperature +=
      shapefn_[i] * nodes_[i]->temperature(mpm::ParticlePhase::Solid);
  }

  // Update particle velocity from interpolated nodal acceleration
  this->temperature_rate_ += 
          0.5 * dt * (nodal_temperature_ddot + this->temperature_ddot_);
  // this->temperature_rate_ = nodal_temperature_rate;
  this->temperature_ddot_ = nodal_temperature_ddot;

  // Update acceleration
  this->temperature_ += nodal_temperature_increment;
  this->temperature_pic_ = nodal_temperature;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
      // If temperature boundary, normal is zero is zero in the fixed direction
      const auto& temperature_constraints = 
                              nodes_[i]->temperature_constraints();

      if (temperature_constraints.size() != 0 ) {
        // // std::cout << "1 constraint id: " << id_ << "\n";
        for (const auto& pair : temperature_constraints) {
            this->temperature_ = pair.second;
            this->temperature_rate_ = 0;
            this->temperature_ddot_ = 0;
        }
        break;
      }
  }
}