// Initialise particle thermal properties
template <unsigned Tdim>
void mpm::Particle<Tdim>::initialise_thermal() {
  // Scalar properties
  temperature_ = 0.;
  temperature_pic_ = 0.;
  temperature_flip_ = 0.;
  temperature_rate_ = 0.;
  temperature_ddot_ = 0.;
  temperature_increment_ = 0.;
  dthermal_volumetric_strain_ = 0.;
  heat_source_ = 0;

  // Vector properties
  temperature_gradient_.setZero();
  mass_gradient_.setZero(); 
  outward_normal_.setZero();  

  // Tensor properties 
  dthermal_strain_.setZero();  
  thermal_strain_.setZero();
  heat_flux_.setZero();  

  // Bool properties
  set_heat_source_ = false;

  // Initialize scalar, vector, and tensor data properties
  this->scalar_properties_.insert({   
      {"temperatures",           [&]() {return this->temperature_;}},
      {"PIC_temperatures",       [&]() {return temperature_pic_;}}   
  });

  this->vector_properties_.insert({  
      {"temperature_gradients",  [&]() {return this->temperature_gradient_;}},
      {"outward_normals",        [&]() {return this->outward_normal_;}}, 
      {"mass_gradients",         [&]() {return this->mass_gradient_;}},
      {"heat_fluxes",            [&]() {return this->heat_flux_;}}      
  });

  this->tensor_properties_.insert({  
      {"thermal_strains",        [&]() {return this->thermal_strain_;}} 
  });
}

// Map particle heat capacity and heat to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_heat_to_nodes() {
  // get the specific_heat 
  const double specific_heat = 
        this->material(mpm::ParticlePhase::Solid)
            ->template property<double>(std::string("specific_heat"));
  // Map heat capacity and heat to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_heat_capacity(true, mpm::ParticlePhase::Solid,
                                mass_ * specific_heat * shapefn_[i]);
    nodes_[i]->update_heat(true, mpm::ParticlePhase::Solid,
                  mass_ * specific_heat * shapefn_[i] * temperature_);               
  }
}

// Map heat conduction to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_heat_conduction() {
  // Assign the thermal conductivity
  const double k_conductivity = 
          this->material(mpm::ParticlePhase::Solid)
              ->template property<double>(std::string("thermal_conductivity"));
  
  // Assign temperature gradient
  this->compute_temperature_gradient(mpm::ParticlePhase::Solid);

  // Compute nodal heat conduction
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    double heat_conduction = 0;
    for (unsigned j = 0; j < Tdim; ++j){
      heat_conduction += dn_dx_(i, j) * this->temperature_gradient_[j]; 
    }

    heat_conduction *= -1 * this->volume_ * k_conductivity;
    nodes_[i]->update_internal_heat(true, mpm::ParticlePhase::Solid, 
                                      heat_conduction);
  }
}

// Map plastic heat generation to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_plastic_heat_dissipation(double dt) {
  // Assign the plastic heat transfer coefficient
  const double theta_p = this->material(mpm::ParticlePhase::Solid)
        ->template property<double>(std::string("plastic_heat_transfer_coeff"));

  // Assign plastic work from material
  if (state_variables_[mpm::ParticlePhase::Solid].find("dpwork") !=
      state_variables_[mpm::ParticlePhase::Solid].end()) {
    // Get plastic work
    double plastic_heat = 
                state_variables_[mpm::ParticlePhase::Solid]["dpwork"];

    plastic_heat *= -1 * theta_p * this->volume_ / dt;
    // Map plastic heat dissipation to nodes
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      nodes_[i]->update_external_heat(true, mpm::ParticlePhase::Solid, 
                                      plastic_heat);
    }
  }
}

// Map virtual heat flux to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_virtual_heat_flux(bool convective, 
                                          const double vfm_param1,
                                          const double vfm_param2) {
  this->compute_mass_gradient(mpm::ParticlePhase::Solid);
  outward_normal_.setZero();
  this->free_surface_ = false;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i]->free_surface()) {
      outward_normal_ = -mass_gradient_/mass_gradient_.norm();
      this->free_surface_ = true;
      break;
    }
  }

  if (this->free_surface_) {
    std::vector<unsigned> node_indices;
    for (unsigned i = 0; i < nodes_.size(); ++i) {
        // If fixed boundary, normal is zero in the fixed direction
        const auto& velocity_constraints = nodes_[i]->velocity_constraints();

        if (velocity_constraints.size() == 1) {
          for (const auto& pair : velocity_constraints) {
              unsigned dir = pair.first;
              outward_normal_[dir] = 0;
          }
          outward_normal_ /= outward_normal_.norm();
          break;
        }

        else if (velocity_constraints.size() > 1) {
          node_indices.push_back(i);
        } 
    }

    if (Tdim == 2){
      if (node_indices.size() == 2) {
        Eigen::Matrix<double, Tdim, 1> relative_coord = 
                                nodes_[node_indices[0]]->coordinates() - 
                                nodes_[node_indices[1]]->coordinates();
        outward_normal_ = (outward_normal_.dot(relative_coord) / 
                          relative_coord.dot(relative_coord)) * relative_coord; 
        outward_normal_ /= outward_normal_.norm(); 
      }
    }
    else if (Tdim == 3){
      // TODO
    }
  }

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Judge the heat flux type
    // Type 1: convective heat flux 
    if (convective) {
      const double heat_transfer_coeff = vfm_param1;
      const double ambient_temperature = vfm_param2;
      // Only the particles related surface node matter
      if (nodes_[i]->free_surface()) {
        double boundary_heat_flux = 0;
        for (unsigned j = 0; j < Tdim; ++j){
          boundary_heat_flux += dn_dx_(i, j) * heat_transfer_coeff * 
                                (ambient_temperature - temperature_) * 
                                outward_normal_[j];
        }
          boundary_heat_flux *= this->volume_;
          nodes_[i]->update_external_heat(true, mpm::ParticlePhase::Solid, 
                                            boundary_heat_flux);
      }
    } 
    // Type 2: conductive heat flux
    else {
      const double const_heat_flux = vfm_param1;
      // Only the particles related surface node matter
      if (nodes_[i]->free_surface()) {
        double boundary_heat_flux = 0;
        for (unsigned j = 0; j < Tdim; ++j){
          boundary_heat_flux += dn_dx_(i, j) * const_heat_flux * 
                                outward_normal_[j];
        }
        boundary_heat_flux *= this->volume_;
        nodes_[i]->update_external_heat(true, mpm::ParticlePhase::Solid, 
                                          boundary_heat_flux);
      }
    }
  }
}

// Compute temperature gradient of the particle
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, 1> mpm::Particle<Tdim>::
                      compute_temperature_gradient(unsigned phase) noexcept {
  Eigen::Matrix<double, Tdim, 1> temperature_gradient;
  temperature_gradient.setZero();
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    double temperature = nodes_[i]->temperature(phase);
    for (unsigned j = 0; j < Tdim; ++j) {
      // temperature_gradient = partial T / partial X = T_{i,j}
      temperature_gradient[j] += dn_dx_(i, j) * temperature;
      if (std::fabs(temperature_gradient[j]) < 1.E-15)
        temperature_gradient[j] = 0.;
    }
  }
  this->temperature_gradient_ = temperature_gradient;
  return temperature_gradient_;
}

// Compute mass gradient of the particle
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, 1> mpm::Particle<Tdim>::
                            compute_mass_gradient(unsigned phase) noexcept {

  Eigen::Matrix<double, Tdim, 1> mass_gradient;
  mass_gradient.setZero();
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    double mass = nodes_[i]->mass(phase);
    for (unsigned j = 0; j < Tdim; ++j) {
      // mass_gradient = partial T / partial X = m_{i,j}
      mass_gradient[j] += dn_dx_(i, j) * mass;
      if (std::fabs(mass_gradient[j]) < 1.E-15)
        mass_gradient[j] = 0.;
    }
  }
  this->mass_gradient_ = mass_gradient;
  return mass_gradient_;
}

// Compute updated temperature of the particle
template <unsigned Tdim>
void mpm::Particle<Tdim>::update_particle_temperature(double dt) noexcept {

  // Get PIC temperature
  double PIC_temperature = 0;
  temperature_rate_ = 0.;
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    PIC_temperature +=
        shapefn_[i] * nodes_[i]->temperature(mpm::ParticlePhase::Solid);
    this->temperature_rate_ +=
        shapefn_[i] * nodes_[i]->temperature_rate(
                                              mpm::ParticlePhase::Solid);
  }

  // temperature increment
  temperature_increment_ = this->temperature_rate_ * dt;   
  // Get PIC temperature
  this->temperature_pic_= PIC_temperature;
  // Get FLIP temperature
  this->temperature_flip_ = temperature_ + this->temperature_rate_ * dt;
  // Update particle temperature
  temperature_ = this->temperature_flip_;
}

// Compute thermal strain of the particle
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_thermal_strain() noexcept {
  // get the thermal conductivity coefficient
  const double beta_solid =
    this->material(mpm::ParticlePhase::Solid)
        ->template property<double>(std::string("thermal_expansivity"));

  // compute thermal strain increment
  for (unsigned i = 0; i < Tdim; i++) {
    dthermal_strain_[i] = -1 * beta_solid * temperature_increment_;
  }

  // Compute volumetric thermal strain
  dthermal_volumetric_strain_ = dthermal_strain_.head(Tdim).sum();
  // compute total volumetric strain increment
  dvolumetric_strain_ += dthermal_volumetric_strain_;

  // update thermal strain 
  thermal_strain_ += dthermal_strain_;

  // compute total strain increment
  dstrain_ += dthermal_strain_;
  // compute total strain
  strain_ += dthermal_strain_;
}
