//! Initialize nodes, cells and shape functions
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::initialise_thermal() {
#pragma omp parallel sections
  {
    // Spawn a task for initialising nodes and cells
#pragma omp section
    {
      // Initialise nodes
      mesh_->iterate_over_nodes(std::bind(
          &mpm::NodeBase<Tdim>::initialise_implicit, std::placeholders::_1));

      // Initialise nodes
      mesh_->iterate_over_nodes(std::bind(
          &mpm::NodeBase<Tdim>::initialise_thermal, std::placeholders::_1));

      mesh_->iterate_over_cells(
          std::bind(&mpm::Cell<Tdim>::activate_nodes, std::placeholders::_1));
    }
    // Spawn a task for particles
#pragma omp section
    {
      // Iterate over each particle to compute shapefn
      mesh_->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Tdim>::compute_shapefn, std::placeholders::_1));

      // Initialise material
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::initialise_constitutive_law,
                    std::placeholders::_1, dt_));
    }
  }  // Wait to complete
}

//! Compute nodal temperature and temperature rate
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_nodal_temperatures(
                                      unsigned phase, double dt, Index step) {
  // Assign heat capacity, heat, and internal heat to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_heat_to_nodes_newmark,
                std::placeholders::_1));

  // Compute nodal velocity and acceleration
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::compute_temperature_implicit,
                std::placeholders::_1, phase, dt, step),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
}

//! Update nodal kinematics by Newmark scheme
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::update_nodal_thermokinematics_newmark(
    unsigned phase, double newmark_beta, double newmark_gamma, Index step) {

  // Update nodal velocity and acceleration
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::update_velocity_acceleration_newmark,
                std::placeholders::_1, phase, newmark_beta, newmark_gamma, dt_),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  // Update nodal velocity and acceleration
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::update_temperature_variables_newmark,
                std::placeholders::_1, phase, newmark_beta, newmark_gamma, 
                dt_, step),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1)); 
}

//! Compute stress and strain by Newmark scheme
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_stress_strain_thermal(
    unsigned phase, bool pressure_smoothing) {

  // Iterate over each particle to calculate strain and volume using nodal
  // displacement
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_strain_volume_newmark_thermal,
                std::placeholders::_1));

  // Pressure smoothing
  if (pressure_smoothing) this->pressure_smoothing(phase);

  // Iterate over each particle to compute stress
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_stress_newmark,
                std::placeholders::_1, dt_));
}

// Compute forces
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_heats(bool vfm,  
                    std::string ftype, double vfm_param1, double vfm_param2) {
  // Spawn a task for external force
#pragma omp parallel sections
  {
#pragma omp section
    {
      // Iterate over each particle to compute nodal internal heat
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_heat_conduction,
                    std::placeholders::_1));

      // Iterate over each particle to compute external heat
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_plastic_heat_dissipation,
                    std::placeholders::_1, dt_));

      // Iterate over each particle to compute heat flux
      if (vfm) {
        if (ftype == "convective") {
          mesh_->iterate_over_particles(std::bind(
              &mpm::ParticleBase<Tdim>::map_virtual_heat_flux, std::placeholders::_1,
              true, vfm_param1, vfm_param2));
        }    
        else if (ftype == "conductive") {
          mesh_->iterate_over_particles(std::bind(
              &mpm::ParticleBase<Tdim>::map_virtual_heat_flux, std::placeholders::_1,
              false, vfm_param1, 0));
        }
      }                                        
    }
#pragma omp section
    {
      // Iterate over each particle to compute nodal transient heat
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_heat_rate_to_nodes,
                    std::placeholders::_1));
    }
  }  // Wait for tasks to finish
}

// Update particle temperature
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_particle_temperature() {

  // Iterate over each particle to compute updated position
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_updated_temperature_newmark,
                std::placeholders::_1, dt_));
}
