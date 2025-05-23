//! Constructor of stress update with mesh
template <unsigned Tdim>
mpm::MPMScheme<Tdim>::MPMScheme(const std::shared_ptr<mpm::Mesh<Tdim>>& mesh,
                                double dt) {
  // Assign mesh
  mesh_ = mesh;
  // Assign time increment
  dt_ = dt;
#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
#endif
}

//! Initialize nodes, cells and shape functions
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::initialise() {
#pragma omp parallel sections
  {
    // Spawn a task for initialising nodes and cells
#pragma omp section
    {
      // Initialise nodes
      mesh_->iterate_over_nodes(
          std::bind(&mpm::NodeBase<Tdim>::initialise, std::placeholders::_1));

      mesh_->iterate_over_cells(
          std::bind(&mpm::Cell<Tdim>::activate_nodes, std::placeholders::_1));
    }
    // Spawn a task for particles
#pragma omp section
    {
      // Iterate over each particle to compute shapefn
      mesh_->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Tdim>::compute_shapefn, std::placeholders::_1));
    }
  }  // Wait to complete
}

//! Compute nodal kinematics - map mass and momentum to nodes
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::compute_nodal_kinematics(
    mpm::VelocityUpdate velocity_update, unsigned phase) {
  // Assign mass and momentum to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_mass_momentum_to_nodes,
                std::placeholders::_1, velocity_update));

#ifdef USE_MPI
  // Run if there is more than a single MPI task
  if (mpi_size_ > 1) {
    // MPI all reduce nodal mass
    mesh_->template nodal_halo_exchange<double, 1>(
        std::bind(&mpm::NodeBase<Tdim>::mass, std::placeholders::_1, phase),
        std::bind(&mpm::NodeBase<Tdim>::update_mass, std::placeholders::_1,
                  false, phase, std::placeholders::_2));
    // MPI all reduce nodal momentum
    mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
        std::bind(&mpm::NodeBase<Tdim>::momentum, std::placeholders::_1, phase),
        std::bind(&mpm::NodeBase<Tdim>::update_momentum, std::placeholders::_1,
                  false, phase, std::placeholders::_2));
  }
#endif

  // Compute nodal velocity
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::compute_velocity, std::placeholders::_1),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
}

//! Compute stress and strain
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::compute_stress_strain(
    unsigned phase, bool pressure_smoothing) {

  // Iterate over each particle to update deformation gradient increment
  mesh_->iterate_over_particles(std::bind(
      static_cast<void (mpm::ParticleBase<Tdim>::*)(double)>(
          &mpm::ParticleBase<Tdim>::update_deformation_gradient_increment),
      std::placeholders::_1, dt_));

  // Iterate over each particle to calculate strain
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_strain, std::placeholders::_1, dt_));

  // Iterate over each particle to update particle volume
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::update_volume, std::placeholders::_1));

  // Pressure smoothing
  if (pressure_smoothing) this->pressure_smoothing(phase);

  // Iterate over each particle to compute stress
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_stress, std::placeholders::_1, dt_));

  // Iterate over each particle to update deformation gradient
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::update_deformation_gradient,
                std::placeholders::_1));
}

//! Pressure smoothing
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::pressure_smoothing(unsigned phase) {
  // Assign pressure to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_pressure_to_nodes,
                std::placeholders::_1, phase));

#ifdef USE_MPI
  // Run if there is more than a single MPI task
  if (mpi_size_ > 1)
    // MPI all reduce nodal pressure
    mesh_->template nodal_halo_exchange<double, 1>(
        std::bind(&mpm::NodeBase<Tdim>::pressure, std::placeholders::_1, phase),
        std::bind(&mpm::NodeBase<Tdim>::assign_pressure, std::placeholders::_1,
                  phase, std::placeholders::_2));
#endif

  // Smooth pressure over particles
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_pressure_smoothing,
                std::placeholders::_1, phase));
}

// Compute forces
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::compute_forces(
    const Eigen::Matrix<double, Tdim, 1>& gravity, unsigned phase,
    unsigned step, bool concentrated_nodal_forces) {
  // Spawn a task for external force
#pragma omp parallel sections
  {
#pragma omp section
    {
      // Iterate over each particle to compute nodal body force
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_body_force,
                    std::placeholders::_1, gravity));

      // Apply particle traction and map to nodes
      mesh_->apply_traction_on_particles(step * dt_);

      // Iterate over each node to add concentrated node force to external
      // force
      if (concentrated_nodal_forces)
        mesh_->iterate_over_nodes(
            std::bind(&mpm::NodeBase<Tdim>::apply_concentrated_force,
                      std::placeholders::_1, phase, (step * dt_)));
    }

#pragma omp section
    {
      // Spawn a task for internal force
      // Iterate over each particle to compute nodal internal force
      mesh_->iterate_over_particles(std::bind(
          &mpm::ParticleBase<Tdim>::map_internal_force, std::placeholders::_1));
    }
  }  // Wait for tasks to finish

#ifdef USE_MPI
  // Run if there is more than a single MPI task
  if (mpi_size_ > 1) {
    // MPI all reduce external force
    mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
        std::bind(&mpm::NodeBase<Tdim>::external_force, std::placeholders::_1,
                  phase),
        std::bind(&mpm::NodeBase<Tdim>::update_external_force,
                  std::placeholders::_1, false, phase, std::placeholders::_2));
    // MPI all reduce internal force
    mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
        std::bind(&mpm::NodeBase<Tdim>::internal_force, std::placeholders::_1,
                  phase),
        std::bind(&mpm::NodeBase<Tdim>::update_internal_force,
                  std::placeholders::_1, false, phase, std::placeholders::_2));
  }
#endif
}

// Assign Absorbing Boundary Properties
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::absorbing_boundary_properties() {
  // Initialise nodal properties
  mesh_->initialise_nodal_properties();

  // Append material ids to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::append_material_id_to_nodes,
                std::placeholders::_1));

  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_wave_velocities_to_nodes,
                std::placeholders::_1));
  // Map multimaterial displacements from particles to nodes
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::map_multimaterial_displacements_to_nodes,
      std::placeholders::_1));
}

// Compute particle kinematics
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::compute_particle_kinematics(
    mpm::VelocityUpdate velocity_update, double blending_ratio, unsigned phase,
    const std::string& damping_type, double damping_factor, unsigned step) {

  // Update nodal acceleration constraints
  mesh_->update_nodal_acceleration_constraints(step * dt_);

  // Check if damping has been specified and accordingly Iterate over
  // active nodes to compute acceleratation and velocity
  if (damping_type == "Cundall")
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::compute_acceleration_velocity_cundall,
                  std::placeholders::_1, phase, dt_, damping_factor),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
  else
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::compute_acceleration_velocity,
                  std::placeholders::_1, phase, dt_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  // Iterate over each particle to compute updated position
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_updated_position,
                std::placeholders::_1, dt_, velocity_update, blending_ratio));

  // Apply particle velocity constraints
  mesh_->apply_particle_velocity_constraints();
}

// Locate particles
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::locate_particles(bool locate_particles) {

  auto unlocatable_particles = mesh_->locate_particles_mesh();

  // Throw error with listed unlocatable particles
  if (!unlocatable_particles.empty() && locate_particles) {
    std::ostringstream unloc_mp;
    for (const auto& particle : unlocatable_particles)
      unloc_mp << particle->id() << " ";
    throw std::runtime_error("Particle(s) outside the mesh domain: " +
                             unloc_mp.str());
  }
  // If unable to locate particles remove particles
  if (!unlocatable_particles.empty() && !locate_particles)
    for (const auto& remove_particle : unlocatable_particles)
      mesh_->remove_particle(remove_particle);
}
