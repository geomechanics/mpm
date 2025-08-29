//! Constructor
template <unsigned Tdim>
mpm::MPMSchemeNewmark<Tdim>::MPMSchemeNewmark(
    const std::shared_ptr<mpm::Mesh<Tdim>>& mesh, double dt, double beta,
    double gamma, double bossak_alpha)
    : mpm::MPMScheme<Tdim>(mesh, dt),
      beta_(beta),
      gamma_(gamma),
      bossak_alpha_(bossak_alpha) {}

//! Initialize nodes, cells and shape functions
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::initialise() {
#pragma omp parallel sections
  {
    // Spawn a task for initialising nodes and cells
#pragma omp section
    {
      // Initialise nodes
      mesh_->iterate_over_nodes(std::bind(
          &mpm::NodeBase<Tdim>::initialise_implicit, std::placeholders::_1));

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

//! Compute nodal kinematics - map mass, momentum and inertia to nodes
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_nodal_kinematics(
    mpm::VelocityUpdate velocity_update, unsigned phase, unsigned step) {
  // Assign mass, momentum and inertia to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_mass_momentum_inertia_to_nodes,
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
    // MPI all reduce nodal inertia
    mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
        std::bind(&mpm::NodeBase<Tdim>::inertia, std::placeholders::_1, phase),
        std::bind(&mpm::NodeBase<Tdim>::update_inertia, std::placeholders::_1,
                  false, phase, std::placeholders::_2));
  }
#endif

  // Compute nodal velocity and acceleration
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::compute_velocity_acceleration,
                std::placeholders::_1),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
}

//! Update nodal kinematics by Newmark scheme
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::update_nodal_kinematics_newmark(
    unsigned phase, double newmark_beta, double newmark_gamma,
    bool pml_boundary) {

  // Update nodal velocity and acceleration
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::update_velocity_acceleration_newmark,
                std::placeholders::_1, phase, newmark_beta, newmark_gamma, dt_),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  if (pml_boundary)
    mesh_->iterate_over_nodes_predicate(
        std::bind(
            &mpm::NodeBase<Tdim>::update_velocity_acceleration_newmark_pml,
            std::placeholders::_1, phase, newmark_beta, newmark_gamma, dt_),
        std::bind(&mpm::NodeBase<Tdim>::pml, std::placeholders::_1));
}

//! Compute stress and strain by Newmark scheme
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_stress_strain(
    unsigned phase, bool pressure_smoothing) {

  // Iterate over each particle to calculate strain and volume using nodal
  // displacement
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_strain_volume_newmark,
                std::placeholders::_1, dt_));

  // Pressure smoothing
  if (pressure_smoothing) this->pressure_smoothing(phase);

  // Iterate over each particle to compute stress
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_stress_newmark,
                std::placeholders::_1, dt_));
}

//! Precompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::precompute_stress_strain(
    unsigned phase, bool pressure_smoothing) {}

//! Postcompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::postcompute_stress_strain(
    unsigned phase, bool pressure_smoothing) {
  this->compute_stress_strain(phase, pressure_smoothing);
}

// Compute forces
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_forces(
    const Eigen::Matrix<double, Tdim, 1>& gravity, unsigned phase,
    unsigned step, bool concentrated_nodal_forces, bool quasi_static) {
  // Spawn a task for external force
#pragma omp parallel sections
  {
#pragma omp section
    {
      // Iterate over each particle to compute nodal body force
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_body_force,
                    std::placeholders::_1, gravity));

      // Iterate over each particle to compute nodal inertial force
      if (!quasi_static)
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_inertial_force,
                      std::placeholders::_1, bossak_alpha_));

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
      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_internal_force,
                    std::placeholders::_1, dt_));
    }
  }  // Wait for tasks to finish
}

// Update particle kinematics
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::compute_particle_kinematics(
    mpm::VelocityUpdate velocity_update, double blending_ratio, unsigned phase,
    const mpm::Damping damping_type, double damping_factor, unsigned step,
    bool update_defgrad, bool pml_boundary) {

  // Iterate over each particle to compute updated position
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::compute_updated_position_newmark,
                std::placeholders::_1, dt_, gamma_, step, velocity_update,
                blending_ratio));

  // Iterate over each particle to update deformation gradient
  if (update_defgrad)
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::update_deformation_gradient,
                  std::placeholders::_1));
}

// Update particle stress, strain and volume
template <unsigned Tdim>
inline void
    mpm::MPMSchemeNewmark<Tdim>::update_particle_stress_strain_volume() {
  // Iterate over each particle to update particle stress and strain
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::update_stress_strain,
                std::placeholders::_1, dt_));
}

//! Postcompute nodal kinematics - map mass and momentum to nodes
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::postcompute_nodal_kinematics(
    mpm::VelocityUpdate velocity_update, unsigned phase, unsigned step) {}

//! Stress update scheme
template <unsigned Tdim>
inline std::string mpm::MPMSchemeNewmark<Tdim>::scheme() const {
  return "Newmark";
}

// Assign PML Boundary Properties
template <unsigned Tdim>
inline void mpm::MPMSchemeNewmark<Tdim>::initialise_pml_boundary_properties(
    const bool& pml_type) {
  // Initialise nodal properties
  mesh_->initialise_nodal_properties();

  // Map damped mass to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_pml_properties_to_nodes,
                std::placeholders::_1));

// Halo exchange for all mapped pml properties
#ifdef USE_MPI
  // Run if there is more than a single MPI task
  if (mpi_size_ > 1) {
    // All reduce node boolean status of PML
    mesh_->assign_pml_nodes();

    // MPI all reduce nodal damped mass
    mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
        std::bind(&mpm::NodeBase<Tdim>::property, std::placeholders::_1,
                  "damped_masses", 0, Tdim),
        std::bind(&mpm::NodeBase<Tdim>::update_property, std::placeholders::_1,
                  false, "damped_masses", std::placeholders::_2, 0, Tdim));
    // MPI all reduce nodal damped mass displacement
    mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
        std::bind(&mpm::NodeBase<Tdim>::property, std::placeholders::_1,
                  "damped_mass_displacements", 0, Tdim),
        std::bind(&mpm::NodeBase<Tdim>::update_property, std::placeholders::_1,
                  false, "damped_mass_displacements", std::placeholders::_2, 0,
                  Tdim));

    if (pml_type) {
      // MPI all reduce nodal damped mass
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::property, std::placeholders::_1,
                    "damped_mass_displacements_j1", 0, Tdim),
          std::bind(&mpm::NodeBase<Tdim>::update_property,
                    std::placeholders::_1, false,
                    "damped_mass_displacements_j1", std::placeholders::_2, 0,
                    Tdim));
      // MPI all reduce nodal damped mass displacement
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::property, std::placeholders::_1,
                    "damped_mass_displacements_j2", 0, Tdim),
          std::bind(&mpm::NodeBase<Tdim>::update_property,
                    std::placeholders::_1, false,
                    "damped_mass_displacements_j2", std::placeholders::_2, 0,
                    Tdim));
      // MPI all reduce nodal damped mass
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::property, std::placeholders::_1,
                    "damped_mass_displacements_j3", 0, Tdim),
          std::bind(&mpm::NodeBase<Tdim>::update_property,
                    std::placeholders::_1, false,
                    "damped_mass_displacements_j3", std::placeholders::_2, 0,
                    Tdim));
      // MPI all reduce nodal damped mass displacement
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::property, std::placeholders::_1,
                    "damped_mass_displacements_j4", 0, Tdim),
          std::bind(&mpm::NodeBase<Tdim>::update_property,
                    std::placeholders::_1, false,
                    "damped_mass_displacements_j4", std::placeholders::_2, 0,
                    Tdim));
    }
  }
#endif

  // Recompute velocity for PML nodes
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::compute_pml_velocity_acceleration,
                std::placeholders::_1, pml_type),
      std::bind(&mpm::NodeBase<Tdim>::pml, std::placeholders::_1));
}
