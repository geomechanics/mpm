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
  // Next nodal mapping is the first one of the step
  lc_first_mapping_ = true;
#pragma omp parallel sections
  {
    // Spawn a task for initialising nodes and cells
#pragma omp section
    {
      if (mesh_->active_region()) {
        // Reset / activate only where particles are
        mesh_->initialise_active_region(false);
      } else {
        // Initialise nodes
        mesh_->iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::initialise,
                                            std::placeholders::_1));

        mesh_->iterate_over_cells(std::bind(&mpm::Cell<Tdim>::activate_nodes,
                                            std::placeholders::_1));
      }
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
  // Layer contact: a later mapping in the step (MUSL) rebuilds the field
  // momentum
  if (layer_contact_ && !lc_first_mapping_)
    this->layer_contact_reset_momentum();

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
  mesh_->iterate_over_status_nodes(
      std::bind(&mpm::NodeBase<Tdim>::compute_velocity, std::placeholders::_1));

  // Layer contact: field quantities and node states
  if (layer_contact_) {
    if (lc_first_mapping_) {
      this->layer_contact_first_mapping();
      lc_first_mapping_ = false;
    } else {
      this->layer_contact_remapping();
    }
  }
}

//! Layer contact: fields, normals, gaps and node states (first mapping)
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::layer_contact_first_mapping() {
  // Field-1 mass / momentum, mass gradients and welded mass of both fields
  mesh_->iterate_over_particles(
      [](const std::shared_ptr<mpm::ParticleBase<Tdim>>& particle) {
        particle->map_layer_contact_properties();
      });
#ifdef USE_MPI
  if (mpi_size_ > 1) {
    using Sums = typename mpm::NodeBase<Tdim>::LayerContactSums;
    mesh_->template nodal_halo_reduce<Sums, 3 + 3 * Tdim>(
        [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
          return node->layer_contact_sums();
        },
        [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node, const Sums& sums) {
          node->assign_layer_contact_sums(sums);
        },
        Sums::Zero(), MPI_SUM);
  }
#endif

  // Contact normal at nodes holding both fields
  mesh_->iterate_over_status_nodes(
      [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
        node->compute_layer_contact_normal();
      });

  // Surfaces of the two fields along the normal
  mesh_->iterate_over_particles(
      [](const std::shared_ptr<mpm::ParticleBase<Tdim>>& particle) {
        particle->map_layer_contact_extent();
      });
#ifdef USE_MPI
  if (mpi_size_ > 1) {
    using Extent = typename mpm::NodeBase<Tdim>::LayerContactExtent;
    mesh_->template nodal_halo_reduce<Extent, 2>(
        [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
          return node->layer_contact_extent();
        },
        [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node,
           const Extent& extent) { node->assign_layer_contact_extent(extent); },
        Extent::Constant(-std::numeric_limits<double>::max()), MPI_MAX);
  }
#endif

  // Node state from the gap; field velocities at separate nodes
  const double gap_tolerance = lc_gap_tolerance_;
  mesh_->iterate_over_status_nodes(
      [gap_tolerance](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
        node->decide_layer_contact(gap_tolerance);
        node->compute_layer_contact_velocity();
      });

  // Particles next to a contact weld (the layers merge)
  mesh_->iterate_over_particles(
      [](const std::shared_ptr<mpm::ParticleBase<Tdim>>& particle) {
        particle->update_layer_contact_weld();
      });
}

//! Layer contact: reset field mass / momentum before a later mapping
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::layer_contact_reset_momentum() {
  auto reset = [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
    node->reset_layer_contact_momentum();
  };
  if (mpi_size_ > 1 && mesh_->active_region()) {
    mesh_->iterate_over_status_nodes(reset);
    mesh_->iterate_over_domain_shared_nodes(reset);
  } else if (mpi_size_ > 1) {
    mesh_->iterate_over_nodes(reset);
  } else {
    mesh_->iterate_over_status_nodes(reset);
  }
}

//! Layer contact: field momentum and velocities (later mappings)
template <unsigned Tdim>
inline void mpm::MPMScheme<Tdim>::layer_contact_remapping() {
  mesh_->iterate_over_particles(
      [](const std::shared_ptr<mpm::ParticleBase<Tdim>>& particle) {
        particle->map_layer_contact_momentum();
      });
#ifdef USE_MPI
  if (mpi_size_ > 1) {
    using Momentum = typename mpm::NodeBase<Tdim>::LayerContactMomentum;
    mesh_->template nodal_halo_reduce<Momentum, Tdim + 1>(
        [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
          return node->layer_contact_momentum();
        },
        [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node,
           const Momentum& mp) { node->assign_layer_contact_momentum(mp); },
        Momentum::Zero(), MPI_SUM);
  }
#endif
  mesh_->iterate_over_status_nodes(
      [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
        node->compute_layer_contact_velocity();
      });
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
    // Layer contact: field-1 force
    if (layer_contact_)
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>,
                                          Tdim>(
          [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
            return node->layer_contact_force();
          },
          [](const std::shared_ptr<mpm::NodeBase<Tdim>>& node,
             const Eigen::Matrix<double, Tdim, 1>& force) {
            node->assign_layer_contact_force(force);
          });
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
    mesh_->iterate_over_status_nodes(
      std::bind(&mpm::NodeBase<Tdim>::compute_acceleration_velocity_cundall,
                  std::placeholders::_1, phase, dt_, damping_factor));
  else
    mesh_->iterate_over_status_nodes(
      std::bind(&mpm::NodeBase<Tdim>::compute_acceleration_velocity,
                  std::placeholders::_1, phase, dt_));

  // Layer contact: independent field velocities at separate nodes
  if (layer_contact_) {
    const double dt = dt_;
    const double damping = (damping_type == "Cundall") ? damping_factor : 0.;
    mesh_->iterate_over_status_nodes(
        [dt, damping](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
          node->compute_layer_contact_acceleration_velocity(dt, damping);
        });
  }

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
