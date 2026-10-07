//! Constructor
template <unsigned Tdim>
mpm::MPMExplicit<Tdim>::MPMExplicit(const std::shared_ptr<IO>& io)
    : mpm::MPMBase<Tdim>(io) {
  //! Logger
  console_ = spdlog::get("MPMExplicit");
  //! Stress update
  if (this->stress_update_ == "usl")
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSL<Tdim>>(mesh_, dt_);
  else if (this->stress_update_ == "musl")
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeMUSL<Tdim>>(mesh_, dt_);
  else
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSF<Tdim>>(mesh_, dt_);

  //! Interface scheme
  if (this->interface_)
    contact_ = std::make_shared<mpm::ContactFriction<Tdim>>(mesh_);
  else
    contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
}

//! MPM Explicit solver
template <unsigned Tdim>
bool mpm::MPMExplicit<Tdim>::solve() {
  bool status = true;

  console_->info("MPM analysis type {}", io_->analysis_type());

  // Initialise MPI rank and size
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Phase
  const unsigned phase = 0;

  // Test if checkpoint resume is needed
  bool resume = false;
  if (analysis_.find("resume") != analysis_.end())
    resume = analysis_["resume"]["resume"].template get<bool>();

  // Enable repartitioning if resume is done with particles generated outside
  // the MPM code.
  bool repartition = false;
  if (analysis_.find("resume") != analysis_.end() &&
      analysis_["resume"].find("repartition") != analysis_["resume"].end())
    repartition = analysis_["resume"]["repartition"].template get<bool>();

  // Pressure smoothing
  pressure_smoothing_ = io_->analysis_bool("pressure_smoothing");

  // Interface
  interface_ = io_->analysis_bool("interface");

  // Initialise material
  this->initialise_materials();

  // Initialise mesh
  this->initialise_mesh();

  // Check point resume
  if (resume) {
    bool check_resume = this->checkpoint_resume();
    if (!check_resume) resume = false;
  }

  // Resume or Initialise
  bool initial_step = (resume == true) ? false : true;
  if (resume) {
    if (repartition) {
      this->mpi_domain_decompose(initial_step);
    } else {
      mesh_->resume_domain_cell_ranks();
#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
      MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
    }
    //! Particle entity sets and velocity constraints
    this->particle_entity_sets(false);
    this->particle_velocity_constraints();
  } else {
    // Initialise particles
    this->initialise_particles();

    // Compute mass
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

    // Domain decompose
    this->mpi_domain_decompose(initial_step);
  }

  // Create nodal properties
  if (interface_ or absorbing_boundary_) mesh_->create_nodal_properties();

  // Initialise loading conditions
  this->initialise_loads();

  // Interlayer contact for 3D printing
  if (this->three_d_printing_ && this->layer_contact_) {
    if (velocity_update_ != mpm::VelocityUpdate::FLIP &&
        velocity_update_ != mpm::VelocityUpdate::PIC)
      throw std::runtime_error(
          "3D printing layer_contact requires the FLIP or PIC velocity update");
    if (interface_)
      throw std::runtime_error(
          "3D printing layer_contact cannot be combined with interface");
    mpm_scheme_->enable_layer_contact(this->layer_contact_gap_tolerance_);
  }

  // Restrict per-step node / cell loops to the region with particles
  // ("active_region": false in analysis reverts to whole-mesh loops)
  mesh_->enable_active_region(analysis_.value("active_region", true));
  // Features that write to nodes outside that region need a full reset
  mesh_->always_full_node_reset(interface_ || this->set_node_concentrated_force_ ||
                                this->absorbing_boundary_);

  // Write initial outputs
  if (!resume) this->write_outputs(this->step_);

  auto solver_begin = std::chrono::steady_clock::now();
  // Main loop
  for (; step_ < nsteps_; ++step_) {

    if (mpi_rank == 0) console_->info("Step: {} of {}.\n", step_, nsteps_);

#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
    // Run load balancer at a specified frequency
    if (step_ % nload_balance_steps_ == 0 && step_ != 0)
      this->mpi_domain_decompose(false);
#endif
#endif

    // Inject particles
    mesh_->inject_particles(step_ * dt_);

    if (this->three_d_printing_) {
      // Update printing state (segment, nozzle velocity and position)
      this->update_printing_state(step_ * dt_);

      // Inject particles (only while the nozzle path is running)
      if (this->printing_active())
        mesh_->inject_particles_3dp(step_ * dt_, dt_, this->nozzle_position(),
                                    this->nozzle_radius());

      // Locate particles
      mpm_scheme_->locate_particles(this->locate_particles_);

      // Interlayer contact: layer (and velocity field) of each particle
      if (this->layer_contact_) {
        const Eigen::Matrix<double, Tdim, 1> nozzle_pos = this->nozzle_position();
        const double nozzle_r = this->nozzle_radius();
        const int layer = this->current_layer();
        const double z_bed = this->layer_bed_;
        const double height = this->layer_height_;
        mesh_->iterate_over_particles(
            [&nozzle_pos, nozzle_r, layer, z_bed,
             height](std::shared_ptr<mpm::ParticleBase<Tdim>> ptr) {
              ptr->update_layer_contact_layer(nozzle_pos, nozzle_r, layer,
                                              z_bed, height);
            });
      }
    }

    // Initialise nodes, cells and shape functions
    mpm_scheme_->initialise();

    // Initialise nodal properties and append material ids to node
    contact_->initialise();

    if (this->three_d_printing_) {
      // Drive the nodes of particles inside the nozzle with the nozzle
      // velocity (nozzle travel + extrusion)
      const Eigen::Matrix<double, Tdim, 1> nozzle_pos = this->nozzle_position();
      const Eigen::Matrix<double, Tdim, 1> nozzle_vel = this->total_velocity();
      const double nozzle_r = this->nozzle_radius();
      mesh_->iterate_over_particles(
          [&nozzle_pos, &nozzle_vel,
           nozzle_r](std::shared_ptr<mpm::ParticleBase<Tdim>> ptr) {
            ptr->map_3D_printing_velocity(nozzle_pos, nozzle_r, nozzle_vel);
          });
      // Nodes shared between MPI ranks: nozzle node if any rank flagged it
      mesh_->sync_3dp_nozzle_nodes(nozzle_vel);
    }

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->compute_nodal_kinematics(velocity_update_, phase);

    // Map material properties to nodes
    contact_->compute_contact_forces();

    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_);

    // Compute forces
    mpm_scheme_->compute_forces(gravity_, phase, step_,
                                set_node_concentrated_force_);

    // Apply Absorbing Constraint
    if (absorbing_boundary_) {
      mpm_scheme_->absorbing_boundary_properties();
      this->nodal_absorbing_constraints();
    }

    // Particle kinematics
    mpm_scheme_->compute_particle_kinematics(velocity_update_, blending_ratio_,
                                             phase, "Cundall", damping_factor_,
                                             step_);

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->postcompute_nodal_kinematics(velocity_update_, phase);

    // Update Stress Last
    mpm_scheme_->postcompute_stress_strain(phase, pressure_smoothing_);

    // Locate particles
    mpm_scheme_->locate_particles(this->locate_particles_);

#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
    mesh_->transfer_halo_particles();
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

    // Write outputs
    this->write_outputs(this->step_ + 1);
  }
  auto solver_end = std::chrono::steady_clock::now();
  console_->info("Rank {}, Explicit {} solver duration: {} ms", mpi_rank,
                 mpm_scheme_->scheme(),
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     solver_end - solver_begin)
                     .count());

  return status;
}
