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

    // Point entity sets and velocity constraints
    this->point_entity_sets(false);
    this->point_velocity_constraints();
  } else {
    // Initialise particles
    this->initialise_particles();

    // Compute mass
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

    // Initialise points
    this->initialise_points();

    // Domain decompose
    this->mpi_domain_decompose(initial_step);
  }

  // Create nodal properties
  if (interface_ or absorbing_boundary_) mesh_->create_nodal_properties();

  // Initialise loading conditions
  this->initialise_loads();

    // Hourglass (non-inertial frame) motion table
  bool hourglass_motion = false;
  double hourglass_ramping_time = 0.0;
  mpm::HourglassMotion hourglass_table;
  if (analysis_.find("hourglass_motion") != analysis_.end()) {
    if (rotation_forces_)
      throw std::runtime_error(
          "hourglass_motion cannot be combined with rotation forces");
    hourglass_table.load(io_->file_name(
        analysis_["hourglass_motion"]["file"].template get<std::string>()));
    hourglass_motion = true;
    if (analysis_["hourglass_motion"].contains("ramping_time"))
      hourglass_ramping_time =
          analysis_["hourglass_motion"]["ramping_time"].template get<double>();
    console_->info("Hourglass motion table: {} rows, t_end = {} s",
                   hourglass_table.rows(), hourglass_table.t_end());
  }

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

    // Initialise nodes, cells and shape functions
    mpm_scheme_->initialise();

    // Initialise nodal properties and append material ids to node
    contact_->initialise();

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->compute_nodal_kinematics(velocity_update_, phase);

    // Map material properties to nodes
    contact_->compute_contact_forces();

    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_);

    // Ramp body forces if necessary
    double gravity_multiplier = 1.0;
    double rotation_multiplier = 1.0;
    if (this->gravity_ramping_time_ > 0.0)
      gravity_multiplier = std::min(
          1.0, static_cast<double>(step_) * dt_ / this->gravity_ramping_time_);
    if (this->rotation_ramping_time_ > 0.0)
      rotation_multiplier = std::min(
          1.0, static_cast<double>(step_) * dt_ / this->rotation_ramping_time_);

    // Compute forces
    Eigen::Matrix<double, Tdim, 1> current_gravity = gravity_multiplier * gravity_;
    const auto current_rotation = rotation_multiplier * rotation_omega_;
    mpm::FrameState frame;
    const mpm::FrameState* frame_ptr = nullptr;
    if (hourglass_motion) {
      // t = step * dt (C++ steps start at 0; MATLAB's (step-1)*dt was 1-based)
      frame = hourglass_table.at(static_cast<double>(step_) * dt_);
      double hourglass_multiplier = 1.0;
      if (hourglass_ramping_time > 0.0)
        hourglass_multiplier =
            std::min(1.0, static_cast<double>(step_) * dt_ / hourglass_ramping_time);
      frame.M1 *= hourglass_multiplier;
      frame.M2 *= hourglass_multiplier;
      frame.c *= hourglass_multiplier;
      frame.g *= gravity_multiplier;   // gravity ramp from the gravity JSON object
      frame_ptr = &frame;
      current_gravity.setZero();  // gravity now comes from frame.g
    }
    mpm_scheme_->compute_forces(current_gravity, phase, step_,
                                set_node_concentrated_force_, rotation_forces_,
                                rotation_origin_, current_rotation,
                                rotation_clockwise_, frame_ptr);

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
    mesh_->transfer_halo_points();
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
