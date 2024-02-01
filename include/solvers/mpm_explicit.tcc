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

  // LEDT check json file (see mpm_explicit.tcc)

  if (this->interface_)
    if (this->interface_type_ == "multimaterial")
      contact_ = std::make_shared<mpm::ContactFriction<Tdim>>(mesh_);
    else if (this->interface_type_ == "levelset")
      contact_ = std::make_shared<mpm::ContactLevelset<Tdim>>(mesh_);
    else  // default is "none"
      contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
  else
    contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);

  std::cout << "-->MPMExplicit interface_ " << this->interface_
            << std::endl;  // LEDT REMOVE!
  std::cout << "-->MPMExplicit interface_type_ " << this->interface_type_
            << std::endl;  // LEDT REMOVE!
}

// //! Interface scheme //LEDT get from mpm.base
// std::string get_interface_scheme(MPMBase& mpm_base) {
//   if (this->interface_)
//     if (this->interface_type_ == "multimaterial")
//       contact_ = std::make_shared<mpm::ContactFriction<Tdim>>(mesh_);
//     else if (this->interface_type_ == "levelset")
//       contact_ = std::make_shared<mpm::ContactLevelset<Tdim>>(mesh_);
//     else  // default is "none"
//       contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
//   else
//     contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
//   return contact_;
// };

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
  // interface_ = io_->analysis_bool("interface");  // LEDT remove

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

  // Write initial outputs
  if (!resume) this->write_outputs(this->step_);
  std::cout << "-->9 prior to looping steps" << std::endl;  // LEDT REMOVE!

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

    // std::cout << "-->1" << std::endl;  // LEDT REMOVE!
    // Inject particles
    mesh_->inject_particles(step_ * dt_);

    // std::cout << "-->2" << std::endl;  // LEDT REMOVE!
    // Initialise nodes, cells and shape functions
    mpm_scheme_->initialise();

    // std::cout << "-->3" << std::endl;  // LEDT REMOVE!
    // Initialise nodal properties and append material ids to node
    contact_->initialise();  // LEDT check
    // std::cout << "-->e1 contact_> initialise()" << std::endl;  // LEDT
    // REMOVE!

    // std::cout << "-->4" << std::endl;  // LEDT REMOVE!
    // Mass momentum and compute velocity at nodes
    mpm_scheme_->compute_nodal_kinematics(velocity_update_, phase);

    // std::cout << "-->5" << std::endl;  // LEDT REMOVE!
    // Map material properties to nodes
    contact_->compute_contact_forces(dt_);  // LEDT check

    // std::cout << "-->6" << std::endl;  // LEDT REMOVE!
    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_,
                                          stress_rate_);

    // std::cout << "-->7" << std::endl;  // LEDT REMOVE!
    // Compute forces
    mpm_scheme_->compute_forces(gravity_, phase, step_,
                                set_node_concentrated_force_);

    // std::cout << "-->8" << std::endl;  // LEDT REMOVE!
    // Apply Absorbing Constraint
    if (absorbing_boundary_) {
      mpm_scheme_->absorbing_boundary_properties();
      this->nodal_absorbing_constraints();
    }

    // std::cout << "-->9" << std::endl;  // LEDT REMOVE!
    // Particle kinematics
    mpm_scheme_->compute_particle_kinematics(velocity_update_, blending_ratio_,
                                             phase, "Cundall", damping_factor_,
                                             step_, update_defgrad_);

    // std::cout << "-->10" << std::endl;  // LEDT REMOVE!
    // Mass momentum and compute velocity at nodes
    mpm_scheme_->postcompute_nodal_kinematics(velocity_update_, phase);

    // std::cout << "-->11" << std::endl;  // LEDT REMOVE!
    // Update Stress Last
    mpm_scheme_->postcompute_stress_strain(phase, pressure_smoothing_,
                                           stress_rate_);

    // std::cout << "-->12" << std::endl;  // LEDT REMOVE!
    // Locate particles
    mpm_scheme_->locate_particles(this->locate_particles_);
    // std::cout << "-->13" << std::endl;  // LEDT REMOVE!

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
