//! Constructor
template <unsigned Tdim>
mpm::MPMExplicitThermal<Tdim>::MPMExplicitThermal(
    const std::shared_ptr<IO>& io)
    : mpm::MPMBase<Tdim>(io) {
  //! Logger
  console_ = spdlog::get("MPMExplicitThermal");
}

//! MPM Explicit compute stress strain
template <unsigned Tdim>
void mpm::MPMExplicitThermal<Tdim>::compute_stress_strain() {

  // Iterate over each particle to calculate strain of soil_skeleton
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_strain, std::placeholders::_1, dt_));

  // Iterate over each particle to calculate strain of soil_skeleton
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_thermal_strain, std::placeholders::_1));

  // Iterate over each particle to update particle volume
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::update_volume, std::placeholders::_1));

  // Iterate over each particle to compute stress of soil skeleton
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_stress, std::placeholders::_1, dt_));

  // Pressure smoothing
  if (pressure_smoothing_) this->pressure_smoothing(mpm::ParticlePhase::Solid);
}

// Compute time step size
template <unsigned Tdim>
void mpm::MPMExplicitThermal<Tdim>::compute_critical_timestep_size(double dt) {
  const unsigned phase = 0;
  // cell minimum size
  auto mesh_props = io_->json_object("mesh");
  // Get Mesh reader from JSON object
  double cellsize_min = mesh_props.at("cellsize_min").template get<double>();
  // Material parameters 
  auto materials =  materials_.at(phase);
  double youngs_modulus = materials->template property<double>(std::string("youngs_modulus"));
  double poisson_ratio = materials->template property<double>(std::string("poisson_ratio"));
  double density = materials->template property<double>(std::string("density"));
  double specific_heat = materials->template property<double>(std::string("specific_heat"));
  double heat_conduction = materials->template property<double>(std::string("thermal_conductivity"));  
  // Compute timestep fpor one phase MPM                              
  double critical_dt1 = cellsize_min / std::pow(youngs_modulus/density, 0.5);
  console_->info("Critical time step size for elastic wave propagation is {} s", critical_dt1);

  // Compute timestep for heat transfer eqaution - pure liquid
  double critical_dt2 = cellsize_min * cellsize_min * density * specific_heat / 
                        heat_conduction;                                  
  console_->info("Critical time step size for thermal conduction is {} s", critical_dt2);

  if (dt >= std::min(critical_dt1, critical_dt2))
      throw std::runtime_error("Time step size is too large");  
}

//! Thermo-mechanical MPM Explicit solver
template <unsigned Tdim>
bool mpm::MPMExplicitThermal<Tdim>::solve() {
  bool status = true;

  console_->info("MPM analysis type {}", io_->analysis_type());

  std::cout << io_->analysis_type() << "\n";
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
  if (analysis_.find("pressure_smoothing") != analysis_.end())
    pressure_smoothing_ =
        analysis_.at("pressure_smoothing").template get<bool>();
        
  // Free surface detection
  free_surface_detection_ = "none";
  if (analysis_.find("free_surface_detection") != analysis_.end()) {
    free_surface_detection_ = analysis_["free_surface_detection"]["type"]
                                  .template get<std::string>();
    fs_vol_tolerance_ =
        analysis_["free_surface_detection"]["volume_tolerance"]
                                  .template get<double>();
  }

#ifdef USE_MPI
  if (!(free_surface_detection_ == "density" ||
        free_surface_detection_ == "none") &&
      mpi_size > 1) {
    console_->warn(
        "The free-surface detection in MPI setting is automatically set to "
        "default: "
        "\'density\'. Only \'none\' and \'density\' free-surface detection "
        "algorithm are supported for MPI.");
    free_surface_detection_ = "density";
  }
#endif

  // Free surface
  if (analysis_.find("virtual_flux") != analysis_.end()) {
    virtual_flux_ = analysis_["virtual_flux"]["virtual_flux"]
                                                .template get<bool>();
    flux_type_ = analysis_["virtual_flux"]["flux_type"]
                                                .template get<std::string>();
    // Judge the flux type          
    if (flux_type_ == "convective") {
      heat_transfer_coeff_ = analysis_["virtual_flux"]["heat_transfer_coeff"]
                                                .template get<double>();
      ambient_temperature_ = analysis_["virtual_flux"]["ambient_temperature"]
                                                .template get<double>();
    } else if (flux_type_ == "conductive") 
      flux_ = analysis_["virtual_flux"]["flux"].template get<double>();
  }
  // Initialise materials
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

    // Compute mass for each phase
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

    // Domain decompose
    this->mpi_domain_decompose(initial_step);
  }

  // Initialise loading conditions
    this->initialise_loads();

  // Write initial outputs
  if (!resume) this->write_outputs(this->step_);

  auto solver_begin = std::chrono::steady_clock::now();

  this->compute_critical_timestep_size(dt_);

  // Main loop
  for (; step_ < nsteps_; ++step_) {

    current_time_ += dt_;

    if (mpi_rank == 0) console_->info(
                    "uuid : [{}], Step: {} of {}, timestep = {}, time = {}.\n", 
                    uuid_, step_, nsteps_, dt_, current_time_);

#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
    // Run load balancer at a specified frequency
    if (step_ % nload_balance_steps_ == 0 && step_ != 0)
      this->mpi_domain_decompose(false);
#endif
#endif

#pragma omp parallel sections
    {
      // Spawn a task for initialising nodes and cells
#pragma omp section
      {
        // Initialise nodes
        mesh_->iterate_over_nodes(
            std::bind(&mpm::NodeBase<Tdim>::initialise, std::placeholders::_1));

        // Initialise nodes
        mesh_->iterate_over_nodes(
            std::bind(&mpm::NodeBase<Tdim>::initialise_thermal, 
            std::placeholders::_1));

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

    // Assign mass and momentum to nodes
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_mass_momentum_to_nodes,
                  std::placeholders::_1, velocity_update_));

    // Assign heat capacity and heat to nodes
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_heat_to_nodes,
                  std::placeholders::_1));

#ifdef USE_MPI
    // Run if there is more than a single MPI task
    if (mpi_size > 1) {
      // MPI all reduce nodal mass
      mesh_->template nodal_halo_exchange<double, 1>(
          std::bind(&mpm::NodeBase<Tdim>::mass, std::placeholders::_1, phase),
          std::bind(&mpm::NodeBase<Tdim>::update_mass, std::placeholders::_1,
                    false, phase, std::placeholders::_2));
      // MPI all reduce nodal momentum
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::momentum, std::placeholders::_1, phase),
          std::bind(&mpm::NodeBase<Tdim>::update_momentum,
                    std::placeholders::_1, false, phase,
                    std::placeholders::_2));
      // // MPI all reduce nodal heat capacity
      // mesh_->template nodal_halo_exchange<double, 1>(
      //     std::bind(&mpm::NodeBase<Tdim>::heat_capacity, std::placeholders::_1, phase),
      //     std::bind(&mpm::NodeBase<Tdim>::update_heat_capacity, std::placeholders::_1,
      //               false, phase, std::placeholders::_2));
      // // MPI all reduce nodal heat
      // mesh_->template nodal_halo_exchange<double, 1>(
      //     std::bind(&mpm::NodeBase<Tdim>::heat, std::placeholders::_1, phase),
      //     std::bind(&mpm::NodeBase<Tdim>::update_heat,
      //               std::placeholders::_1, false, phase,
      //               std::placeholders::_2));
    }
#endif

    // Compute free surface cells, nodes, and particles
    if (free_surface_detection_ != "none") {
      mesh_->compute_free_surface(free_surface_detection_, fs_vol_tolerance_,
                                  cell_neighbourhood_);
    }

    // Compute nodal velocity
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::compute_velocity,
                  std::placeholders::_1),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
    
    // Compute nodal temperature
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::compute_temperature_explicit,
                  std::placeholders::_1, phase, this->dt_, this->step_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));  

    // Update stress first
    if (this->stress_update_ == "usf") this->compute_stress_strain();

#pragma omp parallel sections
    {
#pragma omp section
      {
        // Iterate over particles to compute nodal body force
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_body_force,
                      std::placeholders::_1, this->gravity_));

        // Apply particle traction and map to nodes
        mesh_->apply_traction_on_particles(this->step_ * this->dt_);

        // Iterate over each node to add concentrated node force to external
        // force
        if (set_node_concentrated_force_)
          mesh_->iterate_over_nodes(
              std::bind(&mpm::NodeBase<Tdim>::apply_concentrated_force,
                        std::placeholders::_1, mpm::ParticlePhase::Solid,
                        (this->step_ * this->dt_)));
      }

#pragma omp section
      // Spawn a task for internal force
      {
        // Iterate over each particle to compute nodal internal force
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_internal_force,
                      std::placeholders::_1));
      }

#pragma omp section
      // Spawn a task for heat conduction
      {
        // Iterate over each particle to compute nodal heat conduction
        mesh_->iterate_over_particles(std::bind(
            &mpm::ParticleBase<Tdim>::map_heat_conduction, std::placeholders::_1));

        // Iterate over each particle to compute nodal heat conduction
        if (virtual_flux_) {
          if (flux_type_ == "convective") 
            mesh_->iterate_over_particles(std::bind(
                &mpm::ParticleBase<Tdim>::map_virtual_heat_flux, std::placeholders::_1,
                true, heat_transfer_coeff_, ambient_temperature_));
          else if (flux_type_ == "conductive")
            mesh_->iterate_over_particles(std::bind(
                &mpm::ParticleBase<Tdim>::map_virtual_heat_flux, std::placeholders::_1,
                false, flux_, 0));
          else
            throw std::runtime_error("Virtual heat flux boudary is not correctly applied");
        }
      }
    }

#ifdef USE_MPI
    // Run if there is more than a single MPI task
    if (mpi_size > 1) {
      // MPI all reduce external force of mixture
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::external_force, std::placeholders::_1,
                    phase),
          std::bind(&mpm::NodeBase<Tdim>::update_external_force,
                    std::placeholders::_1, false, phase,
                    std::placeholders::_2));

      // MPI all reduce internal force of mixture
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::internal_force, std::placeholders::_1,
                    phase),
          std::bind(&mpm::NodeBase<Tdim>::update_internal_force,
                    std::placeholders::_1, false, phase,
                    std::placeholders::_2));

      // // MPI all reduce heat conduction
      // mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
      //     std::bind(&mpm::NodeBase<Tdim>::heat_conduction, std::placeholders::_1,
      //               phase),
      //     std::bind(&mpm::NodeBase<Tdim>::update_internal_heat,
      //               std::placeholders::_1, false, phase,
      //               std::placeholders::_2));
    }
#endif

    // Compute nodal acceleration and update nodal velocity
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::compute_acceleration_velocity,
                  std::placeholders::_1, phase, this->dt_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    // Compute nodal temperature rate(dot) and update nodal temperature
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::update_temperature_explicit,
                  std::placeholders::_1, phase, this->dt_, this->step_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    // Iterate over each particle to compute updated temperature
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::update_particle_temperature,
        std::placeholders::_1, this->dt_));

    // Update particle position and kinematics
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_updated_position,
        std::placeholders::_1, this->dt_, velocity_update_, blending_ratio_));

    // Apply particle velocity constraints
    mesh_->apply_particle_velocity_constraints();

    // // Update Stress Last
    // if (this->stress_update_ == "usl") this->compute_stress_strain();

    // Locate particles
    auto unlocatable_particles = mesh_->locate_particles_mesh();

    if (!unlocatable_particles.empty() && this->locate_particles_)
      throw std::runtime_error("Particle outside the mesh domain");
    // If unable to locate particles remove particles
    if (!unlocatable_particles.empty() && !this->locate_particles_)
      for (const auto& remove_particle : unlocatable_particles)
        mesh_->remove_particle(remove_particle);

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
                  (this->stress_update_ == "usl" ? "USL" : "USF"),
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      solver_end - solver_begin)
                      .count());

  return status;
}
