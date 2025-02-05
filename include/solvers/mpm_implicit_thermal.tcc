//! Constructor
template <unsigned Tdim>
mpm::MPMImplicitThermal<Tdim>::MPMImplicitThermal(const std::shared_ptr<IO>& io)
    : mpm::MPMBase<Tdim>(io) {
  //! Logger
  console_ = spdlog::get("MPMImplicitThermal");

  // Check if stress update is not newmark
  if (stress_update_ != "newmark") {
    console_->warn(
        "The stress_update_ scheme chosen is not available and automatically "
        "set to default: \'newmark\'. Only \'newmark\' scheme is currently "
        "supported for implicit solver.");
    stress_update_ = "newmark";
  }

  // Initialise scheme
  double displacement_tolerance = 1.0e-10;
  double temperature_tolerance = 1.0e-6;
  double residual_tolerance = 1.0e-10;
  double relative_residual_tolerance = 1.0e-6;
  if (stress_update_ == "newmark") {
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeNewmark<Tdim>>(mesh_, dt_);
    if (analysis_.contains("scheme_settings")) {
      // Read boolean of nonlinear analysis
      if (analysis_["scheme_settings"].contains("nonlinear"))
        nonlinear_ =
            analysis_["scheme_settings"].at("nonlinear").template get<bool>();
      // Read boolean of quasi-static analysis
      if (analysis_["scheme_settings"].contains("quasi_static"))
        quasi_static_ = analysis_["scheme_settings"]
                            .at("quasi_static")
                            .template get<bool>();
      // Read parameters of Newmark scheme
      if (analysis_["scheme_settings"].contains("beta"))
        newmark_beta_ =
            analysis_["scheme_settings"].at("beta").template get<double>();
      if (analysis_["scheme_settings"].contains("gamma"))
        newmark_gamma_ =
            analysis_["scheme_settings"].at("gamma").template get<double>();
      // Read parameters of Newton-Raphson interation
      if (nonlinear_) {
        if (analysis_["scheme_settings"].contains("max_iteration"))
          max_iteration_ = analysis_["scheme_settings"]
                                .at("max_iteration")
                                .template get<unsigned>();
        if (analysis_["scheme_settings"].contains("displacement_tolerance"))
          displacement_tolerance = analysis_["scheme_settings"]
                                        .at("displacement_tolerance")
                                        .template get<double>();
        if (analysis_["scheme_settings"].contains("temperature_tolerance"))
          temperature_tolerance = analysis_["scheme_settings"]
                                        .at("temperature_tolerance")
                                        .template get<double>();
        if (analysis_["scheme_settings"].contains("residual_tolerance"))
          residual_tolerance = analysis_["scheme_settings"]
                                    .at("residual_tolerance")
                                    .template get<double>();
        if (analysis_["scheme_settings"].contains(
                "relative_residual_tolerance"))
          relative_residual_tolerance = analysis_["scheme_settings"]
                                            .at("relative_residual_tolerance")
                                            .template get<double>();
        if (analysis_["scheme_settings"].contains("verbosity"))
          verbosity_ = analysis_["scheme_settings"]
                            .at("verbosity")
                            .template get<unsigned>();
      }
    }

    // Initialise convergence criteria
    if (nonlinear_) {
      residual_criterion_ = std::make_shared<mpm::ConvergenceCriterionResidual>(
          relative_residual_tolerance, residual_tolerance, verbosity_);
      displacement_criterion_ = std::make_shared<mpm::ConvergenceCriterionSolution>(
          displacement_tolerance, verbosity_);
      temperature_criterion_ = std::make_shared<mpm::ConvergenceCriterionSolution>(
          temperature_tolerance, verbosity_);          
    }
  }
}

//! MPM Implicit solver
template <unsigned Tdim>
bool mpm::MPMImplicitThermal<Tdim>::solve() {
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

  // Virtual flux
  if (analysis_.find("virtual_flux") != analysis_.end()) {
    virtual_flux_ = analysis_["virtual_flux"]["virtual_flux"]
                                                .template get<bool>();
    std::string flux_type = analysis_["virtual_flux"]["flux_type"]
                                                .template get<std::string>();
    // Judge the flux type          
    if (flux_type_ == "convective") {
      vfm_param1_ = analysis_["virtual_flux"]["heat_transfer_coeff"]
                                                .template get<double>();
      vfm_param2_ = analysis_["virtual_flux"]["ambient_temperature"]
                                                .template get<double>();
    } else if (flux_type_ == "conductive") 
      vfm_param1_ = analysis_["virtual_flux"]["flux"].template get<double>();
    else         
      throw std::runtime_error(
          "Virtual heat flux boudary is not correctly assigned");
  }

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

  // Initialise loading conditions
  this->initialise_loads();

  // Write initial outputs
  if (!resume) this->write_outputs(this->step_);

  // Initialise matrix
  bool matrix_status = this->initialise_matrix();
  if (!matrix_status) {
    status = false;
    throw std::runtime_error("Initialisation of matrix failed");
  }

  auto solver_begin = std::chrono::steady_clock::now();
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

    // Inject particles
    mesh_->inject_particles(step_ * dt_);

    // Initialise nodes, cells and shape functions
    mpm_scheme_->initialise_thermal();

    // Mass momentum inertia and compute velocity and acceleration at nodes
    mpm_scheme_->compute_nodal_kinematics(velocity_update_, phase_);

    // Compute free surface cells, nodes, and particles
    if (free_surface_detection_ != "none") {
      mesh_->compute_free_surface(free_surface_detection_, fs_vol_tolerance_,
                                  cell_neighbourhood_);
    }

    mpm_scheme_->compute_nodal_temperatures(phase_, dt_, step_);

    // Predict nodal kinematics -- Predictor step of Newmark scheme
    mpm_scheme_->update_nodal_thermokinematics_newmark(phase_, newmark_beta_,
                                                      newmark_gamma_, step_);

    // Reinitialise system matrix to construct equillibrium equation
    bool matrix_reinitialization_status = this->reinitialise_matrix();
    if (!matrix_reinitialization_status) {
      status = false;
      throw std::runtime_error("Reinitialisation of matrix failed");
    }

    // Newton-Raphson iteration
    current_iteration_ = 0;

    // Assemble equilibrium equation
    this->assemble_system_equation();

    // Check convergence of residual
    bool convergence = false;
    if (nonlinear_)
      convergence = residual_criterion_->check_convergence(
          assembler_->global_residual_rhs_vector(), true);

    // Iterations
    while (!convergence && current_iteration_ < max_iteration_) {
      // Initialisation of Newton-Raphson iteration
      this->initialise_newton_raphson_iteration();

      // Solve equilibrium equation
      this->solve_system_equation();

      // Predict nodal kinematics -- Predictor step of Newmark scheme
      mpm_scheme_->update_nodal_thermokinematics_newmark(phase_, newmark_beta_,
                                                      newmark_gamma_, step_);

      // Update stress and strain
      mpm_scheme_->compute_stress_strain_thermal(phase_, pressure_smoothing_);

      // Check convergence of Newton-Raphson iteration
      if (nonlinear_) {
        convergence = this->check_newton_raphson_convergence();
      } else {
        convergence = true;
      }

      // Finalisation of Newton-Raphson iteration
      if (convergence || current_iteration_ == max_iteration_)
        this->finalise_newton_raphson_iteration();
      
    }

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
  console_->info("Rank {}, Implicit {} solver duration: {} ms", mpi_rank,
                 mpm_scheme_->scheme(),
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     solver_end - solver_begin)
                     .count());

  return status;
}

// Initialise matrix
template <unsigned Tdim>
bool mpm::MPMImplicitThermal<Tdim>::initialise_matrix() {
  bool status = true;
  try {
    // Get matrix assembler type
    std::string assembler_type = analysis_["linear_solver"]["assembler_type"]
                                     .template get<std::string>();
    // Create matrix assembler
    assembler_ =
        Factory<mpm::AssemblerBase<Tdim>, unsigned>::instance()->create(
            assembler_type, std::move(node_neighbourhood_));

    // Solver settings
    if (analysis_["linear_solver"].contains("solver_settings") &&
        analysis_["linear_solver"].at("solver_settings").is_array() &&
        analysis_["linear_solver"].at("solver_settings").size() > 0) {
      mpm::MPMBase<Tdim>::initialise_linear_solver(
          analysis_["linear_solver"]["solver_settings"], linear_solver_);
    }
    // Default solver settings
    else {
      std::string solver_type = "IterativeEigen";
      unsigned max_iter = 1000;
      double tolerance = 1.E-7;

      // In case the default settings are specified in json
      if (analysis_["linear_solver"].contains("solver_type")) {
        solver_type = analysis_["linear_solver"]["solver_type"]
                          .template get<std::string>();
      }
      // Max iteration steps
      if (analysis_["linear_solver"].contains("max_iter")) {
        max_iter =
            analysis_["linear_solver"]["max_iter"].template get<unsigned>();
      }
      // Tolerance
      if (analysis_["linear_solver"].contains("tolerance")) {
        tolerance =
            analysis_["linear_solver"]["tolerance"].template get<double>();
      }

      // NOTE: Only KrylovPETSC solver is supported for MPI
#ifdef USE_MPI
      // Get number of MPI ranks
      int mpi_size = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

      if (solver_type != "KrylovPETSC" && mpi_size > 1) {
        console_->warn(
            "The linear solver in MPI setting is automatically set to default: "
            "\'KrylovPETSC\'. Only \'KrylovPETSC\' solver is supported for "
            "MPI.");
        solver_type = "KrylovPETSC";
      }
#endif

      // Create matrix solver
      auto lin_solver =
          Factory<mpm::SolverBase<Eigen::SparseMatrix<double>>, unsigned,
                  double>::instance()
              ->create(solver_type, std::move(max_iter), std::move(tolerance));
      // Add solver set to map
      linear_solver_.insert(
          std::pair<
              std::string,
              std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>(
              "solution", lin_solver));
    }

    // Assign mesh pointer to assembler
    assembler_->assign_mesh_pointer(mesh_);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Reinitialise and resize matrices at the beginning of every time step
template <unsigned Tdim>
bool mpm::MPMImplicitThermal<Tdim>::reinitialise_matrix() {
  bool status = true;
  try {
    // Assigning matrix id (in each MPI rank)
    const auto nactive_node = mesh_->assign_active_nodes_id();

    // Assigning matrix id globally (required for rank-to-global mapping)
    unsigned nglobal_active_node = nactive_node;
#ifdef USE_MPI
    nglobal_active_node = mesh_->assign_global_active_nodes_id();
#endif

    // Assign global node indice
    assembler_->assign_global_node_indices(nactive_node, nglobal_active_node);

    // Assign displacement constraints
    assembler_->assign_displacement_constraints(this->step_ * this->dt_);

    // Assign temperature constraints
    assembler_->assign_temperature_constraints(this->step_ * this->dt_);

    // Initialise element matrix
    mesh_->iterate_over_cells(
        std::bind(&mpm::Cell<Tdim>::initialise_element_stiffness_matrix,
                  std::placeholders::_1));

    // Initialise element matrix
    mesh_->iterate_over_cells(
        std::bind(&mpm::Cell<Tdim>::initialise_element_thermal_matrix,
                  std::placeholders::_1));                  

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Reinitialise equilibrium equation
template <unsigned Tdim>
void mpm::MPMImplicitThermal<Tdim>::reinitialise_system_equation() {
  // Initialise element matrix
  mesh_->iterate_over_cells(
      std::bind(&mpm::Cell<Tdim>::initialise_element_stiffness_matrix,
                std::placeholders::_1));

  // Initialise nodal forces
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::initialise_force, std::placeholders::_1),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  // Initialise element matrix
  mesh_->iterate_over_cells(
      std::bind(&mpm::Cell<Tdim>::initialise_element_thermal_matrix,
                std::placeholders::_1));

  // Initialise nodal forces
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::initialise_heat, std::placeholders::_1),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
}

// Assemble equilibrium equation
template <unsigned Tdim>
bool mpm::MPMImplicitThermal<Tdim>::assemble_system_equation() {
  bool status = true;
  try {
    // Compute local cell stiffness matrices
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_stiffness_matrix_to_cell,
                  std::placeholders::_1, newmark_beta_, dt_, quasi_static_));

    // Assemble global stiffness matrix
    assembler_->assemble_stiffness_matrix();

    // Compute local residual force
    mpm_scheme_->compute_forces(gravity_, phase_, step_,
                                set_node_concentrated_force_, quasi_static_);

    // Assemble global residual force RHS vector
    assembler_->assemble_residual_force_right();

    // Apply displacement constraints
    assembler_->apply_displacement_constraints();

    // Compute local cell heat conduction matrices
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_heat_conductivity_to_cell,
                  std::placeholders::_1));

    // Compute local cell heat capacity matrices
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_heat_capacity_to_cell,
                  std::placeholders::_1, dt_, newmark_beta_, newmark_gamma_));

    // Assemble heat conduction matrix
    assembler_->assemble_thermal_conductivity_matrix();

    // Compute local cell heat conduction matrices
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::
                map_thermal_expansivity_to_cell, std::placeholders::_1));

    // Assemble thermal expansion matrix
    assembler_->assemble_thermal_expansivity_matrix();

    // Assemble global stiffness matrix
    assembler_->assemble_global_stiffness_matrix();

    // Compute local residual force
    mpm_scheme_->compute_heats(virtual_flux_, flux_type_, 
                                vfm_param1_, vfm_param2_);

    // Assemble global residual force RHS vector
    assembler_->assemble_residual_heat_right();

    // Apply temperature constraints
    assembler_->apply_temperature_increment_constraints();

    // Apply temperature and displacement coupling constraints
    assembler_->apply_coupling_constraints();

    // Assemble global residual force RHS vector
    assembler_->assemble_global_residual_right();

    // Assign rank global mapper to solver and convergence criteria (only
    // necessary at the initial iteration)
    if (current_iteration_ == 0) {
#ifdef USE_MPI
      // Assign global active dof to solver
      linear_solver_["solution"]->assign_global_active_dof(
          (Tdim +1) * assembler_->global_active_dof());

      // Prepare rank global mapper
      std::vector<int> system_rgm;
      for (unsigned dir = 0; dir < (Tdim + 1); ++dir) {
        auto dir_rgm = assembler_->rank_global_mapper();
        std::for_each(dir_rgm.begin(), dir_rgm.end(),
                      [size = assembler_->global_active_dof(),
                       dir = dir](int& rgm) { rgm += dir * size; });
        system_rgm.insert(system_rgm.end(), dir_rgm.begin(), dir_rgm.end());
      }
      // Assign rank global mapper to solver
      linear_solver_["solution"]->assign_rank_global_mapper(system_rgm);

      if (nonlinear_) {
        displacement_criterion_->assign_global_active_dof(
            (Tdim + 1) * assembler_->global_active_dof());
        residual_criterion_->assign_global_active_dof(
            (Tdim + 1) * assembler_->global_active_dof());
        displacement_criterion_->assign_rank_global_mapper(system_rgm);
        residual_criterion_->assign_rank_global_mapper(system_rgm);
      }
#endif
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Solve equilibrium equation
template <unsigned Tdim>
bool mpm::MPMImplicitThermal<Tdim>::solve_system_equation() {
  bool status = true;
  try {
    // Solve matrix equation and assign solution to assembler
    assembler_->assign_solution_increment(
        linear_solver_["solution"]->solve(
            assembler_->global_stiffness_matrix(),
            assembler_->global_residual_rhs_vector()));

    // Assign displacement increment to nodes
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::update_displacement_increment,
                  std::placeholders::_1, assembler_->displacement_increment(),
                  phase_, assembler_->active_dof()),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    // Assign temperature increment to nodes
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::update_temperature_increment,
                  std::placeholders::_1, assembler_->temperature_increment(),
                  phase_, dt_, step_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));        

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Initialisation of Newton-Raphson iteration
template <unsigned Tdim>
void mpm::MPMImplicitThermal<Tdim>::initialise_newton_raphson_iteration() {
  // Initialise MPI rank
  int mpi_rank = 0;
#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  // Advance iteration
  current_iteration_++;
  if (mpi_rank == 0 && verbosity_ > 0 && nonlinear_)
    console_->info("Newton-Raphson iteration: {} of {}.", current_iteration_,
                    max_iteration_);
}

// Check convergnece of Newton-Raphson iteration
template <unsigned Tdim>
bool mpm::MPMImplicitThermal<Tdim>::check_newton_raphson_convergence() {
  bool convergence;
  // Check convergence of solution (solution increment)
  convergence = (displacement_criterion_->check_convergence(
                          assembler_->displacement_increment()) &
                temperature_criterion_->check_convergence(
                          assembler_->temperature_increment()));


  if (!convergence) {
    // Reinitialise equilibrium equation
    this->reinitialise_system_equation();

    // Assemble equilibrium equation
    this->assemble_system_equation();

    // Check convergence of residual
    convergence = residual_criterion_->check_convergence(
        assembler_->global_residual_rhs_vector());
  }
  return convergence;
}

// finalisation of Newton-Raphson iteration
template <unsigned Tdim>
void mpm::MPMImplicitThermal<Tdim>::finalise_newton_raphson_iteration() {
  // Particle kinematics and volume
  mpm_scheme_->compute_particle_kinematics(velocity_update_, blending_ratio_,
                                            phase_, "Cundall", damping_factor_,
                                            step_);

  // Particle temperature
  mpm_scheme_->compute_particle_temperature();

  // Particle stress, strain and volume
  mpm_scheme_->update_particle_stress_strain_volume();
}