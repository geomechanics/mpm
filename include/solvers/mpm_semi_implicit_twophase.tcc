//! Constructor
template <unsigned Tdim>
mpm::MPMSemiImplicitTwoPhase<Tdim>::MPMSemiImplicitTwoPhase(
    const std::shared_ptr<IO>& io)
    : mpm::MPMBase<Tdim>(io) {
  //! Logger
  console_ = spdlog::get("MPMSemiImplicitTwoPhase");
}

//! MPM Semi-implicit TwoPhase compute stress strain
template <unsigned Tdim>
void mpm::MPMSemiImplicitTwoPhase<Tdim>::compute_stress_strain() {
  // Iterate over each particle to calculate strain of soil_skeleton
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_strain, std::placeholders::_1, dt_));
  // Iterate over each particle to update particle volume
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::update_volume, std::placeholders::_1));
  // Iterate over each particle to update porosity
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::update_porosity, std::placeholders::_1, dt_));
  // Iterate over each particle to compute stress of soil skeleton
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_stress, std::placeholders::_1));
  // Pressure smoothing
  if (pressure_smoothing_) this->pressure_smoothing(mpm::ParticlePhase::Solid);
  // Pore pressure smoothing
  if (pore_pressure_smoothing_) {
    this->pressure_smoothing(mpm::ParticlePhase::Liquid);
  }
}

//! MPM semi-implicit two-phase solver
template <unsigned Tdim>
bool mpm::MPMSemiImplicitTwoPhase<Tdim>::solve() {
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
  if (analysis_.find("pressure_smoothing") != analysis_.end())
    pressure_smoothing_ = analysis_["pressure_smoothing"].template get<bool>();

  // Pore pressure smoothing
  if (analysis_.find("pore_pressure_smoothing") != analysis_.end())
    pore_pressure_smoothing_ =
        analysis_["pore_pressure_smoothing"].template get<bool>();

  // Read settings for two-phase analysis
  if (analysis_.contains("scheme_settings")) {
    // Parameter to determine full and incremental projection
    if (analysis_["scheme_settings"].contains("beta"))
      beta_ = analysis_["scheme_settings"]["beta"].template get<double>();
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

  // Check point resume
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

    // Assign porosity
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::assign_porosity, std::placeholders::_1));

    // Assign permeability
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::assign_permeability, std::placeholders::_1));

    // Compute mass for each phase
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

    // Domain decompose
    this->mpi_domain_decompose(initial_step);
  }

  // Initialise loading conditions
  this->initialise_loads();

  // Initialise matrix
  bool matrix_status = this->initialise_matrix();
  if (!matrix_status) {
    status = false;
    throw std::runtime_error("Initialisation of matrix failed");
  }

  // Assign beta to each particle
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::assign_projection_parameter,
                std::placeholders::_1, beta_));

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

#pragma omp parallel sections
    {
      // Spawn a task for initialising nodes and cells
#pragma omp section
      {
        // Initialise nodes
        mesh_->iterate_over_nodes(std::bind(
            &mpm::NodeBase<Tdim>::initialise_twophase, std::placeholders::_1));

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
                  std::placeholders::_1));

#ifdef USE_MPI
    // Run if there is more than a single MPI task
    if (mpi_size > 1) {
      // MPI all reduce nodal mass for solid phase
      mesh_->template nodal_halo_exchange<double, 1>(
          std::bind(&mpm::NodeBase<Tdim>::mass, std::placeholders::_1,
                    mpm::NodePhase::NSolid),
          std::bind(&mpm::NodeBase<Tdim>::update_mass, std::placeholders::_1,
                    false, mpm::NodePhase::NSolid, std::placeholders::_2));
      // MPI all reduce nodal momentum for solid phase
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::momentum, std::placeholders::_1,
                    mpm::NodePhase::NSolid),
          std::bind(&mpm::NodeBase<Tdim>::update_momentum,
                    std::placeholders::_1, false, mpm::NodePhase::NSolid,
                    std::placeholders::_2));

      // MPI all reduce nodal mass for liquid phase
      mesh_->template nodal_halo_exchange<double, 1>(
          std::bind(&mpm::NodeBase<Tdim>::mass, std::placeholders::_1,
                    mpm::NodePhase::NLiquid),
          std::bind(&mpm::NodeBase<Tdim>::update_mass, std::placeholders::_1,
                    false, mpm::NodePhase::NLiquid, std::placeholders::_2));
      // MPI all reduce nodal momentum for liquid phase
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::momentum, std::placeholders::_1,
                    mpm::NodePhase::NLiquid),
          std::bind(&mpm::NodeBase<Tdim>::update_momentum,
                    std::placeholders::_1, false, mpm::NodePhase::NLiquid,
                    std::placeholders::_2));
    }
#endif

    // Compute free surface cells, nodes, and particles
    mesh_->compute_free_surface(free_surface_detection_, volume_tolerance_);

    // Spawn a task for initializing pressure at free surface
#pragma omp parallel sections
    {
#pragma omp section
      {
        // Assign initial pressure for all free-surface particle
        mesh_->iterate_over_particles_predicate(
            std::bind(&mpm::ParticleBase<Tdim>::assign_pressure,
                      std::placeholders::_1, 0.0, mpm::ParticlePhase::Liquid),
            std::bind(&mpm::ParticleBase<Tdim>::free_surface,
                      std::placeholders::_1));
      }
    }  // Wait to complete

    // Compute nodal velocity at the begining of time step
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::compute_velocity,
                  std::placeholders::_1),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    // Update stress first
    if (this->stress_update_ == "usf") this->compute_stress_strain();

      // Spawn a task for external force
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
      {
        // Spawn a task for internal force
        // Iterate over each particle to compute nodal internal force
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_internal_force,
                      std::placeholders::_1));
      }

#pragma omp section
      {
        // Iterate over particles to compute nodal drag force coefficient
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_drag_force_coefficient,
                      std::placeholders::_1));
      }
    }  // Wait for tasks to finish

    // Reinitialise system matrices to solve predictor equation and PPE
    bool matrix_reinitialization_status = this->reinitialise_matrix();
    if (!matrix_reinitialization_status) {
      status = false;
      throw std::runtime_error("Reinitialisation of matrix failed");
    }

    // Compute intermediate acceleration and velocity
    this->compute_intermediate_acceleration_velocity();

    // Compute poisson equation
    this->compute_poisson_equation();

    // Assign pressure to nodes
    mesh_->iterate_over_nodes_predicate(
        std::bind(&mpm::NodeBase<Tdim>::update_pressure_increment,
                  std::placeholders::_1, assembler_->pressure_increment(),
                  mpm::NodePhase::NLiquid, this->step_ * this->dt_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    // Use nodal pressure to update particle pressure
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::compute_updated_pressure,
                  std::placeholders::_1));

    // Compute correction force
    this->compute_correction_force();

#ifdef USE_MPI
    // Run if there is more than a single MPI task
    if (mpi_size > 1) {
      // MPI all reduce correction force for solid phase
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::correction_force,
                    std::placeholders::_1, mpm::NodePhase::NSolid),
          std::bind(&mpm::NodeBase<Tdim>::update_correction_force,
                    std::placeholders::_1, false, mpm::NodePhase::NSolid,
                    std::placeholders::_2));
      // MPI all reduce correction force for liquid phase
      mesh_->template nodal_halo_exchange<Eigen::Matrix<double, Tdim, 1>, Tdim>(
          std::bind(&mpm::NodeBase<Tdim>::correction_force,
                    std::placeholders::_1, mpm::NodePhase::NLiquid),
          std::bind(&mpm::NodeBase<Tdim>::update_correction_force,
                    std::placeholders::_1, false, mpm::NodePhase::NLiquid,
                    std::placeholders::_2));
    }
#endif

    if (damping_type_ == mpm::Damping::Cundall) {
      // Iterate over active nodes to compute acceleratation and velocity
      mesh_->iterate_over_nodes_predicate(
          std::bind(
              &mpm::NodeBase<Tdim>::
                  compute_acceleration_velocity_semi_implicit_corrector_cundall,
              std::placeholders::_1, mpm::NodePhase::NSolid, this->dt_,
              damping_factor_),
          std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

      mesh_->iterate_over_nodes_predicate(
          std::bind(
              &mpm::NodeBase<Tdim>::
                  compute_acceleration_velocity_semi_implicit_corrector_cundall,
              std::placeholders::_1, mpm::NodePhase::NLiquid, this->dt_,
              damping_factor_),
          std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
    } else {
      // Iterate over active nodes to compute acceleratation and velocity
      mesh_->iterate_over_nodes_predicate(
          std::bind(
              &mpm::NodeBase<
                  Tdim>::compute_acceleration_velocity_semi_implicit_corrector,
              std::placeholders::_1, mpm::NodePhase::NSolid, this->dt_),
          std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

      mesh_->iterate_over_nodes_predicate(
          std::bind(
              &mpm::NodeBase<
                  Tdim>::compute_acceleration_velocity_semi_implicit_corrector,
              std::placeholders::_1, mpm::NodePhase::NLiquid, this->dt_),
          std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
    }

    // Update particle position and kinematics
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::compute_updated_position,
                  std::placeholders::_1, this->dt_, velocity_update_));

    // Apply particle velocity constraints
    mesh_->apply_particle_velocity_constraints();

    // Update stress first
    if (this->stress_update_ == "usl") this->compute_stress_strain();

    // Locate particle
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
  console_->info("Rank {}, SemiImplicit TwoPhase solver duration: {} ms",
                 mpi_rank,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     solver_end - solver_begin)
                     .count());

  return status;
}

// Semi-implicit functions
// Initialise matrix
template <unsigned Tdim>
bool mpm::MPMSemiImplicitTwoPhase<Tdim>::initialise_matrix() {
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
      std::string sub_solver_type_pressure = "cg";
      std::string sub_solver_type_acc = "lscg";

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
            "The linear solver in MPI setting is automatically set to "
            "default: "
            "\'KrylovPETSC\'. Only \'KrylovPETSC\' solver is supported for "
            "MPI.");
        solver_type = "KrylovPETSC";
      }
#endif

      // If "KrylovPETSC" is selected, modify lscg to lsqr for solving acc
      if (solver_type == "KrylovPETSC") sub_solver_type_acc = "lsqr";

      // Create matrix solver for pressure
      auto press_lin_solver =
          Factory<mpm::SolverBase<Eigen::SparseMatrix<double>>, unsigned,
                  double>::instance()
              ->create(solver_type, std::move(max_iter), std::move(tolerance));
      press_lin_solver->set_sub_solver_type(sub_solver_type_pressure);
      linear_solver_.insert(
          std::pair<
              std::string,
              std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>(
              "pressure", press_lin_solver));

      // Create matrix solver for acceleration
      auto acc_lin_solver =
          Factory<mpm::SolverBase<Eigen::SparseMatrix<double>>, unsigned,
                  double>::instance()
              ->create(solver_type, std::move(max_iter), std::move(tolerance));
      acc_lin_solver->set_sub_solver_type(sub_solver_type_acc);
      linear_solver_.insert(
          std::pair<
              std::string,
              std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>(
              "acceleration", acc_lin_solver));
    }

    // Assign mesh pointer to assembler
    assembler_->assign_mesh_pointer(mesh_);

    // Get method to detect free surface detection
    free_surface_detection_ = "density";
    if (analysis_["free_surface_detection"].contains("type"))
      free_surface_detection_ = analysis_["free_surface_detection"]["type"]
                                    .template get<std::string>();
    // Get volume tolerance for free surface
    volume_tolerance_ = analysis_["free_surface_detection"]["volume_tolerance"]
                            .template get<double>();

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Reinitialise and resize matrices at the beginning of every time step
template <unsigned Tdim>
bool mpm::MPMSemiImplicitTwoPhase<Tdim>::reinitialise_matrix() {

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

    // Assign pressure constraints
    assembler_->assign_pressure_constraints(this->beta_,
                                            this->step_ * this->dt_);

    // Assign velocity constraints
    assembler_->assign_velocity_constraints();

    // Initialise element matrix
    mesh_->iterate_over_cells(
        std::bind(&mpm::Cell<Tdim>::initialise_element_matrix_twophase,
                  std::placeholders::_1));

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Compute intermediate velocity
template <unsigned Tdim>
bool mpm::MPMSemiImplicitTwoPhase<
    Tdim>::compute_intermediate_acceleration_velocity() {
  bool status = true;
  try {
    // Map coupling drag matrix to cell
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_drag_matrix_to_cell,
                  std::placeholders::_1));

    // Assemble coefficient matrix LHS for each direction
    assembler_->assemble_predictor_left(dt_);

    // Assemble predictor RHS force vector
    assembler_->assemble_predictor_right(dt_);

    // Apply velocity constraints to predictor LHS and RHS
    assembler_->apply_velocity_constraints();

#ifdef USE_MPI
    // Assign global active dof to solver
    linear_solver_["acceleration"]->assign_global_active_dof(
        2 * assembler_->global_active_dof());

    // Prepare rank global mapper to compute predictor equation
    auto predictor_rgm = assembler_->rank_global_mapper();
    auto predictor_rgm_liquid = assembler_->rank_global_mapper();
    std::for_each(
        predictor_rgm_liquid.begin(), predictor_rgm_liquid.end(),
        [size = assembler_->global_active_dof()](int& rgm) { rgm += size; });
    predictor_rgm.insert(predictor_rgm.end(), predictor_rgm_liquid.begin(),
                         predictor_rgm_liquid.end());

    // Assign rank global mapper to solver
    linear_solver_["acceleration"]->assign_rank_global_mapper(predictor_rgm);
#endif

    // Compute matrix equation of each direction
    for (unsigned dir = 0; dir < Tdim; ++dir) {
      // Solve equation 1 to compute intermediate acceleration
      assembler_->assign_intermediate_acceleration(
          dir, linear_solver_["acceleration"]->solve(
                   assembler_->predictor_lhs_matrix(dir),
                   assembler_->predictor_rhs_vector().col(dir)));
    }

    // Update intermediate acceleration and velocity of solid phase
    mesh_->iterate_over_nodes_predicate(
        std::bind(
            &mpm::NodeBase<Tdim>::update_intermediate_acceleration_velocity,
            std::placeholders::_1, mpm::NodePhase::NSolid,
            assembler_->intermediate_acceleration().topRows(
                assembler_->active_dof()),
            this->dt_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    // Update intermediate acceleration and velocity of water phase
    mesh_->iterate_over_nodes_predicate(
        std::bind(
            &mpm::NodeBase<Tdim>::update_intermediate_acceleration_velocity,
            std::placeholders::_1, mpm::NodePhase::NLiquid,
            assembler_->intermediate_acceleration().bottomRows(
                assembler_->active_dof()),
            this->dt_),
        std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute poisson equation
template <unsigned Tdim>
bool mpm::MPMSemiImplicitTwoPhase<Tdim>::compute_poisson_equation() {
  bool status = true;
  try {
    // Construct local cell laplacian matrix
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_laplacian_to_cell,
                  std::placeholders::_1));

    // Assemble global laplacian matrix
    assembler_->assemble_laplacian_matrix(dt_);

    // Map Poisson RHS matrix
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_poisson_right_to_cell,
                  std::placeholders::_1));

    // Assemble poisson RHS vector
    assembler_->assemble_poisson_right(dt_);

    // Assign free surface to assembler
    assembler_->assign_free_surface(mesh_->free_surface_nodes());

    // Apply constraints
    assembler_->apply_pressure_constraints();

#ifdef USE_MPI
    // Assign global active dof to solver
    linear_solver_["pressure"]->assign_global_active_dof(
        assembler_->global_active_dof());

    // Assign rank global mapper to solver
    linear_solver_["pressure"]->assign_rank_global_mapper(
        assembler_->rank_global_mapper());
#endif

    // Solve matrix equation and assign solution to assembler
    assembler_->assign_pressure_increment(linear_solver_["pressure"]->solve(
        assembler_->laplacian_matrix(), assembler_->poisson_rhs_vector()));

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Compute correction force
template <unsigned Tdim>
bool mpm::MPMSemiImplicitTwoPhase<Tdim>::compute_correction_force() {
  bool status = true;
  try {
    // Map correction matrix from particles to cell
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_correction_matrix_to_cell,
                  std::placeholders::_1));

    // Assemble correction matrix
    assembler_->assemble_corrector_right(dt_);

    // Assign correction force
    mesh_->compute_nodal_correction_force_twophase(
        assembler_->correction_matrix(), assembler_->pressure_increment(), dt_);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}
