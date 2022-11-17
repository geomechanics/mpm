//! Constructor
template <unsigned Tdim>
mpm::XMPMExplicit<Tdim>::XMPMExplicit(const std::shared_ptr<IO>& io)
    : mpm::MPMBase<Tdim>(io) {
  this->xmpm_ = true;
  //! Logger
  console_ = spdlog::get("XMPMExplicit");
  //! Stress update
  if (this->stress_update_ == "usl")
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSL<Tdim>>(mesh_, dt_);
  else if (this->stress_update_ == "musl")
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeMUSL<Tdim>>(mesh_, dt_);
  else
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSF<Tdim>>(mesh_, dt_);

  this->mpm_scheme_->assign_xmpm(true);
}

//! MPM Explicit solver
template <unsigned Tdim>
bool mpm::XMPMExplicit<Tdim>::solve() {
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

    // Initialise discontinuity
    this->initialise_discontinuity();
  } else {
    // Initialise particles
    this->initialise_particles();

    // Initialise discontinuity
    this->initialise_discontinuity();

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

    // Initialise nodes, cells and shape functions
    mpm_scheme_->initialise();

    // Initiation detection of discontinuity
    if (initiation_) {
      if (mesh_->initiation_discontinuity(maximum_pdstrain_, shield_width_,
                                          maximum_num_, initiation_property_))
        // Initialise nodes, cells and shape functions
        mpm_scheme_->initialise();

      if (mesh_->discontinuity_num() >= maximum_num_) initiation_ = false;
    }

    // the pre-process for the discontinuity propagation
    mesh_->preprocess_discontinuity();

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->compute_nodal_kinematics(phase);

    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_);

    // Compute forces
    mpm_scheme_->compute_forces(gravity_, phase, step_,
                                set_node_concentrated_force_);

    // the post-process for the discontinuity propagation
    mesh_->postprocess_discontinuity();

    // Particle kinematics
    mpm_scheme_->compute_particle_kinematics(velocity_update_, phase, "Cundall",
                                             damping_factor_, step_, false);

    // Mass momentum and compute velocity at nodes for musl
    mpm_scheme_->postcompute_nodal_kinematics(phase);

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

// Initialise discontinuity
template <unsigned Tdim>
void mpm::XMPMExplicit<Tdim>::initialise_discontinuity() {
  try {
    // Get discontinuity data
    auto discontinuity_props = io_->json_object("discontinuity");
    if (!discontinuity_props.empty()) {
      // activate the initiation of the discontinuity
      if (discontinuity_props.contains("initiation"))
        initiation_ = discontinuity_props.at("initiation").template get<bool>();

      if (initiation_) {
        // maximum number of the discontinuity
        if (discontinuity_props.contains("maximum_num"))
          maximum_num_ =
              discontinuity_props.at("maximum_num").template get<int>();
        // shield width to determine the discontinuity search region
        if (discontinuity_props.contains("shield_width"))
          shield_width_ =
              discontinuity_props.at("shield_width").template get<double>();
        // maximum_pdstrain for searching the initiation
        if (discontinuity_props.contains("maximum_pdstrain"))
          maximum_pdstrain_ =
              discontinuity_props.at("maximum_pdstrain").template get<double>();

        //! store the properties fot each newly generated discontinuity:
        //! cohesion, friction_coef, contact_distance, width, move_direction,
        //! friction_coef_average
        double cohesion = 0;
        double friction_coef = 0;
        double contact_distance = 0;
        double width = std::numeric_limits<double>::max();
        int move_direction = 1;
        bool friction_coef_average = true;
        bool mls = false;

        // store friction coefficient if it's given in input file
        if (discontinuity_props.contains("friction_coefficient")) {
          friction_coef_average = false;
          friction_coef = discontinuity_props.at("friction_coefficient")
                              .template get<double>();
        }

        // store cohesion if it's given in input file
        if (discontinuity_props.contains("cohesion"))
          cohesion = discontinuity_props.at("cohesion").template get<double>();

        // store contact_distance if it's given in input file
        if (discontinuity_props.contains("contact_distance"))
          contact_distance =
              discontinuity_props.at("contact_distance").template get<double>();

        // store move direction if it's given in input file
        if (discontinuity_props.contains("move_direction"))
          move_direction =
              discontinuity_props.at("move_direction").template get<int>();

        // store width if it's given in input file
        if (discontinuity_props.contains("width"))
          width = discontinuity_props.at("width").template get<double>();

        // store mls if it's given in input file
        if (discontinuity_props.contains("mls"))
          mls = discontinuity_props.at("mls").template get<bool>();

        // Discontinuity initiation property
        initiation_property_ = std::make_tuple(
            cohesion, friction_coef, contact_distance, width, maximum_pdstrain_,
            move_direction, friction_coef_average, mls);
      }

      // generate predefined discontinuity
      auto json_generators = discontinuity_props["generator"];

      for (const auto& json_generator : json_generators) {

        // dis_id following the order: 0, 1, 2 ...
        int dis_id = json_generator["id"].template get<int>();

        // Get discontinuity type
        const std::string type =
            json_generator["type"].template get<std::string>();

        // Create a new discontinuity surface from JSON object
        auto discontinuity = Factory<mpm::DiscontinuityBase<Tdim>, const Json&,
                                     unsigned>::instance()
                                 ->create(type, json_generator, dis_id);

        // Get discontinuity input type
        auto io_type = json_generator["io_type"].template get<std::string>();
        // Create a mesh reader
        auto discontinuity_io =
            Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

        // discontinuity file
        std::string discontinuity_file =
            io_->file_name(json_generator["file"].template get<std::string>());

        if (discontinuity->description_type() == "mark_points") {
          // Create points and cells from file
          discontinuity->initialise(
              discontinuity_io->read_mesh_nodes(discontinuity_file),
              discontinuity_io->read_mesh_cells(discontinuity_file));

          mesh_->insert_discontinuity(discontinuity);

          // initialise particle level set values by the discontinuity mesh
          mesh_->initialise_levelset_discontinuity(dis_id);

        } else if (discontinuity->description_type() == "particle_levelset") {
          mesh_->insert_discontinuity(discontinuity);
          if (!discontinuity_file.empty()) {
            mesh_->assign_particles_levelset(
                discontinuity_io->read_id_levelset(discontinuity_file), dis_id);
          }
        }
        // TODO: Check node levelset for predetermined discontinuities
        else if (discontinuity->description_type() == "node_levelset") {
          mesh_->insert_discontinuity(discontinuity);
          if (!discontinuity_file.empty()) {
            mesh_->assign_nodes_levelset(
                discontinuity_io->read_id_levelset(discontinuity_file), dis_id);
          }
        }
      }
    }
  } catch (std::exception& exception) {
    console_->warn("{} #{}: No discontinuity is defined", __FILE__, __LINE__,
                   exception.what());
  }
}

//! Output results
template <unsigned Tdim>
void mpm::XMPMExplicit<Tdim>::write_outputs(mpm::Index step) {
  mpm::MPMBase<Tdim>::write_outputs(step);
  if (step % this->output_steps_ == 0) {
    const auto& points = mesh_->discontinuity_points();
#ifdef USE_VTK
    // VTK outputs
    this->write_point_vtk(step, this->nsteps_, points);
#endif
  }
}
