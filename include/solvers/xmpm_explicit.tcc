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
  else
    mpm_scheme_ = std::make_shared<mpm::MPMSchemeUSF<Tdim>>(mesh_, dt_);

  //! Interface scheme
  if (this->interface_)
    contact_ = std::make_shared<mpm::ContactFriction<Tdim>>(mesh_);
  else
    contact_ = std::make_shared<mpm::Contact<Tdim>>(mesh_);
}

//! MPM Explicit compute stress strain
template <unsigned Tdim>
void mpm::XMPMExplicit<Tdim>::compute_stress_strain(unsigned phase) {
  // Iterate over each particle to calculate strain
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_strain, std::placeholders::_1, dt_));

  // Iterate over each particle to update particle volume
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::update_volume, std::placeholders::_1));

  // Pressure smoothing
  if (pressure_smoothing_) this->pressure_smoothing(phase);

  // Iterate over each particle to compute stress
  mesh_->iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::compute_stress, std::placeholders::_1));
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

  // Pressure smoothing
  pressure_smoothing_ = io_->analysis_bool("pressure_smoothing");

  // Interface
  interface_ = io_->analysis_bool("interface");

  // Initialise material
  this->initialise_materials();

  // Initialise mesh
  this->initialise_mesh();

  // Initialise particles
  if (!resume) this->initialise_particles();

  // Create nodal properties
  if (interface_) mesh_->create_nodal_properties();

  // Initialise discontinuity
  this->initialise_discontinuity();

  // Create nodal properties for discontinuity
  if (setdiscontinuity_) mesh_->create_nodal_properties_discontinuity();

  // Compute mass
  if (!resume)
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

  // Check point resume
  if (resume) {
    this->checkpoint_resume();
    --this->step_;
    mesh_->resume_domain_cell_ranks();
#ifdef USE_MPI
#ifdef USE_GRAPH_PARTITIONING
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
  } else {
    // Domain decompose
    bool initial_step = (resume == true) ? false : true;
    this->mpi_domain_decompose(initial_step);
  }

  //! Particle entity sets and velocity constraints
  if (resume) {
    this->particle_entity_sets(false);
    this->particle_velocity_constraints();
  }

  // Initialise loading conditions
  this->initialise_loads();

  // Initialise the levelset values for particles
  if (surfacemesh_) mesh_->initialise_levelset_discontinuity();

  auto solver_begin = std::chrono::steady_clock::now();

  if (!resume) {
    // Main loop
    // HDF5 outputs
    // HDF5 outputs the initial status
    this->write_hdf5(this->step_, this->nsteps_);
#ifdef USE_VTK
    // VTK outputs
    this->write_vtk(this->step_, this->nsteps_);
#endif
#ifdef USE_PARTIO
    // Partio outputs
    this->write_partio(this->step_, this->nsteps_);
#endif
  }

  for (; step_ < nsteps_; ++step_) {
    // to do
    bool nodal_update = false;

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

    if (step_ == 0 || resume == true) {
      // predefine level set values
      mesh_->define_levelset();
      resume = false;
    }
    // Initialise nodal properties and append material ids to node
    contact_->initialise();

    if (initiation_) initiation_ = !mesh_->initiation_discontinuity();

    if (setdiscontinuity_ && !initiation_) {
      // Initialise nodal properties
      mesh_->initialise_nodal_properties();
      // Initialise element properties
      mesh_->iterate_over_cells(std::bind(
          &mpm::Cell<Tdim>::initialise_element_properties_discontinuity,
          std::placeholders::_1));
      if (!particle_levelet_) {
        // locate points of discontinuity
        mesh_->locate_discontinuity();

        // Iterate over each points to compute shapefn
        mesh_->compute_shapefn_discontinuity();
      }

      //     // obtain nodal volume

      mesh_->iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_volume_to_nodes,
                    std::placeholders::_1));
      if (!nodal_update) {
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_levelset_to_nodes,
                      std::placeholders::_1));
      }

      // modify the nodal levelset_phi by mls
      if (nodal_levelset_ == "mls") mesh_->modify_nodal_levelset_mls();

      // obtain nodal frictional_coefficient
      if (friction_coef_average_)
        mesh_->iterate_over_particles(
            std::bind(&mpm::ParticleBase<Tdim>::map_friction_coef_to_nodes,
                      std::placeholders::_1, discontinuity_->friction_coef()));

      if (propagation_) {
        // find the potential tip element
        mesh_->iterate_over_cells(std::bind(
            &mpm::Cell<Tdim>::potential_tip_element, std::placeholders::_1));
      }
      if (particle_levelet_) {
        // determine the celltype by the nodal level set
        mesh_->iterate_over_cells(std::bind(&mpm::Cell<Tdim>::determine_crossed,
                                            std::placeholders::_1));
      }

      // obtain the normal direction of non-regular cell
      mesh_->compute_cell_normal_vector_discontinuity();

      mesh_->iterate_over_cells(std::bind(
          &mpm::Cell<Tdim>::compute_area_discontinuity, std::placeholders::_1));

      if (propagation_)
        // remove the spurious potential tip element
        mesh_->spurious_potential_tip_element();

      // assign_node_enrich
      mesh_->assign_node_enrich(friction_coef_average_, nodal_update);

      mesh_->check_particle_levelset(particle_levelet_);

      // obtain the normal direction of each cell and enrich nodes
      mesh_->compute_nodal_normal_vector_discontinuity();

      if (propagation_)  // find the tip element
      {
        mesh_->iterate_over_cells(
            std::bind(&mpm::Cell<Tdim>::tip_element, std::placeholders::_1));
      }

      // mesh_->output_celltype(step_);
      mesh_->selfcontact_detection();
    }

    // mesh_->iterate_over_particles(std::bind(
    //     &mpm::ParticleBase<Tdim>::check_levelset, std::placeholders::_1));

    // Mass momentum and compute velocity at nodes
    mpm_scheme_->compute_nodal_kinematics(phase);

    if (particle_levelet_ || nodal_update) mesh_->update_node_enrich();

    // Map material properties to nodes
    contact_->compute_contact_forces();

    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_);

    // Iterate over each particle to calculate dudx
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_dudx, std::placeholders::_1, dt_));

    // Compute forces
    mpm_scheme_->compute_forces(gravity_, phase, step_,
                                set_node_concentrated_force_);

    // integrate momentum by iterating over nodes
    if (damping_type_ == mpm::Damping::Cundall)
      mesh_->iterate_over_nodes_predicate(
          std::bind(
              &mpm::NodeBase<Tdim>::compute_momentum_discontinuity_cundall,
              std::placeholders::_1, phase, dt_, damping_factor_),
          std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));
    else
      mesh_->iterate_over_nodes_predicate(
          std::bind(&mpm::NodeBase<Tdim>::compute_momentum_discontinuity,
                    std::placeholders::_1, phase, this->dt_),
          std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

    if (setdiscontinuity_ && !initiation_) {

      if (propagation_) {
        // find the next tip element
        mesh_->next_tip_element_discontinuity();

        // discontinuity growth
        mesh_->update_discontinuity();
      }

      // Update the discontinuity position
      if (!particle_levelet_)
        mesh_->compute_updated_position_discontinuity(this->dt_);
    }

    // Iterate over each particle to compute updated position
    mesh_->iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::compute_updated_position,
                  std::placeholders::_1, dt_, velocity_update_));
    // Apply particle velocity constraints
    mesh_->apply_particle_velocity_constraints();

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

    if ((step_ + 1) % output_steps_ == 0) {
      // HDF5 outputs
      this->write_hdf5(this->step_ + 1, this->nsteps_);
      // to do
      mesh_->output_discontinuity(this->step_ + 1);
      // mesh_->output_force(step_);
#ifdef USE_VTK
      // VTK outputs
      this->write_vtk(this->step_ + 1, this->nsteps_);
#endif
#ifdef USE_PARTIO
      // Partio outputs
      this->write_partio(this->step_ + 1, this->nsteps_);
#endif
    }
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
      setdiscontinuity_ = true;
      // Get discontinuity type
      const std::string discontunity_type =
          discontinuity_props["type"].template get<std::string>();

      // Create a new discontinuity surface from JSON object
      auto discontinuity =
          Factory<mpm::DiscontinuityBase<Tdim>, const Json&>::instance()
              ->create(discontunity_type, discontinuity_props);

      if (discontinuity_props.contains("initiation"))
        initiation_ = discontinuity_props.at("initiation").template get<bool>();

      if (discontinuity_props.contains("nodal_levelset"))
        nodal_levelset_ = discontinuity_props.at("nodal_levelset")
                              .template get<std::string>();

      if (discontinuity_props.contains("friction_coefficient_average"))
        friction_coef_average_ =
            discontinuity_props.at("friction_coefficient_average")
                .template get<bool>();

      if (discontinuity_props.contains("propagation"))
        propagation_ =
            discontinuity_props.at("propagation").template get<bool>();

      if (discontinuity_props.contains("io_type") &&
          discontinuity_props.contains("file")) {

        surfacemesh_ = true;
        // particle_levelet_ = true;
        // Get discontinuity  input type
        auto io_type =
            discontinuity_props["io_type"].template get<std::string>();

        // discontinuity file
        std::string discontinuity_file = io_->file_name(
            discontinuity_props["file"].template get<std::string>());
        // Create a mesh reader
        auto discontunity_io =
            Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

        // Create points and cells from file
        discontinuity->initialise(
            discontunity_io->read_mesh_nodes(discontinuity_file),
            discontunity_io->read_mesh_cells(discontinuity_file));
      } else if (discontinuity_props.contains("particle_levelset")) {
        particle_levelet_ =
            discontinuity_props.at("particle_levelset").template get<bool>();
      }

      // Add discontinuity
      discontinuity_ = discontinuity;
      // Copy discontinuity to mesh
      mesh_->initialise_discontinuity(this->discontinuity_);
    }
  } catch (std::exception& exception) {
    console_->warn("{} #{}: No discontinuity is defined", __FILE__, __LINE__,
                   exception.what());
  }
}

//! Checkpoint resume
template <unsigned Tdim>
bool mpm::XMPMExplicit<Tdim>::checkpoint_resume() {
  bool checkpoint = true;
  try {
    // TODO: Set phase
    const unsigned phase = 0;

    int mpi_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

    if (!analysis_["resume"]["resume"].template get<bool>())
      throw std::runtime_error("Resume analysis option is disabled!");

    // Get unique analysis id
    this->uuid_ = analysis_["resume"]["uuid"].template get<std::string>();
    // Get step
    this->step_ = analysis_["resume"]["step"].template get<mpm::Index>();

    // Input particle h5 file for resume
    std::string attribute = "particles";
    std::string extension = ".h5";

    auto particles_file =
        io_->output_file(attribute, extension, uuid_, step_, this->nsteps_)
            .string();

    // Load particle information from file
    const std::string particle_type = (Tdim == 2) ? "P2DXMPM" : "P3DXMPM";
    mesh_->read_particles_hdf5(particles_file, particle_type);

    // Clear all particle ids
    mesh_->iterate_over_cells(
        std::bind(&mpm::Cell<Tdim>::clear_particle_ids, std::placeholders::_1));

    // Locate particles
    auto unlocatable_particles = mesh_->locate_particles_mesh();

    if (!unlocatable_particles.empty())
      throw std::runtime_error("Particle outside the mesh domain");

    // Increament step
    ++this->step_;
    console_->info("Checkpoint resume at step {} of {}", this->step_,
                   this->nsteps_);

  } catch (std::exception& exception) {
    console_->info("{} {} Resume failed, restarting analysis: {}", __FILE__,
                   __LINE__, exception.what());
    this->step_ = 0;
    checkpoint = false;
  }
  return checkpoint;
}
