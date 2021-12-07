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
  } else {
    // Initialise particles
    this->initialise_particles();

    // Compute mass
    mesh_->iterate_over_particles(std::bind(
        &mpm::ParticleBase<Tdim>::compute_mass, std::placeholders::_1));

    // Domain decompose
    this->mpi_domain_decompose(initial_step);
  }

  // Initialise discontinuity
  this->initialise_discontinuity();

  // Create nodal properties for discontinuity
  // if (setdiscontinuity_) mesh_->create_nodal_properties_discontinuity();

  // Initialise loading conditions
  this->initialise_loads();

  bool surfacemesh_ = false;
  bool particle_levelset_ = false;
  bool propagation_ = false;

  // Initialise the levelset values for particles
  if (surfacemesh_) mesh_->initialise_levelset_discontinuity();

  auto solver_begin = std::chrono::steady_clock::now();

  for (; step_ < nsteps_; ++step_) {
    // FIXME: Modify this to a more generic function
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

    // FIXME: Remove this before merging to master
    // if (step_ == 0 || resume == true) {
    //   // predefine level set values
    //   mesh_->define_levelset();
    //   resume = false;
    // }

    if (initiation_) initiation_ = !mesh_->initiation_discontinuity();

    if (!initiation_) {  // setdiscontinuity_ &&
      // Initialise nodal properties
      mesh_->initialise_nodal_properties();

      // Initialise element properties
      mesh_->iterate_over_cells(std::bind(
          &mpm::Cell<Tdim>::initialise_element_properties_discontinuity,
          std::placeholders::_1));

      if (!particle_levelset_) {
        // locate points of discontinuity
        mesh_->locate_discontinuity();

        // Iterate over each points to compute shapefn
        mesh_->compute_shapefn_discontinuity();
      }

      // obtain nodal volume

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
      //   if (friction_coef_average_)
      //     mesh_->iterate_over_particles(
      //         std::bind(&mpm::ParticleBase<Tdim>::map_friction_coef_to_nodes,
      //                   std::placeholders::_1,
      //                   discontinuity_->friction_coef()));

      if (propagation_) {
        // find the potential tip element
        mesh_->iterate_over_cells(std::bind(
            &mpm::Cell<Tdim>::potential_tip_element, std::placeholders::_1));
      }
      //   if (particle_levelset_) {
      //     // determine the celltype by the nodal level set
      //     mesh_->iterate_over_cells(std::bind(&mpm::Cell<Tdim>::determine_crossed,
      //                                         std::placeholders::_1));
      //   }

      // obtain the normal direction of non-regular cell
      //   mesh_->compute_cell_normal_vector_discontinuity();

      //   mesh_->iterate_over_cells(std::bind(
      //       &mpm::Cell<Tdim>::compute_area_discontinuity,
      //       std::placeholders::_1));

      if (propagation_)
        // remove the spurious potential tip element
        mesh_->spurious_potential_tip_element();

      //   // assign_node_enrich
      //   mesh_->assign_node_enrich(friction_coef_average_, nodal_update);

      //   mesh_->check_particle_levelset(particle_levelset_);

      //   // obtain the normal direction of each cell and enrich nodes
      //   mesh_->compute_nodal_normal_vector_discontinuity();

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

    // if (particle_levelset_ || nodal_update) mesh_->update_node_enrich();

    // Update stress first
    mpm_scheme_->precompute_stress_strain(phase, pressure_smoothing_);

    // Iterate over each particle to calculate dudx
    // mesh_->iterate_over_particles(
    //     std::bind(&mpm::ParticleBase<Tdim>::compute_displacement_gradient,
    //               std::placeholders::_1, dt_));

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

    if (!initiation_) {  // setdiscontinuity_ &&

      if (propagation_) {
        // find the next tip element
        mesh_->next_tip_element_discontinuity();

        // discontinuity growth
        mesh_->update_discontinuity();
      }

      // Update the discontinuity position
      if (!particle_levelset_)
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
      // activate the initiation of the discontinuity
      if (discontinuity_props.contains("initiation"))
        initiation_ = discontinuity_props.at("initiation").template get<bool>();
      // maximum number of the discontinuity
      if (discontinuity_props.contains("maximum_num"))
        maximum_num_ =
            discontinuity_props.at("maximum_num").template get<int>();
      // shield width for searching the initiation
      if (discontinuity_props.contains("shield_width"))
        shield_width_ =
            discontinuity_props.at("shield_width").template get<double>();
      // maximum_pdstrain for searching the initiation
      if (discontinuity_props.contains("maximum_pdstrain"))
        maximum_pdstrain_ =
            discontinuity_props.at("maximum_pdstrain").template get<double>();
      auto json_generators = discontinuity_props["generator"];
      for (const auto& json_generator : json_generators) {
        // Get discontinuity type
        const std::string type =
            json_generator["type"].template get<std::string>();

        // Create a new discontinuity surface from JSON object
        auto discontinuity =
            Factory<mpm::DiscontinuityBase<Tdim>, const Json&>::instance()
                ->create(type, json_generator);
        if (discontinuity->description_type() ==
            mpm::DescriptionType::mark_points) {
          // initiate with the mesh file
          if (json_generator.contains("io_type") &&
              json_generator.contains("file")) {

            // Get discontinuity  input type
            auto io_type =
                json_generator["io_type"].template get<std::string>();

            // discontinuity file
            std::string discontinuity_file = io_->file_name(
                json_generator["file"].template get<std::string>());
            // Create a mesh reader
            auto discontunity_io =
                Factory<mpm::IOMesh<Tdim>>::instance()->create(io_type);

            // Create points and cells from file
            discontinuity->initialise(
                discontunity_io->read_mesh_nodes(discontinuity_file),
                discontunity_io->read_mesh_cells(discontinuity_file));
          }
        } else {
          // to do: read the mark points and directions
        }

        // insert
        discontinuity_ = discontinuity;
      }

      // Add discontinuity
      // discontinuity_ = discontinuity;
      // Copy discontinuity to mesh
      mesh_->initialise_discontinuity(this->discontinuity_);
    }
  } catch (std::exception& exception) {
    console_->warn("{} #{}: No discontinuity is defined", __FILE__, __LINE__,
                   exception.what());
  }
}
