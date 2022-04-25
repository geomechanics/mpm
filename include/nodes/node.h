#ifndef MPM_NODE_H_
#define MPM_NODE_H_

#include "logger.h"
#include "mutex.h"
#include "nodal_properties.h"
#include "node_base.h"

namespace mpm {

// Node base class
//! \brief Base class that stores the information about nodes
//! \details Node class: id_ and coordinates.
//! \tparam Tdim Dimension
//! \tparam Tdof Degrees of Freedom
//! \tparam Tnphases Number of phases
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
class Node : public NodeBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id, coordinates and dof
  //! \param[in] id Node id
  //! \param[in] coord coordinates of the node
  Node(Index id, const VectorDim& coord);

  //! Virtual destructor
  ~Node() override{};

  //! Delete copy constructor
  Node(const Node<Tdim, Tdof, Tnphases>&) = delete;

  //! Delete assignement operator
  Node& operator=(const Node<Tdim, Tdof, Tnphases>&) = delete;

  //! Initialise nodal properties
  void initialise() noexcept override;

  //! Return id of the nodebase
  Index id() const override { return id_; }

  //! Initialise shared pointer to nodal properties pool
  //! \param[in] prop_id Property id in the nodal property pool
  //! \param[in] nodal_properties Shared pointer to nodal properties pool
  void initialise_property_handle(
      unsigned prop_id,
      std::shared_ptr<mpm::NodalProperties> property_handle) noexcept override;

  //! Assign coordinates
  //! \param[in] coord Assign coord as coordinates of the nodebase
  void assign_coordinates(const VectorDim& coord) override {
    coordinates_ = coord;
  }

  //! Return coordinates
  //! \retval coordinates_ return coordinates of the nodebase
  VectorDim coordinates() const override { return coordinates_; }

  //! Return degrees of freedom
  unsigned dof() const override { return dof_; }

  //! Assign status
  void assign_status(bool status) override { status_ = status; }

  //! Return status
  bool status() const override { return status_; }

  //! Assign solving status
  void assign_solving_status(bool status) override { solving_status_ = status; }

  //! Return solving status
  bool solving_status() const override { return solving_status_; }

  //! Update mass at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] mass Mass from the particles in a cell
  void update_mass(bool update, unsigned phase, double mass) noexcept override;

  //! Return mass at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  double mass(unsigned phase) const override { return mass_(phase); }

  //! Update volume at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] volume Volume from the particles in a cell
  void update_volume(bool update, unsigned phase,
                     double volume) noexcept override;

  //! Return volume at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  double volume(unsigned phase) const override { return volume_(phase); }

  //! Assign concentrated force to the node
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] direction Index corresponding to the direction of traction
  //! \param[in] force Nodal concentrated force in specified direction
  //! \param[in] function math function
  //! \retval status Assignment status
  bool assign_concentrated_force(
      unsigned phase, unsigned direction, double force,
      const std::shared_ptr<FunctionBase>& function) override;

  //! Apply concentrated force to external force
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] current time
  void apply_concentrated_force(unsigned phase, double current_time) override;

  //! Update external force (body force / traction force)
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] force External force from the particles in a cell
  void update_external_force(bool update, unsigned phase,
                             const VectorDim& force) noexcept override;

  //! Return external force at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  VectorDim external_force(unsigned phase) const override {
    return external_force_.col(phase);
  }

  //! Update internal force (body force / traction force)
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] force Internal force from the particles in a cell
  void update_internal_force(bool update, unsigned phase,
                             const VectorDim& force) noexcept override;

  //! Return internal force at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  VectorDim internal_force(unsigned phase) const override {
    return internal_force_.col(phase);
  }

  //! Update pressure at the nodes from particle
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] mass_pressure Product of mass x pressure of a particle
  void update_mass_pressure(unsigned phase,
                            double mass_pressure) noexcept override;

  //! Assign pressure constraint
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] pressure Applied pressure constraint
  //! \param[in] function math function
  bool assign_pressure_constraint(
      unsigned phase, double pressure,
      const std::shared_ptr<FunctionBase>& function) override;

  //! Apply pressure constraint
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \param[in] step Step in analysis
  void apply_pressure_constraint(unsigned phase, double dt = 0,
                                 Index step = 0) noexcept override;

  //! Assign pressure at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] mass_pressure Product of mass x pressure of a particle
  void assign_pressure(unsigned phase, double mass_pressure) override;

  //! Return pressure at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  double pressure(unsigned phase) const override { return pressure_(phase); }

  //! Update momentum at the nodes
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] momentum Momentum from the particles in a cell
  void update_momentum(bool update, unsigned phase,
                       const VectorDim& momentum) noexcept override;

  //! Return momentum at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  VectorDim momentum(unsigned phase) const override {
    return momentum_.col(phase);
  }

  //! Compute velocity from the momentum
  void compute_velocity() override;

  //! Return velocity at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  VectorDim velocity(unsigned phase) const override {
    return velocity_.col(phase);
  }

  //! Update nodal acceleration
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] acceleration Acceleration from the particles in a cell
  void update_acceleration(bool update, unsigned phase,
                           const VectorDim& acceleration) noexcept override;

  //! Return acceleration at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  VectorDim acceleration(unsigned phase) const override {
    return acceleration_.col(phase);
  }

  //! Compute acceleration and velocity
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  bool compute_acceleration_velocity(unsigned phase,
                                     double dt) noexcept override;

  //! Compute acceleration and velocity with cundall damping factor
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \param[in] damping_factor Damping factor
  bool compute_acceleration_velocity_cundall(
      unsigned phase, double dt, double damping_factor) noexcept override;

  //! Assign velocity constraint
  //! Directions can take values between 0 and Dim * Nphases
  //! \param[in] dir Direction of velocity constraint
  //! \param[in] velocity Applied velocity constraint
  bool assign_velocity_constraint(unsigned dir, double velocity) override;

  //! Apply velocity constraints
  void apply_velocity_constraints() override;

  //! Assign friction constraint
  //! Directions can take values between 0 and Dim * Nphases
  //! \param[in] dir Direction of friction constraint
  //! \param[in] sign Sign of normal wrt coordinate system for friction
  //! \param[in] friction Applied friction constraint
  bool assign_friction_constraint(unsigned dir, int sign,
                                  double friction) override;

  //! Apply friction constraints
  //! \param[in] dt Time-step
  void apply_friction_constraints(double dt) override;

  //! Apply absorbing constraint
  //! \param[in] dir Direction of p-wave propagation in model
  //! \param[in] delta Virtual viscous layer thickness
  //! \param[in] h_min Characteristic length (cell height)
  //! \param[in] a Dimensionless dashpot weight factor, p-wave
  //! \param[in] b Dimensionless dashpot weight factor, s-wave
  //! \param[in] position Nodal position along boundary
  bool apply_absorbing_constraint(unsigned dir, double delta, double h_min,
                                  double a, double b,
                                  mpm::Position position) override;

  //! Assign rotation matrix
  //! \param[in] rotation_matrix Rotation matrix of the node
  void assign_rotation_matrix(
      const Eigen::Matrix<double, Tdim, Tdim>& rotation_matrix) override {
    rotation_matrix_ = rotation_matrix;
    generic_boundary_constraints_ = true;
  }

  //! Add material id from material points to list of materials in materials_
  //! \param[in] id Material id to be stored at the node
  void append_material_id(unsigned id) override;

  //! Return material ids in node
  std::set<unsigned> material_ids() const override { return material_ids_; }

  //! Assign MPI rank to node
  //! \param[in] rank MPI Rank of the node
  bool mpi_rank(unsigned rank) override;

  //! Assign MPI rank to node
  //! \param[in] rank MPI Rank of the node
  std::set<unsigned> mpi_ranks() const override { return mpi_ranks_; }

  //! Clear MPI rank
  void clear_mpi_ranks() override { mpi_ranks_.clear(); }

  //! Return ghost id
  Index ghost_id() const override { return ghost_id_; }

  //! Set ghost id
  void ghost_id(Index gid) override { ghost_id_ = gid; }

  //! Update nodal property at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] property Property name
  //! \param[in] property_value Property quantity from the particles in the cell
  //! \param[in] mat_id Id of the material within the property data
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  void update_property(bool update, const std::string& property,
                       const Eigen::MatrixXd& property_value, unsigned mat_id,
                       unsigned nprops) noexcept override;

  //! Compute multimaterial change in momentum
  void compute_multimaterial_change_in_momentum() override;

  //! Compute multimaterial separation vector
  void compute_multimaterial_separation_vector() override;

  //! Compute multimaterial normal unit vector
  void compute_multimaterial_normal_unit_vector() override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Initialise nodal properties for implicit solver
  //! \ingroup Impolicit
  void initialise_implicit() noexcept override;

  //! Initialise nodal forces
  //! \ingroup Impolicit
  void initialise_force() noexcept override;

  //! Update inertia at the nodes
  //! \ingroup Implicit
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] inertia Inertia from the particles in a cell
  void update_inertia(bool update, unsigned phase,
                      const VectorDim& inertia) noexcept override;

  //! Return inertia at a given node for a given phase
  //! \ingroup Implicit
  //! \param[in] phase Index corresponding to the phase
  VectorDim inertia(unsigned phase) const override {
    return inertia_.col(phase);
  }

  //! Compute velocity and acceleration from the momentum and inertia
  //! \ingroup Implicit
  void compute_velocity_acceleration() override;

  //! Return displacement at a given node for a given phase
  //! \ingroup Implicit
  //! \param[in] phase Index corresponding to the phase
  VectorDim displacement(unsigned phase) const override {
    return displacement_.col(phase);
  };

  //! Update velocity and acceleration by Newmark scheme
  //! \ingroup Implicit
  //! \param[in] newmark_beta Parameter beta of Newmark scheme
  //! \param[in] newmark_gamma Parameter gamma of Newmark scheme
  //! \param[in] dt Time-step
  void update_velocity_acceleration_newmark(unsigned phase, double newmark_beta,
                                            double newmark_gamma,
                                            double dt) override;

  //! Assign displacement constraint for implicit solver
  //! Directions can take values between 0 and Dim * Nphases
  //! \ingroup Implicit
  //! \param[in] dir Direction of displacement constraint
  //! \param[in] displacement Applied pressure constraint
  //! \param[in] function math function
  bool assign_displacement_constraint(
      const unsigned dir, const double displacement,
      const std::shared_ptr<FunctionBase>& function) override;

  //! Return displacement constraint
  //! \ingroup Implicit
  double displacement_constraint(const unsigned dir,
                                 const double current_time) const override {
    double constraint = std::numeric_limits<double>::max();
    if (displacement_constraints_.find(dir) !=
        displacement_constraints_.end()) {
      const double scalar =
          (displacement_function_.find(dir) != displacement_function_.end())
              ? displacement_function_.at(dir)->value(current_time)
              : 1.0;

      constraint = scalar * displacement_constraints_.at(dir);
    }
    return constraint;
  }

  //! Update displacement increment at the node
  //! \ingroup Implicit
  void update_displacement_increment(
      const Eigen::VectorXd& displacement_increment, unsigned phase,
      unsigned nactive_node) override;
  /**@{*/

  /**
   * \defgroup MultiPhase Functions dealing with multi-phase MPM
   */
  /**@{*/

  //! Return interpolated density at a given node for a given phase
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  double density(unsigned phase) const override { return density_(phase); }

  //! Compute nodal density
  //! \ingroup MultiPhase
  void compute_density() override;

  //! Assign free surface
  //! \ingroup MultiPhase
  void assign_free_surface(bool free_surface) override {
    node_mutex_.lock();
    free_surface_ = free_surface;
    node_mutex_.unlock();
  }

  //! Return free surface bool
  //! \ingroup MultiPhase
  bool free_surface() const override { return free_surface_; }

  //! Initialise two-phase nodal properties
  //! \ingroup MultiPhase
  void initialise_twophase() noexcept override;

  //! Update internal force (body force / traction force)
  //! \ingroup MultiPhase
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] drag_force Drag force from the particles in a cell
  //! \retval status Update status
  void update_drag_force_coefficient(bool update,
                                     const VectorDim& drag_force) override;

  //! Return drag force at a given node
  //! \ingroup MultiPhase
  VectorDim drag_force_coefficient() const override {
    return drag_force_coefficient_;
  }

  //! Compute acceleration and velocity for two phase
  //! \ingroup MultiPhase
  //! \param[in] dt Timestep in analysis
  bool compute_acceleration_velocity_twophase_explicit(
      double dt) noexcept override;

  //! Compute acceleration and velocity for two phase with cundall damping
  //! \ingroup MultiPhase
  //! \param[in] dt Timestep in analysis \param[in] damping_factor
  //! Damping factor
  bool compute_acceleration_velocity_twophase_explicit_cundall(
      double dt, double damping_factor) noexcept override;

  //! Compute semi-implicit acceleration and velocity
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \details Can be used for both semi-implicit navier-stokes and two-phase
  //! solvers
  //! \retval status Computation status
  bool compute_acceleration_velocity_semi_implicit_corrector(
      unsigned phase, double dt) override;

  //! Compute semi-implicit acceleration and velocity with Cundall damping
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \details Can be used for both semi-implicit navier-stokes and two-phase
  //! solvers
  //! \retval status Computation status
  bool compute_acceleration_velocity_semi_implicit_corrector_cundall(
      unsigned phase, double dt, double damping_factor) override;

  //! Assign active id
  //! \ingroup MultiPhase
  void assign_active_id(Index id) override { active_id_ = id; }

  //! Return active id
  //! \ingroup MultiPhase
  mpm::Index active_id() const override { return active_id_; }

  //! Assign global active id
  //! \ingroup MultiPhase
  void assign_global_active_id(Index id) override { global_active_id_ = id; }

  //! Return global active id
  //! \ingroup MultiPhase
  mpm::Index global_active_id() const override { return global_active_id_; }

  //! Return nodal pressure constraint
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] current_time current time of the analysis
  //! \retval pressure constraint at proper time for given phase
  double pressure_constraint(const unsigned phase,
                             const double current_time) const override {
    double constraint = std::numeric_limits<double>::max();
    if (pressure_constraints_.find(phase) != pressure_constraints_.end()) {
      const double scalar =
          (pressure_function_.find(phase) != pressure_function_.end())
              ? pressure_function_.at(phase)->value(current_time)
              : 1.0;

      constraint = scalar * pressure_constraints_.at(phase);
    }
    return constraint;
  }

  //! Update pressure increment at the node
  //! \ingroup MultiPhase
  void update_pressure_increment(const Eigen::VectorXd& pressure_increment,
                                 unsigned phase,
                                 double current_time = 0.) override;

  //! Return nodal pressure increment
  //! \ingroup MultiPhase
  double pressure_increment() const override { return pressure_increment_; }

  //! Return map of velocity constraints
  //! \ingroup MultiPhase
  std::map<unsigned, double>& velocity_constraints() override {
    return velocity_constraints_;
  }

  //! Update nodal intermediate velocity
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] acceleration_inter solved intermediate acceleration
  //! \param[in] dt Timestep in analysis
  void update_intermediate_acceleration_velocity(
      const unsigned phase, const Eigen::MatrixXd& acceleration_inter,
      double dt) override;

  //! Update correction force
  //! \ingroup MultiPhase
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] force Correction force from the particles in a cell
  void update_correction_force(bool update, unsigned phase,
                               const VectorDim& force) noexcept override;

  //! Return correction force at a given node for a given phase
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  VectorDim correction_force(unsigned phase) const override {
    return correction_force_.col(phase);
  }

  /**@}*/

  /**
   * \defgroup Nonlocal Functions dealing with MPM with nonlocal shape function
   */
  /**@{*/

  //! Function that initialise variables for nonlocal MPM
  //! \ingroup Nonlocal
  void initialise_nonlocal_node() noexcept override;

  //! Assign nodal nonlocal type
  //! \ingroup Nonlocal
  //! \param[in] dir Direction of node type
  //! \param[in] type Integer denoting the node type
  //! \brief: The list of node type is
  //! Regular = 0 (Default),
  //! LowerBoundary = 1,
  //! LowerIntermediate = 2,
  //! UpperIntermediate = 3
  //! UpperBoundary = 4
  void assign_nonlocal_node_type(unsigned dir, unsigned type) override {
    nonlocal_node_type_[dir] = type;
  }

  //! Function which return nodal nonlocal type vector
  //! \ingroup Nonlocal
  std::vector<unsigned> nonlocal_node_type() const override {
    return nonlocal_node_type_;
  }
  /**@}*/

 private:
  //! Mutex
  SpinMutex node_mutex_;
  //! nodebase id
  Index id_{std::numeric_limits<Index>::max()};
  //! nodal property id
  unsigned prop_id_{std::numeric_limits<unsigned>::max()};
  //! shared ghost id
  Index ghost_id_{std::numeric_limits<Index>::max()};
  //! nodal coordinates
  VectorDim coordinates_;
  //! Degrees of freedom
  unsigned dof_{std::numeric_limits<unsigned>::max()};
  //! Status
  bool status_{false};
  //! Solving status
  bool solving_status_{false};
  //! Mass
  Eigen::Matrix<double, 1, Tnphases> mass_;
  //! Volume
  Eigen::Matrix<double, 1, Tnphases> volume_;
  //! External force
  Eigen::Matrix<double, Tdim, Tnphases> external_force_;
  //! Internal force
  Eigen::Matrix<double, Tdim, Tnphases> internal_force_;
  //! Pressure
  Eigen::Matrix<double, 1, Tnphases> pressure_;
  //! Displacement
  VectorDim contact_displacement_;
  //! Velocity
  Eigen::Matrix<double, Tdim, Tnphases> velocity_;
  //! Momentum
  Eigen::Matrix<double, Tdim, Tnphases> momentum_;
  //! Acceleration
  Eigen::Matrix<double, Tdim, Tnphases> acceleration_;
  //! Velocity constraints
  std::map<unsigned, double> velocity_constraints_;
  //! Pressure constraint
  std::map<unsigned, double> pressure_constraints_;
  //! Rotation matrix for general velocity constraints
  Eigen::Matrix<double, Tdim, Tdim> rotation_matrix_;
  //! Material ids whose information was passed to this node
  std::set<unsigned> material_ids_;
  //! A general velocity (non-Cartesian/inclined) constraint is specified at the
  //! node
  bool generic_boundary_constraints_{false};
  //! Frictional constraints
  bool friction_{false};
  std::tuple<unsigned, int, double> friction_constraint_;
  //! Mathematical function for pressure
  std::map<unsigned, std::shared_ptr<FunctionBase>> pressure_function_;
  //! Concentrated force
  Eigen::Matrix<double, Tdim, Tnphases> concentrated_force_;
  //! Mathematical function for force
  std::shared_ptr<FunctionBase> force_function_{nullptr};
  //! Nodal property pool
  std::shared_ptr<mpm::NodalProperties> property_handle_{nullptr};
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! MPI ranks
  std::set<unsigned> mpi_ranks_;
  //! Global index for active node (in each rank)
  Index active_id_{std::numeric_limits<Index>::max()};
  //! Global index for active node (globally)
  Index global_active_id_{std::numeric_limits<Index>::max()};

  /**
   * \defgroup ImplicitVariables Variables dealing with implicit MPM
   */
  /**@{*/
  //! Inertia
  Eigen::Matrix<double, Tdim, Tnphases> inertia_;
  //! Displacement
  Eigen::Matrix<double, Tdim, Tnphases> displacement_;
  //! Displacement constraints
  std::map<unsigned, double> displacement_constraints_;
  //! Mathematical function for displacement
  std::map<unsigned, std::shared_ptr<FunctionBase>> displacement_function_;
  /**@}*/

  /**
   * \defgroup MultiPhaseVariables Variables for multi-phase MPM
   * @{
   */
  //! Free surface
  bool free_surface_{false};
  //! Signed distance
  double signed_distance_;
  //! Interpolated density
  Eigen::Matrix<double, 1, Tnphases> density_;
  //! p^(t+1) - beta * p^(t)
  double pressure_increment_;
  //! Correction force
  Eigen::Matrix<double, Tdim, Tnphases> correction_force_;
  //! Drag force
  Eigen::Matrix<double, Tdim, 1> drag_force_coefficient_;
  /**@}*/

  /**
   * \defgroup NonlocalMeshVariables Variables for nonlocal MPM
   * @{
   */
  //! Node type vector in each direction
  std::vector<unsigned> nonlocal_node_type_;
  /**@}*/
};  // Node class
}  // namespace mpm

#include "node.tcc"
#include "node_implicit.tcc"
#include "node_multiphase.tcc"

#endif  // MPM_NODE_H_
