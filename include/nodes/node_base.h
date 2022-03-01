#ifndef MPM_NODE_BASE_H_
#define MPM_NODE_BASE_H_

#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "data_types.h"
#include "function_base.h"
#include "nodal_properties.h"

namespace mpm {

//! Particle phases
enum NodePhase : unsigned int {
  NSolid = 0,
  NLiquid = 1,
  NGas = 2,
  NMixture = 0,
  NSinglePhase = 0
};

//! NodeBase base class for nodes
//! \brief Base class that stores the information about node_bases
//! \details NodeBase class: id_ and coordinates.
//! \tparam Tdim Dimension
template <unsigned Tdim>
class NodeBase {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  // Constructor with id and coordinates
  //! \param[in] id assign as the id_ of the node
  //! \param[in] coords coordinates of the node
  NodeBase(mpm::Index id, const VectorDim& coords){};

  //! Destructor
  virtual ~NodeBase(){};

  //! Delete copy constructor
  NodeBase(const NodeBase<Tdim>&) = delete;

  //! Delete assignement operator
  NodeBase& operator=(const NodeBase<Tdim>&) = delete;

  //! Return id of the nodebase
  virtual Index id() const = 0;

  //! Initialise shared pointer to nodal properties pool
  //! \param[in] prop_id Property id in the nodal property pool
  //! \param[in] nodal_properties Shared pointer to nodal properties pool
  virtual void initialise_property_handle(
      unsigned prop_id,
      std::shared_ptr<mpm::NodalProperties> property_handle) noexcept = 0;

  //! Assign coordinates
  virtual void assign_coordinates(const VectorDim& coord) = 0;

  //! Return coordinates
  //! \retval coordinates_ return coordinates of the nodebase
  virtual VectorDim coordinates() const = 0;

  //! Initialise properties
  virtual void initialise() noexcept = 0;

  //! Return degrees of freedom
  virtual unsigned dof() const = 0;

  //! Assign status
  virtual void assign_status(bool status) = 0;

  //! Return status
  virtual bool status() const = 0;

  //! Assign status
  virtual void assign_solving_status(bool status) = 0;

  //! Return status
  virtual bool solving_status() const = 0;

  //! Update mass at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] mass Mass from the particles in a cell
  virtual void update_mass(bool update, unsigned phase,
                           double mass) noexcept = 0;

  //! Return mass at a given node for a given phase
  virtual double mass(unsigned phase) const = 0;

  //! Update volume at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] volume Volume from the particles in a cell
  virtual void update_volume(bool update, unsigned phase,
                             double volume) noexcept = 0;

  //! Return volume at a given node for a given phase
  virtual double volume(unsigned phase) const = 0;

  //! Assign concentrated force to the node
  //! \param[in] direction Index corresponding to the direction of traction
  //! \param[in] traction Nodal concentrated force in specified direction
  //! \param[in] function math function
  //! \retval status Assignment status
  virtual bool assign_concentrated_force(
      unsigned phase, unsigned direction, double traction,
      const std::shared_ptr<FunctionBase>& function) = 0;

  //! Apply concentrated force to external force
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] current time
  virtual void apply_concentrated_force(unsigned phase,
                                        double current_time) = 0;

  //! Update external force (body force / traction force)
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] force External force from the particles in a cell
  virtual void update_external_force(bool update, unsigned phase,
                                     const VectorDim& force) noexcept = 0;

  //! Return external force
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim external_force(unsigned phase) const = 0;

  //! Update internal force (body force / traction force)
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] force Internal force from the particles in a cell
  virtual void update_internal_force(bool update, unsigned phase,
                                     const VectorDim& force) noexcept = 0;

  //! Return internal force
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim internal_force(unsigned phase) const = 0;

  //! Update pressure at the nodes from particle
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] mass_pressure Product of mass x pressure of a particle
  virtual void update_mass_pressure(unsigned phase,
                                    double mass_pressure) noexcept = 0;

  //! Apply pressure constraint
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \param[in] step Step in analysis
  virtual void apply_pressure_constraint(unsigned phase, double dt = 0,
                                         Index step = 0) noexcept = 0;

  //! Assign pressure at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] mass_pressure Product of mass x pressure of a particle
  virtual void assign_pressure(unsigned phase, double mass_pressure) = 0;

  //! Return pressure at a given node for a given phase
  //! \param[in] phase Index corresponding to the phase
  virtual double pressure(unsigned phase) const = 0;

  //! Update nodal momentum
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] momentum Momentum from the particles in a cell
  virtual void update_momentum(bool update, unsigned phase,
                               const VectorDim& momentum) noexcept = 0;

  //! Return momentum
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim momentum(unsigned phase) const = 0;

  //! Compute velocity from the momentum
  virtual void compute_velocity() = 0;

  //! Return velocity
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim velocity(unsigned phase) const = 0;

  //! Update nodal acceleration
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] acceleration Acceleration from the particles in a cell
  virtual void update_acceleration(bool update, unsigned phase,
                                   const VectorDim& acceleration) = 0;

  //! Return acceleration
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim acceleration(unsigned phase) const = 0;

  //! Compute acceleration
  //! \param[in] dt Time-step
  virtual bool compute_acceleration_velocity(unsigned phase,
                                             double dt) noexcept = 0;

  //! Compute acceleration and velocity with cundall damping factor
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \param[in] damping_factor Damping factor
  virtual bool compute_acceleration_velocity_cundall(
      unsigned phase, double dt, double damping_factor) noexcept = 0;

  //! Assign pressure constraint
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] pressure Applied pressure constraint
  //! \param[in] function math function
  virtual bool assign_pressure_constraint(
      unsigned phase, double pressure,
      const std::shared_ptr<FunctionBase>& function) = 0;

  //! Assign velocity constraint
  //! Directions can take values between 0 and Dim * Nphases
  //! \param[in] dir Direction of velocity constraint
  //! \param[in] velocity Applied velocity constraint
  virtual bool assign_velocity_constraint(unsigned dir, double velocity) = 0;

  //! Apply velocity constraints
  virtual void apply_velocity_constraints() = 0;

  //! Assign friction constraint
  //! Directions can take values between 0 and Dim * Nphases
  //! \param[in] dir Direction of friction constraint
  //! \param[in] sign Sign of normal wrt coordinate system for friction
  //! \param[in] friction Applied friction constraint
  virtual bool assign_friction_constraint(unsigned dir, int sign,
                                          double friction) = 0;

  //! Apply friction constraints
  //! \param[in] dt Time-step
  virtual void apply_friction_constraints(double dt) = 0;

  //! Assign rotation matrix
  //! \param[in] rotation_matrix Rotation matrix of the node
  virtual void assign_rotation_matrix(
      const Eigen::Matrix<double, Tdim, Tdim>& rotation_matrix) = 0;

  //! Add material id from material points to list of materials in materials_
  //! \param[in] id Material id to be stored at the node
  virtual void append_material_id(unsigned id) = 0;

  //! Return material ids in node
  virtual std::set<unsigned> material_ids() const = 0;

  //! Assign MPI rank to node
  //! \param[in] rank MPI Rank of the node
  virtual bool mpi_rank(unsigned rank) = 0;

  //! Assign MPI rank to node
  //! \param[in] rank MPI Rank of the node
  virtual std::set<unsigned> mpi_ranks() const = 0;

  //! Clear MPI ranks on node
  virtual void clear_mpi_ranks() = 0;

  //! Return ghost id
  virtual Index ghost_id() const = 0;

  //! Set ghost id
  virtual void ghost_id(Index gid) = 0;

  //! Update nodal property at the nodes from particle
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] property Property name
  //! \param[in] property_value Property quantity from the particles in the cell
  //! \param[in] mat_id Id of the material within the property data
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  virtual void update_property(bool update, const std::string& property,
                               const Eigen::MatrixXd& property_value,
                               unsigned mat_id, unsigned nprops) noexcept = 0;

  //! Compute multimaterial change in momentum
  virtual void compute_multimaterial_change_in_momentum() = 0;

  //! Compute multimaterial separation vector
  virtual void compute_multimaterial_separation_vector() = 0;

  //! Compute multimaterial normal unit vector
  virtual void compute_multimaterial_normal_unit_vector() = 0;

  /**
   * \defgroup XMPM Functions dealing with XMPM
   */
  /**@{*/
  //! Initialise shared pointer to nodal properties pool for discontinuity
  //! \param[in] prop_id Property id in the nodal property pool
  //! \param[in] nodal_properties Shared pointer to nodal properties pool
  virtual void initialise_discontinuity_property_handle(
      unsigned prop_id, std::shared_ptr<mpm::NodalProperties> property_handle) {
    throw std::runtime_error(
        "Calling the base class function (discontinuity_property) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return data in the nodal discontinuity properties map at a specific index
  //! \ingroup XMPM
  //! \param[in] property Property name
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  virtual Eigen::MatrixXd discontinuity_property(const std::string& property,
                                                 unsigned nprops = 1) {
    throw std::runtime_error(
        "Calling the base class function (discontinuity_property) in "
        "NodeBase:: "
        "illegal operation!");

    return Eigen::Matrix<double, 1, 1>::Zero();
  }

  //! Assign whether the node is enriched
  //! \ingroup XMPM
  //! \param[in] discontinuity discontinuity_enrich: true or false
  //! \param[in] discontinuity id
  virtual void assign_discontinuity_enrich(bool discontinuity,
                                           unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function (assign_discontinuity_enrich) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return whether the node is enriched
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  virtual bool discontinuity_enrich(unsigned dis_id) const {
    throw std::runtime_error(
        "Calling the base class function (discontinuity_enrich) in "
        "NodeBase:: "
        "illegal operation!");
    return true;
  }

  //! Update nodal property at the nodes from particle for discontinuity
  //! \ingroup XMPM
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] property Property name
  //! \param[in] property_value Property quantity from the particles in the cell
  //! \param[in] discontinuity_id Id of the material within the property data
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  virtual void update_discontinuity_property(
      bool update, const std::string& property,
      const Eigen::MatrixXd& property_value, unsigned discontinuity_id,
      unsigned nprops) {
    throw std::runtime_error(
        "Calling the base class function (update_discontinuity_property) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Assign nodal property at the nodes from particle for discontinuity
  //! \ingroup XMPM
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] property Property name
  //! \param[in] property_value Property quantity from the particles in the cell
  //! \param[in] discontinuity_id Id of the material within the property data
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  virtual void assign_discontinuity_property(
      bool update, const std::string& property,
      const Eigen::MatrixXd& property_value, unsigned discontinuity_id,
      unsigned nprops) {
    throw std::runtime_error(
        "Calling the base class function (assign_discontinuity_property) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Compute momentum for discontinuity
  //! \ingroup XMPM
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  virtual bool compute_momentum_discontinuity(unsigned phase, double dt) {
    throw std::runtime_error(
        "Calling the base class function (compute_momentum_discontinuity) in "
        "NodeBase:: "
        "illegal operation!");
    return true;
  }

  //! Compute momentum for discontinuity with cundall damping factor
  //! \ingroup XMPM
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \param[in] damping_factor Damping factor
  virtual bool compute_momentum_discontinuity_cundall(unsigned phase, double dt,
                                                      double damping_factor) {
    throw std::runtime_error(
        "Calling the base class function "
        "(compute_momentum_discontinuity_cundall) in "
        "NodeBase:: "
        "illegal operation!");

    return true;
  }

  //! Apply self-contact force of the discontinuity
  //! \ingroup XMPM
  //! \param[in] dt Time-step
  virtual void self_contact_discontinuity(double dt) {
    throw std::runtime_error(
        "Calling the base class function (self_contact_discontinuity) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return the discontinuity_prop_id
  //! \ingroup XMPM
  virtual unsigned discontinuity_prop_id() const {
    throw std::runtime_error(
        "Calling the base class function (discontinuity_prop_id) in "
        "NodeBase:: "
        "illegal operation!");
    return 0;
  }

  //! Add a cell id
  //! \ingroup XMPM
  virtual void add_cell_id(Index id) {
    throw std::runtime_error(
        "Calling the base class function (add_cell_id) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return cells_
  //! \ingroup XMPM
  virtual std::vector<Index> cells() {
    throw std::runtime_error(
        "Calling the base class function (cells) in "
        "NodeBase:: "
        "illegal operation!");
    std::vector<Index> cells;
    return cells;
  }

  //! Update the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the value of the nodal levelset_phi
  //! \param[in]the discontinuity id
  virtual void update_levelset_phi(double phi, int dis_id) {
    throw std::runtime_error(
        "Calling the base class function (update_levelset_phi) in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Update the nodal enriched mass
  //! \ingroup XMPM
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] the value of the enriched mass
  virtual void update_mass_enrich(bool update, Eigen::Matrix<double, 3, 1>) {
    throw std::runtime_error(
        "Calling the base class function update_mass_enriched in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Update the nodal enriched momentum
  //! \ingroup XMPM
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] the value of the enriched momentum
  virtual void update_momentum_enrich(bool update,
                                      Eigen::Matrix<double, Tdim, 3> momenta) {
    throw std::runtime_error(
        "Calling the base class function update_momenta_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Update the nodal enriched internal_force
  //! \ingroup XMPM
  //! \param[in] the value of the enriched momentum
  virtual void update_internal_force_enrich(
      Eigen::Matrix<double, Tdim, 3> internal_force) {
    throw std::runtime_error(
        "Calling the base class function update_internal_force_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Update the nodal enriched external_force
  //! \ingroup XMPM
  //! \param[in] the value of the enriched external_force
  virtual void update_external_force_enrich(
      Eigen::Matrix<double, Tdim, 3> external_force) {
    throw std::runtime_error(
        "Calling the base class function update_external_force_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return mass_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  virtual Eigen::Matrix<double, 3, 1> mass_enrich() {
    throw std::runtime_error(
        "Calling the base class function mass_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }
  //! Return momentum_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  virtual Eigen::Matrix<double, Tdim, 3> momentum_enrich() {
    throw std::runtime_error(
        "Calling the base class function momentum_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }
  //! Return internal_force_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  virtual Eigen::Matrix<double, Tdim, 3> internal_force_enrich() {
    throw std::runtime_error(
        "Calling the base class function internal_force_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }
  //! Return external_force_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  virtual Eigen::Matrix<double, Tdim, 3> external_force_enrich() {
    throw std::runtime_error(
        "Calling the base class function external_force_enrich in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Assign the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the value of the nodal levelset_phi
  //! \param[in] discontinuity id
  virtual void assign_levelset_phi(double phi, int dis_id) {
    throw std::runtime_error(
        "Calling the base class function assign_levelset_phi in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the discontinuity_id
  virtual double levelset_phi(int discontinuity_id) {
    throw std::runtime_error(
        "Calling the base class function levelset_phi in "
        "NodeBase:: "
        "illegal operation!");
  }
  //! Update the nodal mass_h_
  //! \ingroup XMPM
  //! \param[in] the value of mass_h_
  virtual void update_mass_h(double mass_h) {
    throw std::runtime_error(
        "Calling the base class function update_mass_h in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Initialise the nodal mass_h_
  //! \ingroup XMPM
  virtual void initialise_mass_h() {
    throw std::runtime_error(
        "Calling the base class function initialise_mass_h in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Initialise the nodal mass
  //! \param[in] phase Index corresponding to the phase
  //! \ingroup XMPM
  virtual void initialise_mass(unsigned phase) {
    throw std::runtime_error(
        "Calling the base class function initialise_mass in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Determine node type from mass and enriched mass
  //! \param[in] the discontinuity_id
  //! \ingroup XMPM
  virtual void determine_node_type(int discontinuity_id) {
    throw std::runtime_error(
        "Calling the base class function determine_node_type in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Return the discontinuity id at nodes
  //! \ingroup XMPM
  //! \retval discontinuity_id_
  virtual Eigen::Matrix<int, 2, 1> discontinuity_id() {
    throw std::runtime_error(
        "Calling the base class function discontinuity_id in "
        "NodeBase:: "
        "illegal operation!");
    return Eigen::Matrix<int, 2, 1>::Zero();
  }

  //! Return enrich_type
  //! \ingroup XMPM
  //! \retval enrich_type_
  virtual int enrich_type() {
    throw std::runtime_error(
        "Calling the base class function enrich_type in "
        "NodeBase:: "
        "illegal operation!");
    return 0;
  }

  //! Return normal at a given node
  //! \ingroup XMPM
  //! \param[in] the discontinuity id
  virtual VectorDim normal(unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function normal in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Assign normal at a given node
  //! \ingroup XMPM
  //! \param[in] the discontinuity id
  virtual void assign_normal(VectorDim normal, unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function normal in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Reset the size of the discontinuity
  //! \ingroup XMPM
  //! \param[in] the number of the discontinuity
  virtual void reset_discontinuity_size(int size) {
    throw std::runtime_error(
        "Calling the base class function reset_discontinuity_size in "
        "NodeBase:: "
        "illegal operation!");
  };

  //! Assign contact condition by contact distance
  //! \ingroup XMPM
  //! \param[in] the contact id
  //! \param[in] contact or not
  virtual void assign_contact(unsigned id, bool status) {
    throw std::runtime_error(
        "Calling the base class function assign_contact in "
        "NodeBase:: "
        "illegal operation!");
  };

  //! Assign friction coefficient
  //! \ingroup XMPM
  //! \param[in] the friction coefficient
  //! \param[in] the discontinuity id
  virtual void assign_friction_coef(double friction_coef, unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function assign_friction_coef in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! Assign cohesion
  //! \ingroup XMPM
  //! \param[in] the cohesion
  //! \param[in] the discontinuity id
  virtual void assign_cohesion(double cohesion, unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function assign_cohesion in "
        "NodeBase:: "
        "illegal operation!");
  }

  //! update cohesion area
  //! \ingroup XMPM
  //! \param[in] the cohesion area
  //! \param[in] the discontinuity id
  virtual void update_cohesion_area(double cohesion, unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function update_cohesion_area in "
        "NodeBase:: "
        "illegal operation!");
  }

  /**@}*/

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Initialise nodal properties for implicit solver
  //! \ingroup Implicit
  virtual void initialise_implicit() noexcept = 0;

  //! Initialise nodal forces
  //! \ingroup Impolicit
  virtual void initialise_force() noexcept = 0;

  //! Update nodal inertia
  //! \ingroup Implicit
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] inertia Inertia from the particles in a cell
  virtual void update_inertia(bool update, unsigned phase,
                              const VectorDim& inertia) noexcept = 0;

  //! Return inertia
  //! \ingroup Implicit
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim inertia(unsigned phase) const = 0;

  //! Compute velocity and acceleration from the momentum and inertia
  //! \ingroup Implicit
  virtual void compute_velocity_acceleration() = 0;

  //! Return displacement
  //! \ingroup Implicit
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim displacement(unsigned phase) const = 0;

  //! Update velocity and acceleration by Newmark scheme
  //! \ingroup Implicit
  //! \param[in] newmark_beta Parameter beta of Newmark scheme
  //! \param[in] newmark_gamma Parameter gamma of Newmark scheme
  //! \param[in] dt Time-step
  virtual void update_velocity_acceleration_newmark(unsigned phase,
                                                    double newmark_beta,
                                                    double newmark_gamma,
                                                    double dt) = 0;

  //! Assign displacement constraint for implicit solver
  //! Directions can take values between 0 and Dim * Nphases
  //! \ingroup Implicit
  //! \param[in] dir Direction of displacement constraint
  //! \param[in] displacement Applied pressure constraint
  //! \param[in] function math function
  virtual bool assign_displacement_constraint(
      const unsigned dir, const double displacement,
      const std::shared_ptr<FunctionBase>& function) = 0;

  //! Return displacement constraint
  //! \ingroup Implicit
  virtual double displacement_constraint(const unsigned dir,
                                         const double current_time) const = 0;

  //! Update displacement increment at the node
  //! \ingroup Implicit
  virtual void update_displacement_increment(
      const Eigen::VectorXd& displacement_increment, unsigned phase,
      unsigned nactive_node) = 0;
  /**@{*/

  /**
   * \defgroup MultiPhase Functions dealing with multi-phase MPM
   */
  /**@{*/

  //! Return interpolated density at a given node for a given phase
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  virtual double density(unsigned phase) const = 0;

  //! Compute nodal density
  //! \ingroup MultiPhase
  virtual void compute_density() = 0;

  //! Assign free surface
  //! \ingroup MultiPhase
  virtual void assign_free_surface(bool free_surface) = 0;

  //! Return free surface bool
  //! \ingroup MultiPhase
  virtual bool free_surface() const = 0;

  //! Initialise two-phase nodal properties
  //! \ingroup MultiPhase
  virtual void initialise_twophase() noexcept = 0;

  //! Update internal force (body force / traction force)
  //! \ingroup MultiPhase
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] drag_force Drag force from the particles in a cell
  //! \retval status Update status
  virtual void update_drag_force_coefficient(bool update,
                                             const VectorDim& drag_force) = 0;

  //! Return drag force at a given node
  //! \ingroup MultiPhase
  virtual VectorDim drag_force_coefficient() const = 0;

  //! Compute acceleration and velocity for two phase
  //! \ingroup MultiPhase
  //! \param[in] dt Timestep in analysis
  virtual bool compute_acceleration_velocity_twophase_explicit(
      double dt) noexcept = 0;

  //! Compute acceleration and velocity for two phase with cundall damping
  //! \ingroup MultiPhase
  //! \param[in] dt Timestep in analysis
  virtual bool compute_acceleration_velocity_twophase_explicit_cundall(
      double dt, double damping_factor) noexcept = 0;

  //! Compute semi-implicit acceleration and velocity
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \details Can be used for both semi-implicit navier-stokes and two-phase
  //! solvers
  //! \retval status Computation status
  virtual bool compute_acceleration_velocity_semi_implicit_corrector(
      unsigned phase, double dt) = 0;

  //! Compute semi-implicit acceleration and velocity with Cundall damping
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \details Can be used for both semi-implicit navier-stokes and two-phase
  //! solvers
  //! \retval status Computation status
  virtual bool compute_acceleration_velocity_semi_implicit_corrector_cundall(
      unsigned phase, double dt, double damping_factor) = 0;

  //! Assign active id
  //! \ingroup MultiPhase
  virtual void assign_active_id(Index id) = 0;

  //! Return active id
  //! \ingroup MultiPhase
  virtual mpm::Index active_id() const = 0;

  //! Assign global active id
  //! \ingroup MultiPhase
  virtual void assign_global_active_id(Index id) = 0;

  //! Return global active id
  //! \ingroup MultiPhase
  virtual mpm::Index global_active_id() const = 0;

  //! Return pressure constraint
  //! \ingroup MultiPhase
  virtual double pressure_constraint(const unsigned phase,
                                     const double current_time) const = 0;

  //! Update pressure increment at the node
  //! \ingroup MultiPhase
  virtual void update_pressure_increment(
      const Eigen::VectorXd& pressure_increment, unsigned phase,
      double current_time = 0.) = 0;

  //! Return nodal pressure increment
  //! \ingroup MultiPhase
  virtual double pressure_increment() const = 0;

  //! Return map of velocity constraints
  //! \ingroup MultiPhase
  virtual std::map<unsigned, double>& velocity_constraints() = 0;

  //! Update intermediate velocity at the node
  //! \ingroup MultiPhase
  virtual void update_intermediate_acceleration_velocity(
      const unsigned phase, const Eigen::MatrixXd& acceleration_inter,
      double dt) = 0;

  //! Update correction force
  //! \ingroup MultiPhase
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] force Correction force from the particles in a cell
  virtual void update_correction_force(bool update, unsigned phase,
                                       const VectorDim& force) noexcept = 0;

  //! Return correction force
  //! \ingroup MultiPhase
  //! \param[in] phase Index corresponding to the phase
  virtual VectorDim correction_force(unsigned phase) const = 0;

  /**@}*/

  /**
   * \defgroup Nonlocal Functions dealing with MPM with nonlocal shape function
   */
  /**@{*/

  //! Function that initialise variables for nonlocal MPM
  //! \ingroup Nonlocal
  virtual void initialise_nonlocal_node() noexcept = 0;

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
  virtual void assign_nonlocal_node_type(unsigned dir, unsigned type) = 0;

  //! Function which return nodal nonlocal type vector
  //! \ingroup Nonlocal
  virtual std::vector<unsigned> nonlocal_node_type() const = 0;

  /**@}*/

};  // NodeBase class
}  // namespace mpm

#endif  // MPM_NODE_BASE_H_
