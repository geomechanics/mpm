#ifndef MPM_CELL_H_
#define MPM_CELL_H_

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/LU"

#include "affine_transform.h"
#include "discontinuity_element.h"
#include "element.h"
#include "geometry.h"
#include "logger.h"
#include "map.h"
#include "node_base.h"
#include "node_xmpm.h"
#include "quadrature.h"

namespace mpm {

//! Cell class
//! \brief Base class that stores the information about cells
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Cell {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define DOFs
  static const unsigned Tdof = (Tdim == 1) ? 1 : 3 * (Tdim - 1);

  //! Constructor with id, number of nodes and element
  //! \param[in] id Global cell id
  //! \param[in] nnodes Number of nodes per cell
  //! \param[in] elementptr Pointer to an element type
  //! \param[in] isoparametric Cell type is isoparametric
  Cell(Index id, unsigned nnodes,
       const std::shared_ptr<Element<Tdim>>& elementptr,
       bool isoparametric = true);

  //! Default destructor
  ~Cell() = default;

  //! Delete copy constructor
  Cell(const Cell<Tdim>&) = delete;

  //! Delete assignement operator
  Cell& operator=(const Cell<Tdim>&) = delete;

  //! Return id of the cell
  Index id() const { return id_; }

  //! Initialise cell properties
  bool initialise();

  //! Return the initialisation status of cells
  //! \retval initialisation_status Cell has nodes, element type and volumes
  bool is_initialised() const;

  //! Assign quadrature
  void assign_quadrature(unsigned nquadratures);

  //! Generate points
  std::vector<Eigen::Matrix<double, Tdim, 1>> generate_points();

  //! Return the number of particles
  unsigned nparticles() const { return particles_.size(); }

  //! Assign global nparticles
  //! \param[in] nparticles Number of global particles of cell
  void nglobal_particles(unsigned nparticles) {
    nglobal_particles_ = nparticles;
  }

  //! nglobal particles
  //! \retval nglobal_particles_ Number of global particles of cell
  unsigned nglobal_particles() const { return nglobal_particles_; }

  //! Return the status of a cell: active (if a particle is present)
  bool status() const { return particles_.size(); }

  //! Return particles_
  std::vector<Index> particles() const { return particles_; }

  //! Number of nodes
  unsigned nnodes() const { return nodes_.size(); }

  //! Return nodes of the cell
  std::vector<std::shared_ptr<mpm::NodeBase<Tdim>>> nodes() const {
    return nodes_;
  }

  //! Return nodes id in a cell
  std::set<mpm::Index> nodes_id() const {
    std::set<mpm::Index> nodes_id_lists;
    for (const auto& node : nodes_) nodes_id_lists.insert(node->id());
    return nodes_id_lists;
  }

  //! Side node pair ids
  std::vector<std::array<mpm::Index, 2>> side_node_pairs() const;

  //! Activate nodes if particle is present
  void activate_nodes();

  //! Return a pointer to element type of a cell
  std::shared_ptr<Element<Tdim>> element_ptr() { return element_; }

  //! Return the number of shape functions, returns zero if the element type is
  //! not set.
  unsigned nfunctions() const {
    return (this->element_ != nullptr ? this->element_->nfunctions() : 0);
  };

  //! Add a node pointer to cell
  //! \param[in] local_id local id of the node
  //! \param[in] node A shared pointer to the node
  //! \retval insertion_status Return the successful addition of a node
  bool add_node(unsigned local_id, const std::shared_ptr<NodeBase<Tdim>>& node);

  //! Add a neighbour cell
  //! \param[in] neighbour_id id of the neighbouring cell
  //! \retval insertion_status Return the successful addition of a node
  bool add_neighbour(mpm::Index neighbour_id);

  //! Number of neighbours
  unsigned nneighbours() const { return neighbours_.size(); }

  //! Return neighbour ids
  std::set<mpm::Index> neighbours() const { return neighbours_; }

  //! Add an id of a particle in the cell
  //! \param[in] id Global id of a particle
  //! \retval status Return the successful addition of a particle id
  bool add_particle_id(Index id);

  //! Remove a particle id from the cell (moved to a different cell / killed)
  //! \param[in] id Global id of a particle
  void remove_particle_id(Index id);

  //! Clear all particle ids in the cell
  void clear_particle_ids() { particles_.clear(); }

  //! Compute the volume of the cell
  void compute_volume();

  //! Return the volume of the cell
  double volume() const { return volume_; }

  //! Compute the centroid of the cell
  void compute_centroid();

  //! Return the centroid of the cell
  Eigen::Matrix<double, Tdim, 1> centroid() const { return centroid_; }

  //! Return the dN/dx at the centroid of the cell
  Eigen::MatrixXd dn_dx_centroid() const { return dn_dx_centroid_; }

  //! Compute mean length of cell
  void compute_mean_length();

  //! Return the mean_length
  double mean_length() const { return mean_length_; }

  //! Return nodal coordinates
  Eigen::MatrixXd nodal_coordinates() const { return nodal_coordinates_; }

  //! Check if a point is in a cartesian cell by checking the domain ranges
  //! \param[in] point Coordinates of point
  inline bool point_in_cartesian_cell(
      const Eigen::Matrix<double, Tdim, 1>& point);

  //! Check if a point is in a isoparametric cell
  //! Use an affine transformation and NR to check if a transformed point is in
  //! a unit cell. This is useful for points on the surface, where
  //! volume calculations are tricky. The transformed point should be between -1
  //! and 1 in a unit cell
  //! \param[in] point Coordinates of point
  //! \param[in|out] xi Local coordinates of point
  //! \retval status Return if a point is in cell or not
  inline bool is_point_in_cell(const Eigen::Matrix<double, Tdim, 1>& point,
                               Eigen::Matrix<double, Tdim, 1>* xi);

  //! Return the local coordinates of a point in a cell
  //! \param[in] point Coordinates of a point
  //! \retval xi Local coordinates of a point
  inline Eigen::Matrix<double, Tdim, 1> local_coordinates_point(
      const Eigen::Matrix<double, Tdim, 1>& point);

  //! Return the local coordinates of a point in a unit cell
  //! Using newton iteration / affine transformation
  //! \param[in] point Coordinates of a point
  //! \retval xi Local coordinates of a point
  inline Eigen::Matrix<double, Tdim, 1> transform_real_to_unit_cell(
      const Eigen::Matrix<double, Tdim, 1>& point);

  //! Assign MPI rank to nodes
  void assign_mpi_rank_to_nodes();

  //! Compute normal vector
  void compute_normals();

  //! Return sorted face node ids
  std::vector<std::vector<mpm::Index>> sorted_face_node_ids();

  //! Assign ranks
  //! \param[in] Rank of cell
  void rank(unsigned mpi_rank);

  //! Return rank
  unsigned rank() const;

  //! Return previous mpi rank
  unsigned previous_mpirank() const;

  /**
   * \defgroup XMPM Functions dealing with XMPM
   */
  /**@{*/

  //! Assign discontinuity element type
  //! \ingroup XMPM
  //! \param[in] the enriched type
  //! \param[in] dis_id the discontinuity id
  void assign_discontinuity_type(mpm::EnrichType type, unsigned dis_id);

  //! Initialize discontinuity element properties
  //! \ingroup XMPM
  void initialise_element_properties_discontinuity();

  //! Assign the normal direction of the discontinuity in the cell
  //! \ingroup XMPM
  //! \param[in] the normal direction
  void assign_normal_discontinuity(VectorDim normal, unsigned dis_id);

  //! Assign the constant parameters of the discontinuity in the cell
  //! \ingroup XMPM
  //! \param[in] the constant parameters
  //! \param[in] dis_id the discontinuity id
  void assign_d_discontinuity(double d, unsigned dis_id) {
    this->discontinuity_element_[dis_id]->assign_d(d);
  };

  //! Assign the normal direction of the discontinuity in the cell
  //! \ingroup XMPM
  //! \param[in] the normal direction
  //! \param[in] the plane constant
  //! \param[in] dis_id the discontinuity id
  void assign_normal_discontinuity(VectorDim normal, double d, unsigned dis_id);

  //! Return the normal direction of the discontinuity in the cell
  //! \ingroup XMPM
  //! \retval the normal direction of the discontinuity
  //! \param[in] the number of the discontinuity
  VectorDim normal_discontinuity(unsigned dis_id) {
    if (discontinuity_element_[dis_id] == nullptr)
      return Eigen::Matrix<double, Tdim, 1>::Zero();
    return discontinuity_element_[dis_id]->normal_discontinuity();
  };

  //! Return discontinuity element type
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  unsigned element_discontinuity_type(unsigned dis_id);

  //! Find the potential tip element
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  void find_potential_tip_cell(unsigned dis_id);

  //! Determine tip element
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  void find_tip_cell(unsigned dis_id);

  //! Compute normal vector of discontinuity by the nodal level set values
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  void compute_normal_vector_discontinuity(unsigned dis_id);

  //! Compute the discontinuity plane by the nodal level set values
  //! \ingroup XMPM
  //! \param[in] true: compute by the enriched nodes
  //! \param[in] discontinuity id
  void compute_plane_discontinuity(bool enrich, unsigned dis_id);

  //! Compute the discontinuity point: the average coordinates of the
  //! \ingroup XMPM
  //! \param[in] mark points list
  //! \param[in] the id of the discontinuity
  void compute_discontinuity_point(std::vector<VectorDim>& coordinates,
                                   unsigned dis_id);

  //! Return the product of the maximum and minimum nodal level set value
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  //! \retval the product of the maximum and minimum nodal level set value
  double product_levelset(unsigned dis_id);

  //! Return the constant value of the discontinuity plane
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  double d_discontinuity(unsigned dis_id) {
    return this->discontinuity_element_[dis_id]->d_discontinuity();
  }

  //! Determine the celltype by the nodal level set
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  void determine_crossed_cell(unsigned dis_id);

  //! Compute the nodal level set values by plane equations
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  void compute_nodal_levelset_equation(unsigned dis_id);

  //! Compute the area of the discontinuity
  //! \ingroup XMPM
  void compute_area_discontinuity(unsigned dis_id);

  //! Return the area of the discontinuity
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  double discontinuity_area(unsigned dis_id) {
    return this->discontinuity_element_[dis_id]->area();
  }

  //! Assign the area of the discontinuity to nodes
  //! \ingroup XMPM
  //! \param[in] discontinuity id
  void assign_cohesion_area(unsigned dis_id);

  //! Reset the size of the discontinuity
  //! \ingroup XMPM
  //! \param[in] the number of the discontinuity
  void reset_discontinuity_size(int size) {
    discontinuity_element_.resize(size, nullptr);
  };

  //! Assign interaction element type
  //! \param[in] the interaction type
  void assign_interaction_type(unsigned dis_id, mpm::InteractionType type) {
    std::lock_guard<std::mutex> guard(cell_mutex_);
    this->discontinuity_element_[dis_id]->assign_interaction_type(type);
  }

  //! Return interaction element type
  //! \retval the interaction type
  unsigned interaction_type(unsigned dis_id) {
    if (discontinuity_element_[dis_id] == nullptr)
      return mpm::InteractionType::Propagate;
    return this->discontinuity_element_[dis_id]->interaction_type();
  }

  //! Return the maximum displacement
  //! \param[in] the number of the discontinuity
  //! \retval the maximum displacement
  double max_dudx(unsigned dis_id) {
    return this->discontinuity_element_[dis_id]->max_dudx();
  }

  //! Assign the maximum displacement
  //! \param[in] max_dudx the maximum displacement
  //! \param[in] the number of the discontinuity
  void assign_max_dudx(double max_dudx, unsigned dis_id) {
    this->discontinuity_element_[dis_id]->assign_max_dudx(max_dudx);
  }

  //! Return the discontinuity id which can propagates in this cell
  //! \retval discontinuity id
  unsigned dis_id() { return dis_id_; }

  //! Assign the discontinuity id which can propagates in this cell
  //! \param[in] dis_id discontinuity id
  void assign_dis_id(unsigned dis_id) {
    std::lock_guard<std::mutex> guard(cell_mutex_);
    dis_id_ = dis_id;
  }

  /**@}*/

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/

  //! Initialize local elemental stiffness matrices
  //! \ingroup Implicit
  bool initialise_element_stiffness_matrix();

  //! Return local stiffness
  //! \ingroup Implicit
  const Eigen::MatrixXd& stiffness_matrix() { return stiffness_matrix_; };

  //! Compute local material stiffness matrix (Used in equilibrium equation)
  //! \ingroup Implicit
  //! \param[in] bmatrix B matrix
  //! \param[in] dmatrix constitutive relations matrix
  //! \param[in] pvolume particle volume
  //! \param[in] multiplier multiplier
  void compute_local_material_stiffness_matrix(
      const Eigen::MatrixXd& bmatrix, const Eigen::MatrixXd& dmatrix,
      double pvolume, double multiplier = 1.0) noexcept;

  //! Compute local geometric stiffness matrix (Used in equilibrium equation)
  //! \ingroup Implicit
  //! \param[in] geometric_stiffness local geometric stiffness matrix
  //! contribution from material point
  //! \param[in] pvolume particle volume
  //! \param[in] multiplier multiplier
  void compute_local_geometric_stiffness_matrix(
      const Eigen::MatrixXd& geometric_stiffness, double pvolume,
      double multiplier = 1.0) noexcept;

  //! Compute local mass matrix (Used in equilibrium equation)
  //! \ingroup Implicit
  //! \param[in] shapefn shape function
  //! \param[in] pvolume particle volume
  //! \param[in] multiplier multiplier
  void compute_local_mass_matrix(const Eigen::VectorXd& shapefn, double pvolume,
                                 double multiplier = 1.0) noexcept;
  /**@}*/

  /**
   * \defgroup MultiPhase Functions dealing with multi-phase MPM
   */
  /**@{*/

  //! Assign solving status
  //! \ingroup MultiPhase
  //! \param[in] status Cell solving status for parallel free-surface detection
  void assign_solving_status(bool status) { solving_status_ = status; }

  //! Return solving status of a cell: active (if a particle is present in all
  //! MPI rank)
  //! \ingroup MultiPhase
  bool solving_status() const { return solving_status_; }

  //! Assign free surface
  //! \ingroup MultiPhase
  //! \param[in] free_surface boolean indicating free surface cell
  void assign_free_surface(bool free_surface) { free_surface_ = free_surface; };

  //! Return free surface bool
  //! \ingroup MultiPhase
  //! \retval free_surface_ indicating free surface cell
  bool free_surface() const { return free_surface_; };

  //! Assign volume traction to node
  //! \ingroup MultiPhase
  //! \param[in] volume_fraction cell volume fraction
  void assign_volume_fraction(double volume_fraction) {
    volume_fraction_ = volume_fraction;
  };

  //! Return cell volume fraction
  //! \ingroup MultiPhase
  //! \retval volume_fraction_ cell volume fraction
  double volume_fraction() const { return volume_fraction_; };

  //! Map cell volume to the nodes
  //! \ingroup MultiPhase
  //! \param[in] phase to map volume
  void map_cell_volume_to_nodes(unsigned phase);

  //! Initialize local elemental matrices
  //! \ingroup MultiPhase
  bool initialise_element_matrix();

  //! Initialize local elemental matrices for two-phase one-point solver
  //! \ingroup MultiPhase
  bool initialise_element_matrix_twophase();

  //! Return local node indices
  //! \ingroup MultiPhase
  Eigen::VectorXi local_node_indices();

  //! Return drag matrix for two-phase one-point solver
  //! \ingroup MultiPhase
  //! \param[in] dir Desired direction
  const Eigen::MatrixXd& drag_matrix(unsigned dir) {
    return drag_matrix_[dir];
  };

  //! Compute local matrix for drag force coupling for two-phase one-point
  //! solver
  //! \ingroup MultiPhase
  //! \param[in] shapefn Shape function
  //! \param[in] pvolume Volume weight
  //! \param[in] multiplier Drag force multiplier
  void compute_local_drag_matrix(const Eigen::VectorXd& shapefn, double pvolume,
                                 const VectorDim& multiplier) noexcept;

  //! Return local laplacian
  //! \ingroup MultiPhase
  const Eigen::MatrixXd& laplacian_matrix() { return laplacian_matrix_; };

  //! Compute local laplacian matrix (Used in poisson equation)
  //! \ingroup MultiPhase
  //! \param[in] grad_shapefn shape function gradient
  //! \param[in] pvolume volume weight
  //! \param[in] multiplier multiplier
  void compute_local_laplacian(const Eigen::MatrixXd& grad_shapefn,
                               double pvolume,
                               double multiplier = 1.0) noexcept;

  //! Return local laplacian RHS matrix
  //! \ingroup MultiPhase
  const Eigen::MatrixXd& poisson_right_matrix() {
    return poisson_right_matrix_;
  };

  //! Return local laplacian RHS matrix for twophase
  //! \ingroup MultiPhase
  //! \param[in] phase Phase identifier
  const Eigen::MatrixXd& poisson_right_matrix(unsigned phase) {
    return poisson_right_matrix_twophase_[phase];
  };

  //! Compute local poisson RHS matrix (Used in poisson equation)
  //! \ingroup MultiPhase
  //! \param[in] shapefn shape function
  //! \param[in] grad_shapefn shape function gradient
  //! \param[in] pvolume volume weight
  void compute_local_poisson_right(const Eigen::VectorXd& shapefn,
                                   const Eigen::MatrixXd& grad_shapefn,
                                   double pvolume,
                                   double multiplier = 1.0) noexcept;

  //! Compute local poisson RHS matrix (Used in poisson equation)
  //! \ingroup MultiPhase
  //! \param[in] phase Phase identifier
  //! \param[in] shapefn shape function
  //! \param[in] grad_shapefn shape function gradient
  //! \param[in] pvolume volume weight
  void compute_local_poisson_right_twophase(unsigned phase,
                                            const Eigen::VectorXd& shapefn,
                                            const Eigen::MatrixXd& grad_shapefn,
                                            double pvolume,
                                            double multiplier = 1.0) noexcept;

  //! Return local correction matrix
  //! \ingroup MultiPhase
  const Eigen::MatrixXd& correction_matrix() { return correction_matrix_; };

  //! Return local correction matrix for two phase
  //! \ingroup MultiPhase
  //! \param[in] phase Phase identifier
  const Eigen::MatrixXd& correction_matrix(unsigned phase) {
    return correction_matrix_twophase_[phase];
  };

  //! Compute local correction matrix (Used to correct velocity)
  //! \ingroup MultiPhase
  //! \param[in] shapefn shape function
  //! \param[in] grad_shapefn shape function gradient
  //! \param[in] pvolume volume weight
  void compute_local_correction_matrix(const Eigen::VectorXd& shapefn,
                                       const Eigen::MatrixXd& grad_shapefn,
                                       double pvolume) noexcept;

  //! Compute local correction matrix for two phase (Used to correct velocity)
  //! \ingroup MultiPhase
  //! \param[in] phase Phase identifier
  //! \param[in] shapefn shape function
  //! \param[in] grad_shapefn shape function gradient
  //! \param[in] pvolume volume weight
  void compute_local_correction_matrix_twophase(
      unsigned phase, const Eigen::VectorXd& shapefn,
      const Eigen::MatrixXd& grad_shapefn, double pvolume,
      double multiplier = 1.0) noexcept;

  /**@}*/

  /**
   * \defgroup Nonlocal Functions dealing with MPM with nonlocal shape function
   */
  /**@{*/

  //! Function that check and return the status whether the cell is upgradable
  //! \ingroup Nonlocal
  //! \param[in] new_nnodes New number of nodes to be assigned
  bool upgrade_status(unsigned new_nnodes);

  //! Assign a new elementptr for nonlocal element with a specific node size
  //! \ingroup Nonlocal
  //! \param[in] elementptr New element pointer associated with nonlocal type
  bool assign_nonlocal_elementptr(
      const std::shared_ptr<Element<Tdim>>& elementptr);

  //! Initialising nonlocal cell-element properties
  //! \ingroup Nonlocal
  //! \brief Assign a specific connectivity properties to the assigned nonlocal
  //! elements
  //! \param[in] nonlocal_properties A map of selected nonlocal element
  //! properties
  bool initialiase_nonlocal(
      const tsl::robin_map<std::string, double>& nonlocal_properties);

  /**@}*/

 private:
  //! Approximately check if a point is in a cell
  //! \param[in] point Coordinates of point
  bool approx_point_in_cell(const Eigen::Matrix<double, Tdim, 1>& point);

 private:
  //! Mutex
  std::mutex cell_mutex_;
  //! cell id
  Index id_{std::numeric_limits<Index>::max()};
  //! MPI Rank
  unsigned rank_{0};
  //! Previous MPI Rank
  unsigned previous_mpirank_{0};
  //! Isoparametric
  bool isoparametric_{true};
  //! Number of nodes
  unsigned nnodes_{0};
  //! Volume
  double volume_{std::numeric_limits<double>::lowest()};
  //! Centroid
  VectorDim centroid_;
  //! mean_length of cell
  double mean_length_{std::numeric_limits<double>::max()};
  //! particles ids in cell
  std::vector<Index> particles_;
  //! Number of global nparticles
  unsigned nglobal_particles_{0};
  //! Container of node pointers (local id, node pointer)
  std::vector<std::shared_ptr<NodeBase<Tdim>>> nodes_;
  //! Nodal coordinates
  Eigen::MatrixXd nodal_coordinates_;
  //! Container of cell neighbour ids
  std::set<mpm::Index> neighbours_;
  //! Shape function
  std::shared_ptr<Element<Tdim>> element_{nullptr};
  //! Quadrature
  std::shared_ptr<Quadrature<Tdim>> quadrature_{nullptr};
  //! dN/dx
  Eigen::MatrixXd dn_dx_centroid_;
  //! Velocity constraints
  //! key: face_id, value: pair of direction [0/1/2] and velocity value
  std::map<unsigned, std::vector<std::pair<unsigned, double>>>
      velocity_constraints_;
  //! Normal of face
  //! first-> face_id, second->vector of the normal
  std::map<unsigned, Eigen::VectorXd> face_normals_;

  /**
   * \defgroup ImplicitVariables Variables for single-phase implicit
   * MPM
   * @{
   */
  //! Local stiffness matrix
  Eigen::MatrixXd stiffness_matrix_;
  /**@}*/

  /**
   * \defgroup MultiPhaseVariables Variables for multi-phase MPM
   * @{
   */
  //! Solving status
  bool solving_status_{false};
  //! Free surface bool
  bool free_surface_{false};
  //! Volume fraction
  double volume_fraction_{0.0};
  //! Local laplacian matrix
  Eigen::MatrixXd laplacian_matrix_;
  //! Local poisson RHS matrix
  Eigen::MatrixXd poisson_right_matrix_;
  //! Local correction RHS matrix
  Eigen::MatrixXd correction_matrix_;
  //! Drag force coefficient
  std::vector<Eigen::MatrixXd> drag_matrix_;
  //! Local poisson RHS matrix for twophase
  std::vector<Eigen::MatrixXd> poisson_right_matrix_twophase_;
  //! Local poisson RHS matrix for twophase
  std::vector<Eigen::MatrixXd> correction_matrix_twophase_;
  /**@}*/

  /**
   * \defgroup XMPM Variables for XMPM
   */
  /**@{*/
  //! discontinuity element list
  std::vector<std::shared_ptr<DiscontinuityElement<Tdim>>>
      discontinuity_element_{nullptr};
  //! discontinuity id which propagates normally in this cell
  unsigned dis_id_;
  /**@}*/

  //! Logger
  std::unique_ptr<spdlog::logger> console_;
};  // Cell class
}  // namespace mpm

#include "cell.tcc"
#include "cell_implicit.tcc"
#include "cell_multiphase.tcc"
#include "cell_xmpm.tcc"

#endif  // MPM_CELL_H_