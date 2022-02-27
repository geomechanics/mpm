#ifndef MPM_NODE_XMPM_H_
#define MPM_NODE_XMPM_H_

#include "logger.h"
#include "mutex.h"
#include "nodal_properties.h"
#include "node.h"

namespace mpm {

enum NodeEnrichType {
  regular = 0,
  single_enriched = 1,
  double_enriched = 2,
  multiple_enriched = 3
};

// Node XMPM class
//! \brief class that stores the information about XMPM nodes
//! \details Node XMPM class: id_ and coordinates.
//! \tparam Tdim Dimension
//! \tparam Tdof Degrees of Freedom
//! \tparam Tnphases Number of phases
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
class NodeXMPM : public Node<Tdim, Tdof, Tnphases> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id, coordinates and dof
  //! \param[in] id Node id
  //! \param[in] coord coordinates of the node
  NodeXMPM(Index id, const VectorDim& coord);

  //! Virtual destructor
  ~NodeXMPM() override{};

  //! Delete copy constructor
  NodeXMPM(const NodeXMPM<Tdim, Tdof, Tnphases>&) = delete;

  //! Delete assignement operator
  NodeXMPM& operator=(const NodeXMPM<Tdim, Tdof, Tnphases>&) = delete;

  //! Initialise nodal properties
  void initialise() noexcept override;

  //! Return the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the discontinuity_id
  double levelset_phi(int dis_id) { return levelset_phi_[dis_id]; };

  //! Update the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the value of the nodal levelset_phi
  //! \param[in] the discontinuity id
  void update_levelset_phi(double phi, int dis_id) {
    levelset_phi_[dis_id] += phi;
  };

  //! assign the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the value of the nodal levelset_phi
  //! \param[in] the discontinuity id
  void assign_levelset_phi(double phi, int dis_id) {
    levelset_phi_[dis_id] = phi;
  };

  //! Update the nodal enriched mass
  //! \ingroup XMPM
  //! \param[in] the value of the enriched mass
  void update_mass_enrich(Eigen::Matrix<double, 3, 1> mass) {
    mass_enrich_ += mass;
  };

  //! Update the nodal enriched momentum
  //! \ingroup XMPM
  //! \param[in] the value of the enriched momentum
  void update_momentum_enrich(Eigen::Matrix<double, Tdim, 3> momentum) {
    momentum_enrich_ += momentum;
  };

  //! Update the nodal enriched internal_force
  //! \ingroup XMPM
  //! \param[in] the value of the enriched momentum
  virtual void update_internal_force_enrich(
      Eigen::Matrix<double, Tdim, 3> internal_force) {
    internal_force_enrich_ += internal_force;
  }

  //! Update the nodal enriched external_force
  //! \ingroup XMPM
  //! \param[in] the value of the enriched external_force
  virtual void update_external_force_enrich(
      Eigen::Matrix<double, Tdim, 3> external_force) {
    external_force_enrich_ += external_force;
  }

  //! Update the nodal mass_h_
  //! \ingroup XMPM
  //! \param[in] the value of mass_h_
  void update_mass_h(double mass_h) { mass_h_ += mass_h; }

  //! Initialise the nodal mass_h_
  //! \ingroup XMPM
  void initialise_mass_h() { mass_h_ = 0; }
  //! Initialise the nodal mass_
  //! \param[in] phase Index corresponding to the phase
  //! \ingroup XMPM
  void initialise_mass(unsigned phase) { mass_(phase) = 0; }
  //! Return mass_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  Eigen::Matrix<double, 3, 1> mass_enrich() { return mass_enrich_; }
  //! Return momentum_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  virtual Eigen::Matrix<double, Tdim, 3> momentum_enrich() {
    return momentum_enrich_;
  }
  //! Return internal_force_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  Eigen::Matrix<double, Tdim, 3> internal_force_enrich() {
    return internal_force_enrich_;
  }
  //! Return external_force_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  Eigen::Matrix<double, Tdim, 3> external_force_enrich() {
    return external_force_enrich_;
  }

  //! Compute momentum for discontinuity
  //! \ingroup XMPM
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  bool compute_momentum_discontinuity(unsigned phase, double dt) override;

  //! Compute momentum for discontinuity with cundall damping factor
  //! \ingroup XMPM
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] dt Timestep in analysis
  //! \param[in] damping_factor Damping factor
  bool compute_momentum_discontinuity_cundall(unsigned phase, double dt,
                                              double damping_factor) override;

  //! Determine node type
  //! \param[in] the discontinuity_id
  //! \ingroup XMPM
  void determine_node_type(int discontinuity_id) override;

  //! Return the discontinuity id at nodes
  //! \retval discontinuity_id_
  Eigen::Matrix<int, 2, 1> discontinuity_id() { return discontinuity_id_; }

  //! Return enrich_type
  //! \retval enrich_type_
  int enrich_type() { return enrich_type_; }

  //! Add a cell id
  void add_cell_id(Index id) override;

  //! Apply self-contact force of the discontinuity
  //! \param[in] dt Time-step
  void self_contact_discontinuity(double dt) override;

  //! Initialise shared pointer to nodal properties pool for discontinuity
  //! \param[in] prop_id Property id in the nodal property pool
  //! \param[in] nodal_properties Shared pointer to nodal properties pool
  void initialise_discontinuity_property_handle(
      unsigned prop_id,
      std::shared_ptr<mpm::NodalProperties> property_handle) override;

  //! Assign whether the node is enriched
  //! \param[in] discontinuity_enrich_: true or false
  //! \param[in] discontinuity id
  void assign_discontinuity_enrich(bool enrich, unsigned dis_id);

  //! Return whether the node is enriched
  //! \param[in] discontinuity id
  bool discontinuity_enrich(unsigned dis_id) const {
    bool status = false;
    if (enrich_type_ == mpm::NodeEnrichType::regular) return status;
    if (enrich_type_ == mpm::NodeEnrichType::single_enriched &&
        discontinuity_id_[0] == dis_id) {
      status = true;
    } else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {
      if (discontinuity_id_[0] == dis_id || discontinuity_id_[1] == dis_id)
        status = true;
    }
    return status;
  };

  //! Update nodal property at the nodes from particle for discontinuity
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] property Property name
  //! \param[in] property_value Property quantity from the particles in the cell
  //! \param[in] discontinuity_id Id of the material within the property data
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  void update_discontinuity_property(bool update, const std::string& property,
                                     const Eigen::MatrixXd& property_value,
                                     unsigned discontinuity_id,
                                     unsigned nprops);

  //! assign nodal property at the nodes from particle for discontinuity
  //! \param[in] update A boolean to update (true) or assign (false)
  //! \param[in] property Property name
  //! \param[in] property_value Property quantity from the particles in the cell
  //! \param[in] discontinuity_id Id of the material within the property data
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  void assign_discontinuity_property(bool update, const std::string& property,
                                     const Eigen::MatrixXd& property_value,
                                     unsigned discontinuity_id,
                                     unsigned nprops);

  //! Return data in the nodal discontinuity properties map at a specific index
  //! \param[in] property Property name
  //! \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  Eigen::MatrixXd discontinuity_property(const std::string& property,
                                         unsigned nprops = 1) override;

  //! Return the discontinuity_prop_id
  unsigned discontinuity_prop_id() const { return discontinuity_prop_id_; };

  //! Apply velocity constraints
  void apply_velocity_constraints() override;

  //! Return normal at a given node for a given phase
  //! \param[in] the discontinuity id
  VectorDim normal(unsigned dis_id) { return normal_[dis_id]; }

  //! Assign normal at a given node
  //! \ingroup XMPM
  //! \param[in] the normal direction
  //! \param[in] the discontinuity id
  void assign_normal(VectorDim normal, unsigned dis_id) {
    normal_[dis_id] = normal;
  }

  //! Reset the size of the discontinuity
  //! \param[in] the number of the discontinuity
  void reset_discontinuity_size(int size) {
    levelset_phi_.resize(size, 0);
    normal_.resize(size, VectorDim::Zero());
  }

  //! Assign contact condition by contact distance
  //! \ingroup XMPM
  //! \param[in] the contact id
  //! \param[in] contact or not
  void assign_contact(unsigned id, bool status) {
    contact_detection_[id] = status;
  }

  //! Assign friction coefficient
  //! \ingroup XMPM
  //! \param[in] the friction coefficient
  //! \param[in] the discontinuity id
  void assign_friction_coef(double friction_coef, unsigned dis_id) {
    if (enrich_type_ == mpm::NodeEnrichType::regular) return;
    if (enrich_type_ == mpm::NodeEnrichType::single_enriched &&
        discontinuity_id_[0] == dis_id) {
      friction_coef_[0] = friction_coef;
    } else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {
      if (discontinuity_id_[0] == dis_id) friction_coef_[0] = friction_coef;
      if (discontinuity_id_[1] == dis_id) friction_coef_[1] = friction_coef;
    }
  }

  //! Assign cohesion
  //! \ingroup XMPM
  //! \param[in] the cohesion
  //! \param[in] the discontinuity id
  void assign_cohesion(double cohesion, unsigned dis_id) {
    if (enrich_type_ == mpm::NodeEnrichType::regular) return;
    if (enrich_type_ == mpm::NodeEnrichType::single_enriched &&
        discontinuity_id_[0] == dis_id) {
      cohesion_[0] = cohesion;
    } else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {
      if (discontinuity_id_[0] == dis_id) cohesion_[0] = cohesion;
      if (discontinuity_id_[1] == dis_id) cohesion_[1] = cohesion;
    }
  }

  //! Update cohesion area
  //! \ingroup XMPM
  //! \param[in] the cohesion area
  //! \param[in] the discontinuity id
  void update_cohesion_area(double cohesion_area, unsigned dis_id) {
    if (enrich_type_ == mpm::NodeEnrichType::regular) return;
    if (enrich_type_ == mpm::NodeEnrichType::single_enriched &&
        discontinuity_id_[0] == dis_id) {
      cohesion_area_[0] += cohesion_area;
    } else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {
      if (discontinuity_id_[0] == dis_id) cohesion_area_[0] += cohesion_area;
      if (discontinuity_id_[1] == dis_id) cohesion_area_[1] += cohesion_area;
    }
  }

  //! Return  connected cells
  //! \retval cells_ connected cells
  std::vector<Index> cells() { return cells_; }

 private:
  //! mass
  using Node<Tdim, Tdof, Tnphases>::mass_;
  //! External force
  using Node<Tdim, Tdof, Tnphases>::external_force_;
  //! Internal force
  using Node<Tdim, Tdof, Tnphases>::internal_force_;
  //! Momentum
  using Node<Tdim, Tdof, Tnphases>::momentum_;
  //! Logger
  using Node<Tdim, Tdof, Tnphases>::console_;
  //! Nodal property pool
  using Node<Tdim, Tdof, Tnphases>::property_handle_;
  //! Mutex
  using Node<Tdim, Tdof, Tnphases>::node_mutex_;

  //! cells ids including the node
  std::vector<Index> cells_;
  // need to be done
  bool discontinuity_enrich_{false};
  //! nodal discontinuity property id
  unsigned discontinuity_prop_id_{std::numeric_limits<unsigned>::max()};
  //! mass*h
  double mass_h_{0};
  //! cohesion
  Eigen::Matrix<double, 2, 1> cohesion_;
  //! cohesion area
  Eigen::Matrix<double, 2, 1> cohesion_area_;
  //! frictional coefficient
  Eigen::Matrix<double, 2, 1> friction_coef_;
  //! Discontinuity enrich
  Eigen::Matrix<int, 2, 1> discontinuity_id_;
  //! Nodal levelset values
  std::vector<double> levelset_phi_;
  //! Nodal levelset values
  std::vector<VectorDim> normal_;
  //! Enriched mass
  Eigen::Matrix<double, 3, 1> mass_enrich_;
  //! Enriched momentum
  Eigen::Matrix<double, Tdim, 3> momentum_enrich_;
  //! Enriched internal force
  Eigen::Matrix<double, Tdim, 3> internal_force_enrich_;
  //! Enriched external force
  Eigen::Matrix<double, Tdim, 3> external_force_enrich_;
  //! Enrich type of the node
  mpm::NodeEnrichType enrich_type_;
  //! Contact detection by distance
  Eigen::Matrix<bool, 6, 1> contact_detection_;
};  // Node XMPM class
}  // namespace mpm

#include "node_xmpm.tcc"

#endif  // MPM_NODE_XMPM_H_
