#ifndef MPM_NODE_XMPM_H_
#define MPM_NODE_XMPM_H_

#include "logger.h"
#include "mutex.h"
#include "nodal_properties.h"
#include "node.h"

namespace mpm {

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

  //! Update the nodal levelset values
  //! \ingroup XMPM
  //! \param[in] the value of the nodal levelset_phi
  void update_levelset_phi(int discontinuity_id, double phi) {
    levelset_phi_[discontinuity_id] += phi;
  };

  //! Update the nodal enriched mass
  //! \ingroup XMPM
  //! \param[in] the value of the enriched mass
  void update_mass_enrich(double mass[3]) {
    for (unsigned int i = 0; i < 3; i++) mass_enrich_[i] += mass[i];
  };

  //! Update the nodal enriched momentum
  //! \ingroup XMPM
  //! \param[in] the value of the enriched momentum
  void update_momentum_enrich(Eigen::Matrix<double, Tdim, 3> momentum) {
    for (unsigned int i = 0; i < 3; i++)
      momentum_enrich_.col(i) += momentum.col(i);
  };

  //! Update the nodal enriched internal_force
  //! \ingroup XMPM
  //! \param[in] the value of the enriched momentum
  virtual void update_internal_force_enrich(
      Eigen::Matrix<double, Tdim, 3> internal_force) {
    for (unsigned int i = 0; i < 3; i++)
      internal_force.col(i) += internal_force.col(i);
  }

  //! Update the nodal enriched external_force
  //! \ingroup XMPM
  //! \param[in] the value of the enriched external_force
  virtual void update_external_force_enrich(
      Eigen::Matrix<double, Tdim, 3> external_force) {
    for (unsigned int i = 0; i < 3; i++)
      external_force.col(i) += external_force.col(i);
  }

  //! Return mass_enrich_ at a given node for a given phase
  //! \ingroup XMPM
  virtual double* mass_enrich() { return mass_enrich_; }
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
  bool compute_momentum_discontinuity(unsigned phase,
                                      double dt) noexcept override;

 private:
  //! mass
  using Node<Tdim, Tdof, Tnphases>::mass_;
  //! External force
  using Node<Tdim, Tdof, Tnphases>::external_force_;
  //! Internal force
  using Node<Tdim, Tdof, Tnphases>::internal_force_;
  //! Momentum
  using Node<Tdim, Tdof, Tnphases>::momentum_;
  //! Discontinuity enrich
  bool dis_enrich_[2];

  //! Nodal levelset values
  double levelset_phi_[2];

  //! Enriched mass
  double mass_enrich_[3];

  //! Enriched momentum
  Eigen::Matrix<double, Tdim, 3> momentum_enrich_;

  //! Enriched internal force
  Eigen::Matrix<double, Tdim, 3> internal_force_enrich_;

  //! Enriched external force
  Eigen::Matrix<double, Tdim, 3> external_force_enrich_;

};  // Node XMPM class
}  // namespace mpm

#include "node_xmpm_branch.tcc"

#endif  // MPM_NODE_XMPM_H_

// nodal_properties_->create_property("normal_unit_vectors_discontinuity",
// nrows,
//                                    1);
// nodal_properties_->create_property("friction_coef", nodes_.size(), 1);
// nodal_properties_->create_property("cohesion", nodes_.size(), 1);
// nodal_properties_->create_property("cohesion_area", nodes_.size(), 1);
// nodal_properties_->create_property("contact_distance", nodes_.size(), 1);