#ifndef MPM_COHESION_CONSTRAINT_H_
#define MPM_COHESION_CONSTRAINT_H_

#include "data_types.h"

namespace mpm {

//! CohesionConstraint class to store cohesion constraint on a set
//! \brief CohesionConstraint class to store a constraint on a set
//! \details CohesionConstraint stores the constraint as a static value
class CohesionConstraint {
 public:
  // Constructor
  //! \param[in] setid  set id
  //! \param[in] dir Direction of constraint load (normal)
  //! \param[in] sign_n Sign of normal vector
  //! \param[in] cohesion Constraint cohesion
  //! \param[in] h_min Characteristic length (cell height)
  //! \param[in] nposition Nodal location, nposition, along boundary
  CohesionConstraint(int setid, unsigned dir, int sign_n, double cohesion,
                     double h_min, int nposition = 0)
      : setid_{setid},
        dir_{dir},
        sign_n_{sign_n},
        cohesion_{cohesion},
        h_min_{h_min},
        nposition_{nposition} {};

  // Set id
  int setid() const { return setid_; }

  // Direction
  unsigned dir() const { return dir_; }

  // Sign of normal direction
  int sign_n() const { return sign_n_; }

  // Return cohesion
  double cohesion() const { return cohesion_; }

  // Cell height
  double h_min() const { return h_min_; }

  // Return nposition
  int nposition() const { return nposition_; }
  // 0: None (no nposition is specified)
  // 1: Corner (nodes at boundary corners)
  // 2: Edge (nodes along boundary edges)
  // 3: Face (nodes on boundary faces)

 private:
  // ID
  int setid_;
  // Direction (normal)
  unsigned dir_;
  // Sign of normal direction
  int sign_n_;
  // Cohesion
  double cohesion_;
  // Cell height
  double h_min_;
  // Node nposition
  int nposition_;
};
}  // namespace mpm
#endif  // MPM_COHESION_CONSTRAINT_H_