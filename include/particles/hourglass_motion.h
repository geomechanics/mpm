#ifndef MPM_HOURGLASS_MOTION_H_
#define MPM_HOURGLASS_MOTION_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace mpm {

//! Non-inertial frame state at one instant (hourglass frame B, MPM axes).
//! Pseudo acceleration of a grain with velocity v and MPM position x:
//!     a = M1 * v + M2 * x + c
//! and gravity expressed in frame B is g (already rotated by Q(t)).
//!     M1 = 2 Qdot Q^T
//!     M2 = Qddot Q^T + 2 Qdot Qdot^T
//!     c  = yddot - M1 ydot - M2 (y + pivot)
struct FrameState {
  Eigen::Matrix3d M1{Eigen::Matrix3d::Zero()};
  Eigen::Matrix3d M2{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d c{Eigen::Vector3d::Zero()};
  Eigen::Vector3d g{Eigen::Vector3d::Zero()};
};

//! Time table written by exportHourglassMotion.m
//! Format: '#' comment lines, then one row per sample:
//!     t  M1(9, row-major)  M2(9, row-major)  c(3)  g(3)       (25 numbers)
//! Linear interpolation inside the table, last row held after it
//! (exact for a single flip: all terms are constant once the flip is over).
class HourglassMotion {
 public:
  //! Read the table
  void load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open())
      throw std::runtime_error("HourglassMotion: cannot open " + filename);
    times_.clear();
    states_.clear();
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      double v[25];
      for (int i = 0; i < 25; ++i)
        if (!(ss >> v[i]))
          throw std::runtime_error(
              "HourglassMotion: expected 25 numbers per row in " + filename);
      FrameState s;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          s.M1(i, j) = v[1 + 3 * i + j];
          s.M2(i, j) = v[10 + 3 * i + j];
        }
      for (int i = 0; i < 3; ++i) {
        s.c(i) = v[19 + i];
        s.g(i) = v[22 + i];
      }
      if (!times_.empty() && v[0] <= times_.back())
        throw std::runtime_error("HourglassMotion: times must increase");
      times_.push_back(v[0]);
      states_.push_back(s);
    }
    if (times_.size() < 2)
      throw std::runtime_error("HourglassMotion: table needs >= 2 rows");
  }

  //! Frame state at time t [s]
  FrameState at(double t) const {
    if (t <= times_.front()) return states_.front();
    if (t >= times_.back()) return states_.back();
    const auto it = std::upper_bound(times_.begin(), times_.end(), t);
    const std::size_t i1 = static_cast<std::size_t>(it - times_.begin());
    const std::size_t i0 = i1 - 1;
    const double w = (t - times_[i0]) / (times_[i1] - times_[i0]);
    FrameState s;
    s.M1 = (1. - w) * states_[i0].M1 + w * states_[i1].M1;
    s.M2 = (1. - w) * states_[i0].M2 + w * states_[i1].M2;
    s.c = (1. - w) * states_[i0].c + w * states_[i1].c;
    s.g = (1. - w) * states_[i0].g + w * states_[i1].g;
    return s;
  }

  double t_end() const { return times_.back(); }
  std::size_t rows() const { return times_.size(); }

 private:
  std::vector<double> times_;
  std::vector<FrameState> states_;
};

}  // namespace mpm

#endif  // MPM_HOURGLASS_MOTION_H_
