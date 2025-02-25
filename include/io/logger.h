#ifndef MPM_LOGGER_H_
#define MPM_LOGGER_H_

#include <memory>

// Speed log
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

//! MPM namespace
namespace mpm {

// Create an stdout colour sink
const std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> stdout_sink =
    std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

struct Logger {
  // Create a logger for IO
  static const std::shared_ptr<spdlog::logger> io_logger;

  // Create a logger for reading mesh
  static const std::shared_ptr<spdlog::logger> io_mesh_logger;

  // Create a logger for reading ascii mesh
  static const std::shared_ptr<spdlog::logger> io_mesh_ascii_logger;

  // Create a logger for point generator
  static const std::shared_ptr<spdlog::logger> point_generator_logger;

  // Create a logger for MPM
  static const std::shared_ptr<spdlog::logger> mpm_logger;

  // Create a logger for MPM Base
  static const std::shared_ptr<spdlog::logger> mpm_base_logger;

  // Create a logger for MPM Explicit
  static const std::shared_ptr<spdlog::logger> mpm_explicit_logger;

  // Create a logger for MPM Explicit USF
  static const std::shared_ptr<spdlog::logger> mpm_explicit_usf_logger;

  // Create a logger for MPM Explicit USL
  static const std::shared_ptr<spdlog::logger> mpm_explicit_usl_logger;

  // Create a logger for MPM Explicit MUSL
  static const std::shared_ptr<spdlog::logger> mpm_explicit_musl_logger;

  // Create a logger for MPM Implicit
  static const std::shared_ptr<spdlog::logger> mpm_implicit_logger;

  // Create a logger for MPM Implicit Newmark
  static const std::shared_ptr<spdlog::logger> mpm_implicit_newmark_logger;

  // Create a logger for MPM Semi-implicit Navier Stokes
  static const std::shared_ptr<spdlog::logger>
      mpm_semi_implicit_navier_stokes_logger;

  // Create a logger for MPM Explicit Two Phase
  static const std::shared_ptr<spdlog::logger> mpm_explicit_two_phase_logger;

  // Create a logger for MPM Semi-implicit Two Phase
  static const std::shared_ptr<spdlog::logger>
      mpm_semi_implicit_two_phase_logger;

  // Create a logger for Thermo-Mechanical MPM explicit
  static const std::shared_ptr<spdlog::logger>
      mpm_explicit_thermal_logger;

  // Create a logger for Thermo-Mechanical MPM mplicit
  static const std::shared_ptr<spdlog::logger>
      mpm_implicit_thermal_logger;      

};

}  // namespace mpm

#endif  // MPM_LOGGER_H_
