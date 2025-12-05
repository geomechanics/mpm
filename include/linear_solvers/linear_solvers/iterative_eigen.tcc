//! Main solve function - Enhanced version with preconditioner support
template <typename Traits>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
  Eigen::VectorXd x;
  try {
    // Solver start
    auto solver_begin = std::chrono::steady_clock::now();
    if (verbosity_ > 0)
      console_->info("Type: \"{}\", Preconditioner: \"{}\", Begin!",
                     sub_solver_type_, preconditioner_type_);

    // Verbose output for debugging
    if (verbosity_ == 3) {
      console_->info("Matrix properties:");
      console_->info("  - Size: {} x {}", A.rows(), A.cols());
      console_->info("  - Non-zeros: {}", A.nonZeros());
      console_->info("  - Sparsity: {:.2f}%", 
                     (1.0 - static_cast<double>(A.nonZeros()) / 
                      (static_cast<double>(A.rows()) * A.cols())) * 100);
      if (A.rows() <= 10) {
        std::cout << "Coefficient Matrix A:\n" << A << std::endl;
        std::cout << "RHS Vector b:\n" << b.transpose() << std::endl;
      }
    }

    // Solve based on solver type
    if (sub_solver_type_ == "cg") {
      x = solveConjugateGradient(A, b);
    } else if (sub_solver_type_ == "lscg") {
      x = solveLeastSquaresCG(A, b);
    } else if (sub_solver_type_ == "bicgstab") {
      x = solveBiCGSTAB(A, b);
    } else if (sub_solver_type_ == "gmres") {
      x = solveGMRES(A, b);
    } else if (sub_solver_type_ == "minres") {
      x = solveMINRES(A, b);
    } else {
      throw std::runtime_error(
          "Sub solver type is not available! Available sub solver types "
          "implemented in IterativeEigen class are: \"cg\", \"lscg\", "
          "\"bicgstab\", \"gmres\", and \"minres\".\n");
    }

    // Verify solution if requested
    if (verbosity_ >= 2) {
      verifySolution(A, x, b);
    }

    // Solver End
    auto solver_end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        solver_end - solver_begin).count();
    
    if (verbosity_ > 0)
      console_->info(
          "Type: \"{}\", Preconditioner: \"{}\", End! Duration: {} ms.",
          sub_solver_type_, preconditioner_type_, duration_ms);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    throw;
  }
  return x;
}

//! Conjugate Gradient solver with preconditioner support
template <typename Traits>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solveConjugateGradient(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
  
  Eigen::VectorXd x;
  
  if (preconditioner_type_ == "none" || preconditioner_type_ == "identity") {
    // No preconditioner
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, 
                             Eigen::Lower|Eigen::Upper> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "CG");
    
  } else if (preconditioner_type_ == "diagonal" || 
             preconditioner_type_ == "jacobi") {
    // Diagonal/Jacobi preconditioner
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, 
                             Eigen::Lower|Eigen::Upper,
                             Eigen::DiagonalPreconditioner<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "CG+Diagonal");
    
  } else if (preconditioner_type_ == "ilu" || 
             preconditioner_type_ == "ilut") {
    // Incomplete LU preconditioner
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, 
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteLUT<double>> solver;
    configureSolver(solver);
    
    // Configure ILUT parameters if available
    if (preconditioner_type_ == "ilut") {
      solver.preconditioner().setDroptol(drop_tolerance_);
      solver.preconditioner().setFillfactor(fill_factor_);
    }
    
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "CG+ILUT");
    
  } else if (preconditioner_type_ == "ic" || 
             preconditioner_type_ == "icc") {
    // Incomplete Cholesky preconditioner (for SPD matrices)
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, 
                             Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteCholesky<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "CG+IC");
    
  } else {
    throw std::runtime_error(
        "Unknown preconditioner type for CG: " + preconditioner_type_);
  }
  
  return x;
}

//! BiCGSTAB solver with preconditioner support
template <typename Traits>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solveBiCGSTAB(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
  
  Eigen::VectorXd x;
  
  if (preconditioner_type_ == "none" || preconditioner_type_ == "identity") {
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "BiCGSTAB");
    
  } else if (preconditioner_type_ == "diagonal" || 
             preconditioner_type_ == "jacobi") {
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,
                    Eigen::DiagonalPreconditioner<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "BiCGSTAB+Diagonal");
    
  } else if (preconditioner_type_ == "ilu" || 
             preconditioner_type_ == "ilut") {
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,
                    Eigen::IncompleteLUT<double>> solver;
    configureSolver(solver);
    
    if (preconditioner_type_ == "ilut") {
      solver.preconditioner().setDroptol(drop_tolerance_);
      solver.preconditioner().setFillfactor(fill_factor_);
    }
    
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "BiCGSTAB+ILUT");
    
  } else if (preconditioner_type_ == "ic" || 
             preconditioner_type_ == "icc") {
    // Incomplete Cholesky can also be used with BiCGSTAB for symmetric problems
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,
                    Eigen::IncompleteCholesky<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "BiCGSTAB+IC");
    
  } else {
    throw std::runtime_error(
        "Unknown preconditioner type for BiCGSTAB: " + preconditioner_type_);
  }
  
  return x;
}

//! Least Squares Conjugate Gradient solver with preconditioner support
template <typename Traits>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solveLeastSquaresCG(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
  
  Eigen::VectorXd x;
  
  if (preconditioner_type_ == "none" || preconditioner_type_ == "identity") {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "LSCG");
    
  } else if (preconditioner_type_ == "diagonal" || 
             preconditioner_type_ == "jacobi") {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>,
                                         Eigen::DiagonalPreconditioner<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "LSCG+Diagonal");
    
  } else if (preconditioner_type_ == "ilu" || 
             preconditioner_type_ == "ilut") {
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>,
                                         Eigen::IncompleteLUT<double>> solver;
    configureSolver(solver);
    
    if (preconditioner_type_ == "ilut") {
      solver.preconditioner().setDroptol(drop_tolerance_);
      solver.preconditioner().setFillfactor(fill_factor_);
    }
    
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "LSCG+ILUT");
    
  } else {
    throw std::runtime_error(
        "Unknown preconditioner type for LSCG: " + preconditioner_type_);
  }
  
  return x;
}

//! GMRES solver with preconditioner support
template <typename Traits>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solveGMRES(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
  
  Eigen::VectorXd x;
  
  if (preconditioner_type_ == "none" || preconditioner_type_ == "identity") {
    Eigen::GMRES<Eigen::SparseMatrix<double>> solver;
    configureSolver(solver);
    if (restart_iterations_ > 0) {
      solver.set_restart(restart_iterations_);
    }
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "GMRES");
    
  } else if (preconditioner_type_ == "diagonal" || 
             preconditioner_type_ == "jacobi") {
    Eigen::GMRES<Eigen::SparseMatrix<double>,
                 Eigen::DiagonalPreconditioner<double>> solver;
    configureSolver(solver);
    if (restart_iterations_ > 0) {
      solver.set_restart(restart_iterations_);
    }
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "GMRES+Diagonal");
    
  } else if (preconditioner_type_ == "ilu" || 
             preconditioner_type_ == "ilut") {
    Eigen::GMRES<Eigen::SparseMatrix<double>,
                 Eigen::IncompleteLUT<double>> solver;
    configureSolver(solver);
    if (restart_iterations_ > 0) {
      solver.set_restart(restart_iterations_);
    }
    
    if (preconditioner_type_ == "ilut") {
      solver.preconditioner().setDroptol(drop_tolerance_);
      solver.preconditioner().setFillfactor(fill_factor_);
    }
    
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "GMRES+ILUT");
    
  } else {
    throw std::runtime_error(
        "Unknown preconditioner type for GMRES: " + preconditioner_type_);
  }
  
  return x;
}

//! MINRES solver (for symmetric matrices)
template <typename Traits>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solveMINRES(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
  
  Eigen::VectorXd x;
  
  if (preconditioner_type_ == "none" || preconditioner_type_ == "identity") {
    Eigen::MINRES<Eigen::SparseMatrix<double>,
                  Eigen::Lower|Eigen::Upper,
                  Eigen::IdentityPreconditioner> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "MINRES");
    
  } else if (preconditioner_type_ == "diagonal" || 
             preconditioner_type_ == "jacobi") {
    Eigen::MINRES<Eigen::SparseMatrix<double>,
                  Eigen::Lower|Eigen::Upper,
                  Eigen::DiagonalPreconditioner<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "MINRES+Diagonal");
    
  } else if (preconditioner_type_ == "ic" || 
             preconditioner_type_ == "icc") {
    Eigen::MINRES<Eigen::SparseMatrix<double>,
                  Eigen::Lower|Eigen::Upper,
                  Eigen::IncompleteCholesky<double>> solver;
    configureSolver(solver);
    solver.compute(A);
    x = solveWithInitialGuess(solver, b);
    reportSolverStatus(solver, "MINRES+IC");
    
  } else {
    throw std::runtime_error(
        "Unknown preconditioner type for MINRES: " + preconditioner_type_);
  }
  
  return x;
}

//! Configure solver parameters
template <typename Traits>
template <typename SolverType>
void mpm::IterativeEigen<Traits>::configureSolver(SolverType& solver) {
  solver.setMaxIterations(max_iter_);
  solver.setTolerance(tolerance_);
}

//! Solve with initial guess support
template <typename Traits>
template <typename SolverType>
Eigen::VectorXd mpm::IterativeEigen<Traits>::solveWithInitialGuess(
    SolverType& solver, const Eigen::VectorXd& b) {
  
  Eigen::VectorXd x;
  
  if (use_initial_guess_ && initial_guess_.size() == b.size()) {
    // Use provided initial guess
    x = solver.solveWithGuess(b, initial_guess_);
    if (verbosity_ >= 2) {
      console_->info("Using provided initial guess");
    }
  } else if (use_last_solution_ && last_solution_.size() == b.size()) {
    // Use last solution as initial guess
    x = solver.solveWithGuess(b, last_solution_);
    if (verbosity_ >= 2) {
      console_->info("Using previous solution as initial guess");
    }
  } else {
    // Default: zero initial guess
    x = solver.solve(b);
    if (verbosity_ >= 2 && (use_initial_guess_ || use_last_solution_)) {
      console_->info("Using default zero initial guess");
    }
  }
  
  // Store solution for potential reuse
  if (use_last_solution_) {
    last_solution_ = x;
  }
  
  return x;
}

//! Report solver status
template <typename Traits>
template <typename SolverType>
void mpm::IterativeEigen<Traits>::reportSolverStatus(
    const SolverType& solver, const std::string& solver_name) {
  
  if (verbosity_ >= 1) {
    console_->info("Solver: {}", solver_name);
    console_->info("  Iterations:     {}", solver.iterations());
    console_->info("  Estimated error: {:.6e}", solver.error());
    
    if (solver.iterations() >= max_iter_) {
      console_->warn("  Maximum iterations reached!");
    }
  }
  
  if (solver.info() != Eigen::Success) {
    std::string error_msg = "Fail to solve linear system with " + solver_name + "!\n";
    
    if (solver.info() == Eigen::NoConvergence) {
      error_msg += "  Reason: No convergence within maximum iterations.\n";
      error_msg += "  Consider: increasing max_iter, relaxing tolerance, ";
      error_msg += "or using a different preconditioner.\n";
    } else if (solver.info() == Eigen::NumericalIssue) {
      error_msg += "  Reason: Numerical issues detected.\n";
      error_msg += "  Consider: checking matrix conditioning, ";
      error_msg += "scaling the problem, or using a more robust solver.\n";
    } else if (solver.info() == Eigen::InvalidInput) {
      error_msg += "  Reason: Invalid input matrix or vector.\n";
      error_msg += "  Check: matrix dimensions, symmetry (for CG/MINRES), ";
      error_msg += "or positive definiteness (for CG).\n";
    }
    
    throw std::runtime_error(error_msg);
  }
}

//! Verify solution quality
template <typename Traits>
void mpm::IterativeEigen<Traits>::verifySolution(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& b) {
  
  Eigen::VectorXd residual = A * x - b;
  double residual_norm = residual.norm();
  double relative_residual = residual_norm / b.norm();
  
  console_->info("Solution verification:");
  console_->info("  - Solution norm: {:.6e}", x.norm());
  console_->info("  - Absolute residual norm: {:.6e}", residual_norm);
  console_->info("  - Relative residual norm: {:.6e}", relative_residual);
  
  if (relative_residual > tolerance_ * 10) {
    console_->warn("  - Solution may not have converged properly!");
    console_->warn("  - Consider adjusting solver parameters.");
  } else if (relative_residual < tolerance_ / 10) {
    console_->info("  - Excellent convergence achieved!");
  }
  
  // Check for NaN or Inf in solution
  if (!x.allFinite()) {
    console_->error("Solution contains NaN or Inf values!");
    throw std::runtime_error("Invalid solution: contains NaN or Inf");
  }
}