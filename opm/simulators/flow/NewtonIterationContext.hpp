/*
  Copyright 2026 SINTEF AS.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_NEWTON_ITERATION_CONTEXT_HPP
#define OPM_NEWTON_ITERATION_CONTEXT_HPP

namespace Opm {

/// \brief Context for iteration-dependent decisions in the Newton solver.
///
/// Provides explicit state for iteration-dependent behavior, replacing implicit
/// queries to numIterations(). Key concepts:
///
/// - **Global iteration**: The top-level Newton iteration count (0-based).
///   Used for NUPCOL checks, group controls, gas lift timing.
///
/// - **Local iteration**: Iteration count within a nested solve (NLDD domains).
///   Starts at 0 for each domain solve.
///
/// - **Timestep initialization**: One-time setup that happens at the start of
///   each timestep, independent of iteration number.
///
/// For NLDD, `forLocalSolve()` creates a context where:
/// - `globalIteration` is preserved (for NUPCOL, group controls)
/// - `localIteration` starts fresh at 0
/// - Timestep initialization is skipped (already done globally)
///
struct NewtonIterationContext {
    // ========== State ==========

    /// Current global Newton iteration (0-based).
    /// Never modified by local solves. Used for NUPCOL, group controls, etc.
    int globalIteration = 0;

    /// Current local iteration within a nested solve (0-based).
    /// For global solves, equals globalIteration.
    /// For NLDD domain solves, counts independently from 0.
    int localIteration = 0;

    /// Whether we are inside a domain-local solve (NLDD).
    bool inLocalSolve = false;

    /// Whether timestep initialization has been performed.
    bool timestepInitialized = false;

    // ========== Semantic Queries ==========

    /// Should timestep initialization run?
    /// True only on first global iteration before initialization is done.
    bool needsTimestepInit() const
    {
        return !timestepInitialized && !inLocalSolve;
    }

    /// Is this the first iteration of the global solve (not a local solve)?
    /// Use for one-time-per-timestep logic like storage cache setup.
    bool isFirstGlobalIteration() const
    {
        return globalIteration == 0 && !inLocalSolve;
    }

    /// Are we within the NUPCOL iteration window?
    /// Always uses global iteration regardless of local solve state.
    bool withinNupcol(int nupcol) const
    {
        return globalIteration < nupcol;
    }

    /// Should tolerances be relaxed based on iteration count?
    /// Uses local iteration for local solves, global otherwise.
    bool shouldRelax(int strictIterations) const
    {
        const int iteration = inLocalSolve ? localIteration : globalIteration;
        return iteration >= strictIterations;
    }

    /// Whether inner well iterations (iterateWellEquations) should run.
    /// Skipped during NLDD local solves; gated by iteration count for global Newton.
    bool shouldRunInnerWellIterations(int maxIter) const
    {
        if (inLocalSolve) return false;
        return globalIteration < maxIter;
    }

    // ========== State Mutations ==========

    /// Mark timestep initialization as complete.
    void markTimestepInitialized()
    {
        timestepInitialized = true;
    }

    /// Advance the local iteration counter.
    void advanceLocalIteration()
    {
        localIteration++;
    }

    /// Reset all state for a new timestep.
    void resetForNewTimestep()
    {
        globalIteration = 0;
        localIteration = 0;
        timestepInitialized = false;
        inLocalSolve = false;
    }

    /// Create a context for a domain-local solve.
    /// Preserves global iteration, resets local iteration to 0.
    NewtonIterationContext forLocalSolve() const
    {
        NewtonIterationContext local;
        local.globalIteration = this->globalIteration;
        local.localIteration = 0;
        local.inLocalSolve = true;
        local.timestepInitialized = true; // Local solves never do timestep init
        return local;
    }
};

/// RAII guard for NLDD domain-local iteration context.
/// Sets a local context on the problem; restores the global
/// context on destruction (handles all exit paths and exceptions).
template<class Problem>
class LocalContextGuard {
public:
    LocalContextGuard(Problem& problem, const NewtonIterationContext& globalCtx)
        : problem_(problem)
        , globalCtx_(globalCtx)
        , localCtx_(globalCtx.forLocalSolve())
    {
        problem_.setIterationContext(localCtx_);
    }

    ~LocalContextGuard()
    {
        problem_.setIterationContext(globalCtx_);
    }

    LocalContextGuard(const LocalContextGuard&) = delete;
    LocalContextGuard& operator=(const LocalContextGuard&) = delete;
    LocalContextGuard(LocalContextGuard&&) = delete;
    LocalContextGuard& operator=(LocalContextGuard&&) = delete;

    NewtonIterationContext& context() { return localCtx_; }
    const NewtonIterationContext& context() const { return localCtx_; }

private:
    Problem& problem_;
    const NewtonIterationContext& globalCtx_;
    NewtonIterationContext localCtx_;
};

} // namespace Opm

#endif // OPM_NEWTON_ITERATION_CONTEXT_HPP
