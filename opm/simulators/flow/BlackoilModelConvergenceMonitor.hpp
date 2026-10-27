/*
  Copyright 2013, 2015 SINTEF ICT, Applied Mathematics.
  Copyright 2014, 2015 Dr. Blatt - HPC-Simulation-Software & Services
  Copyright 2014, 2015 Statoil ASA.
  Copyright 2015 NTNU
  Copyright 2015, 2016, 2017 IRIS AS
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

#ifndef OPM_BLACKOILMODEL_CONV_MONITOR_HEADER_INCLUDED
#define OPM_BLACKOILMODEL_CONV_MONITOR_HEADER_INCLUDED

#include <opm/simulators/flow/BlackoilModelParameters.hpp>
#include <opm/simulators/timestepping/ConvergenceReport.hpp>

#include <vector>

namespace Opm {

/// Convergence monitor based on contraction factors.
///
/// The monitor follows the distance from convergence D of the nonlinear
/// iterations of a time step.  The residual measures of the reservoir (MB
/// and CNV per component) and of the wells are scaled by their strict
/// tolerances and pooled into one distance by their maximum (D <= 1 means
/// converged).  Assuming that the distances form a geometric series, the
/// monitor estimates the contraction factor theta from the latest
/// iterations and compares it with the contraction factor needed to
/// converge within the target number of iterations.  Each iteration is
/// classified as good, ok or bad, and the number of violations is
/// increased for bad and decreased for good iterations.  The time step
/// should be cut when the number of violations exceeds the limit.
template <class Scalar>
class BlackoilModelConvergenceMonitor
{
public:
    using MonitorParams = typename BlackoilModelParameters<Scalar>::ConvergenceMonitorParams;
    using State = ConvergenceReport::ConvergenceMonitorState;

    explicit BlackoilModelConvergenceMonitor(const MonitorParams& param);

    /// Analyse a new nonlinear iteration of the current time step.
    ///
    /// \param[in] scaledResidual Largest residual measure scaled by its
    ///   strict tolerance.
    /// \param[in] iteration Zero-based iteration number.  Iteration zero
    ///   starts a new time step.
    /// \return Monitor state after this iteration.
    const State& update(double scaledResidual, int iteration);

    /// Whether the number of violations exceeds the limit.
    bool shouldCut() const
    { return state_.numViolations > param_.violations_cut_; }

    const State& state() const
    { return state_; }

    const MonitorParams& param() const
    { return param_; }

    void reset();

private:
    double distance(double scaledResidual) const;

    MonitorParams param_;
    std::vector<double> distances_{};
    State state_{};
};

} // namespace Opm

#endif
