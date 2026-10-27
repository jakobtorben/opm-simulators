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
#include <config.h>
#include <opm/simulators/flow/BlackoilModelConvergenceMonitor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace Opm {

template<class Scalar>
BlackoilModelConvergenceMonitor<Scalar>::
BlackoilModelConvergenceMonitor(const MonitorParams& param)
    : param_(param)
{
}

template<class Scalar>
double BlackoilModelConvergenceMonitor<Scalar>::
distance(const double scaledResidual) const
{
    using Distance = typename MonitorParams::Distance;
    if (param_.distance_ == Distance::Log) {
        return 1.0 + std::max(std::log10(scaledResidual), 0.0);
    }
    return std::max(scaledResidual, 1.0);
}

template<class Scalar>
const typename BlackoilModelConvergenceMonitor<Scalar>::State&
BlackoilModelConvergenceMonitor<Scalar>::
update(const double scaledResidual, const int iteration)
{
    if (iteration == 0) {
        this->reset();
    }

    distances_.push_back(this->distance(scaledResidual));
    const std::size_t k = distances_.size();
    const double dist = distances_.back();

    state_.distance = dist;
    state_.theta = 0.0;
    state_.thetaTarget = 0.0;
    state_.oscillating = false;
    state_.status = 0;
    if (k == 1) {
        // No contraction estimate yet: the first iteration is ok.
        return state_;
    }

    // Contraction factor from a least-squares fit of the last memory + 1
    // distances to a geometric series, log(D_i / D_0) = i log(theta).
    const std::size_t n = std::min(static_cast<std::size_t>(param_.memory_) + 1, k);
    const auto* d = distances_.data() + (k - n);
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        num += i * std::log(d[i] / d[0]);
        den += i * i;
    }
    const double theta = std::exp(num / den);

    // Contraction needed to converge from the previous iteration within the
    // remaining iterations to the target.  With j = k - 1 the zero-based
    // index of this iteration, the remaining iterations are
    // max(target - j + 1 - margin, buffer).
    const int j = static_cast<int>(k) - 1;
    const int numTarget = std::max(param_.target_iterations_ - j + 1 - param_.iteration_margin_,
                                   param_.iteration_buffer_);
    const double prev = distances_[k - 2];
    const double thetaTarget = std::pow(prev, -1.0 / numTarget);

    // The previous iteration is a strict local extremum.
    bool oscillating = false;
    if (k >= 3) {
        constexpr double tol = 1.0e-6;
        const double d1 = distances_[k - 1];
        const double d2 = distances_[k - 2];
        const double d3 = distances_[k - 3];
        oscillating = (d1 - d2 > tol && d3 - d2 > tol) ||
                      (d2 - d1 > tol && d2 - d3 > tol);
    }

    // Classify as good (1) if theta <= max(theta_target, theta_fast), ok (0)
    // if theta <= max(theta_target, theta_slow) and bad (-1) otherwise.
    // Converged iterations are good and oscillating good iterations are ok.
    const double thetaLow = std::max(thetaTarget, static_cast<double>(param_.theta_fast_));
    const double thetaHigh = std::max(thetaTarget, static_cast<double>(param_.theta_slow_));
    const bool ok = thetaLow < theta && theta <= thetaHigh;
    int status = theta <= thetaLow ? 1 : (ok ? 0 : -1);
    if (dist <= 1.0) {
        status = 1;
    }
    if (status == 1 && oscillating) {
        status -= 1;
    }
    // An ok contraction is still too slow if it needs more than twice the
    // target number of iterations to converge from the previous distance.
    // An ok theta implies prev > 1 (otherwise thetaLow = 1 >= thetaHigh) and
    // theta <= thetaHigh < 1, so the estimate is finite and positive.
    if (ok) {
        const double itersLeft = -std::log(prev) / std::log(theta);
        if (itersLeft > 2 * param_.target_iterations_) {
            status -= 1;
        }
    }

    if (status == 1) {
        state_.numViolations = std::max(state_.numViolations - 1, 0);
    }
    else if (status == -1) {
        state_.numViolations += 1;
    }

    state_.theta = theta;
    state_.thetaTarget = thetaTarget;
    state_.oscillating = oscillating;
    state_.status = status;

    return state_;
}

template<class Scalar>
void BlackoilModelConvergenceMonitor<Scalar>::
reset()
{
    distances_.clear();
    state_ = State{};
}

template class BlackoilModelConvergenceMonitor<double>;

#if FLOW_INSTANTIATE_FLOAT
template class BlackoilModelConvergenceMonitor<float>;
#endif

} // namespace Opm
