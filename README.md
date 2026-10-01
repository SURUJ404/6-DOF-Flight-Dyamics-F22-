# F-22-Class 6-DOF Flight Dynamics

**Solution Description, Actuator-Compensation Design & Tested Results**

This document is a Markdown conversion of `f22_dynamics_solution_report.pdf`.

![Reaction](assets/reaction.gif)

**Purpose.** This report documents the supplied F-22-class nonlinear flight-dynamics solution and the executed tests for actuator dynamics and gyroscopic coupling. The implementation is an estimated, research/engineering simulation—not a validated model of the real F-22, whose detailed aerodynamic and propulsion data are not public.

## 1. Problem Being Addressed

The underlying control problem is that an ideal controller can assume commanded control inputs appear immediately at the plant, while real actuators have finite bandwidth. The supplied solution therefore extends a nonlinear aircraft model with explicit actuator states and an open-loop lead compensator. It also adds engine gyroscopic coupling so that rotating engine inertia can contribute to the rotational equations of motion.

The solution follows the same broad methodology as the supplied VTOL/UFO thesis: represent actuator lag explicitly, compensate the lag with a lead design, and evaluate the result in time-domain simulation. The nonlinear propeller-thrust allocation used by the VTOL thesis is not directly transferred to the F-22 model because the latter uses aerodynamic control surfaces rather than differential propeller thrust.

## 2. Overall Solution Architecture

| Stage | Description |
|---|---|
| Command / trim | Elevator, aileron, rudder commands |
| Lead compensator | Open-loop transient compensation; unity DC gain |
| Physical actuator | First-order bandwidth-limited surface dynamics |
| Aircraft model | Nonlinear 6-DOF rigid-body equations + propulsion |
| Gyroscopic coupling | Engine angular momentum reaction torque |
| Integration | Fixed-step RK4 nonlinear simulation |
| Analysis | Trim, linearization, eigenvalues, sanity checks, time response |

## 3. Mathematical Solution

### 3.1 Actuator model

Each control surface is represented by a first-order actuator:

```
dx_act/dt = (u_act,cmd − x_act) / τ
```

The estimated time constants used by the supplied implementation are **0.060 s** for the elevator, **0.035 s** for the aileron, and **0.050 s** for the rudder. These are explicitly treated as generic order-of-magnitude estimates, not measured F-22 actuator data.

### 3.2 Open-loop lead compensation

The compensator is implemented as `C(s) = Kc(s+z)/(s+p)`. The zero is placed to cancel the nominal actuator pole, `z = 1/τ`, while the new pole is placed three times faster, `p = 3/τ`. Unity DC gain is obtained with `Kc = p/z`. This preserves the steady-state command while accelerating the transient response.

| Channel | τ (s) | Raw pole 1/τ (s⁻¹) | Compensated pole 3/τ (s⁻¹) |
|---|---|---|---|
| Elevator | 0.060 | 16.667 | 50.000 |
| Aileron | 0.035 | 28.571 | 85.714 |
| Rudder | 0.050 | 20.000 | 60.000 |

### 3.3 Gyroscopic coupling

The engine rotational inertia is represented by an approximate body-x angular momentum `Hx`. The reaction torque is generated from `τ_gyro = −(ω × H)`. For the simplified body-x momentum model used here, the relevant pitch/yaw terms are:

```
M_gyro = −r·Hx      N_gyro = q·Hx
```

Counter-rotation is supported as a switch. In the counter-rotating case the modeled net engine angular momentum cancels; the co-rotating case is used as a conservative sensitivity test for maximum coupling.

## 4. Executed Test: Actuator Compensation

The supplied comparison script was executed with the F-22-class nonlinear model. The test trims the aircraft at **25,000 ft** and **850 ft/s** with a **4° initial angle-of-attack** condition, applies a **4° elevator step at t = 1.0 s**, and integrates from **0 to 4 s** with a **0.002 s** step. The two cases differ only in whether the lead compensator is enabled.

> **Figure 1.** Tested elevator step response and coupled angle-of-attack response supplied with the solution. *(Figure exists in the source PDF; not reproduced in this conversion.)*

| Metric | Lead compensated | Uncompensated |
|---|---|---|
| Peak actual elevator | 4.099° | 4.099° |
| Overshoot | ≈ 0.0% | ≈ 0.0% |
| 2% settling time | 0.080 s | 0.236 s |
| Relative settling-time change | — | 2.95× slower |

**Observed result:** the compensated elevator reaches the 2% settling band in 0.080 s versus 0.236 s without compensation. That is approximately a **2.95× reduction in settling time**, consistent with the intended 3× bandwidth-speedup design. The plotted α response also starts its coupled transient earlier in the compensated case before converging toward the same long-time behavior.

## 5. Executed Test: Gyroscopic Sensitivity

A second executed test uses a trim condition of **750 ft/s at 20,000 ft** and representative body rates **p = 20°/s, q = 10°/s, r = 15°/s**. Gyroscopic moments are compared against the control moments produced by a representative **5° elevator/rudder** deflection.

| Configuration | Hx (slug·ft²/s) | M_gyro (ft·lb) | M_control (ft·lb) | M ratio | N_gyro (ft·lb) | N_control (ft·lb) | N ratio |
|---|---|---|---|---|---|---|---|
| Counter-rotating | 0 | −0.0 | −418,927 | 0.00% | 0.0 | −139,425 | 0.00% |
| Co-rotating (worst case) | 1,702 | −445.7 | −418,927 | 0.11% | 297.1 | −139,425 | 0.21% |

**Observed result:** the counter-rotating configuration produces zero net modeled engine-x angular momentum in this simplified pair model. In the conservative co-rotating case, the gyroscopic pitch and yaw moments are only **0.11%** and **0.21%**, respectively, of the representative control authority used in the test. The effect is therefore small in this operating regime, although it is retained in the model rather than silently ignored.

## 6. Additional Model Tests

The repository also includes trim, linearization, mode, and public-data sanity checks. These are architecture/order-of-magnitude checks rather than validation against classified or unavailable F-22 flight-test data.

| Test | Executed result |
|---|---|
| Supercruise trim | Mach 1.5 at 40,000 ft; α = 2.18°, elevator = 0.34°, throttle = 0.555; residual = 5.41×10⁻¹⁶ |
| Subsonic cruise trim | Mach 0.85 at 30,000 ft; α = 4.68°, elevator = −0.03°, throttle = 0.326 |
| Corner-velocity estimate | 757 ft/s at 15,000 ft for 9g; approximately 0.72 Mach / 448 KCAS |
| Linearized modes | All reported modes stable/neutral; short-period pair ωₙ = 7.865 rad/s, ζ = 0.409 |
| Phugoid-like pair | ωₙ = 0.054 rad/s, ζ = 0.065 |

## 7. State and Simulation Structure

The baseline aircraft model uses the **Stevens & Lewis-style 13-state convention**: airspeed, angle of attack, sideslip, Euler attitude, body rates, north/east position, altitude, and engine-power state. The actuator extension adds three compensator states and three actual control-surface states, giving **19 states** in the extended implementation.

The nonlinear equations are integrated using fixed-step **fourth-order Runge–Kutta (RK4)**. The trim solver provides an equilibrium operating point; numerical linearization around that point provides A/B matrices for mode inspection and small-disturbance analysis.

## 8. What the Tested Data Demonstrate

- **Actuator dynamics matter:** removing the instantaneous-actuator assumption produces a visibly slower control-surface transient.
- **Lead compensation works in the simulated test:** the measured 2% settling time changes from 0.236 s to 0.080 s for the 4° elevator step.
- **Compensation preserves steady state:** both cases reach the same commanded elevator value with essentially zero measured overshoot in this test.
- **Gyroscopic coupling is retained:** the model can switch between counter-rotating and conservative co-rotating engine configurations.
- **Gyro effect is small in the tested regime:** the worst-case sensitivity run produced 0.11% pitch and 0.21% yaw coupling relative to the selected control-authority references.
- **The aircraft model is an estimated F-22-class model:** the sanity checks establish plausible behavior, not real-aircraft validation.

## 9. Limitations and Engineering Interpretation

The most important limitation is data provenance. Real F-22 aerodynamic derivatives, detailed engine performance, mass properties, and flight-control actuator specifications are not publicly available at the fidelity required for a validated high-fidelity model. Accordingly, the repository uses documented estimates and public geometry/performance figures. The actuator time constants in particular are assumed values, not measured F-22 step-response data. The reported 0.080 s and 0.236 s values are therefore test results of this simulation configuration—not claims about a real F-22 actuator.

The current baseline also does not model thrust-vectoring control, and the aerodynamic coefficient buildup is intended mainly for subsonic-to-low-supersonic, moderate-angle operation. Large-alpha/post-stall/high-Mach behavior should not be interpreted quantitatively.

## 10. Conclusion

The supplied solution successfully turns the actuator-dynamics problem into an explicit nonlinear simulation architecture: control commands pass through a lead compensator and finite-bandwidth actuators before entering the 6-DOF aircraft dynamics, while engine gyroscopic coupling is included as an additional rotational effect. The executed tests show a **2.95× reduction in elevator 2%-settling time** for the selected 4° step, while the worst-case modeled gyro moments remain **below 0.21%** of the representative control authority. Together with the trim, linearization, mode, and sanity checks, these results provide a reproducible engineering demonstration of the proposed solution, with the stated data limitations kept explicit.

---

**Test provenance:** values in this report were taken from an execution of the supplied F-22-class repository's validation/mode/actuator-comparison scripts, together with the user-supplied actuator-response figure. No unverified real-aircraft performance numbers have been substituted for the repository's estimated parameters.
