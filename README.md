# Quaternion-Based UAV Attitude Control using MATLAB/Simulink

**Semester 5 Project — Introduction to Data Driven Control of Drones**  
**Group:** CD5  
**Repository:** `Drones_S5_CD5_UAV_Attitude_Control_Quaternion`

> This repository contains the implementation, simulation models, results, presentation, and report for the Semester 5 project on UAV attitude control using quaternion-based representation and MATLAB/Simulink.

---

## Team Members

| Name | Roll Number | Email |
|---|---|---|
| *To be added* | *To be added* | *To be added* |
| *To be added* | *To be added* | *To be added* |
| *To be added* | *To be added* | *To be added* |
| *To be added* | *To be added* | *To be added* |

---

## Title

**Quaternion-Based Attitude Control of a Quadrotor UAV Using MATLAB/Simulink and UAV Toolbox**

---

## Abstract

This project develops and simulates an attitude-control system for a quadrotor UAV using quaternion-based orientation representation. The system is implemented in MATLAB/Simulink and uses UAV Toolbox components, including the **Simulation 3D UAV Vehicle** block, to visualize the simulated vehicle response. A quaternion reference is supplied to the controller, the measured attitude is represented as a quaternion, and the controller generates the required control torque. The quaternion dynamics and angular dynamics are then integrated to obtain the UAV attitude response. The project also includes quaternion-to-yaw-pitch-roll conversion for interpreting the simulated orientation.

The simulation was validated using commanded attitude changes, including a **90-degree roll command**, for which the 3D UAV visualization showed the expected rotation. The repository documents the mathematical formulation, Simulink architecture, MATLAB Function blocks, simulation results, and supporting project material.

---

## Introduction

Attitude control is a fundamental part of autonomous UAV flight. A quadrotor must control its orientation about the three body axes: roll, pitch, and yaw. Accurate attitude regulation is required for stable flight, trajectory tracking, and autonomous operation.

Euler angles are intuitive but can suffer from singularities such as gimbal lock. Quaternions provide a compact four-parameter representation of three-dimensional orientation and avoid this singularity for attitude representation.

In this project, quaternion-based attitude control is implemented in Simulink. The simulated UAV is connected to the UAV Toolbox 3D visualization environment so that the controller response can be observed directly as a quadrotor motion.

---

## Objectives

1. Develop a quaternion-based representation of UAV attitude.
2. Formulate quaternion attitude dynamics.
3. Model the quadrotor angular dynamics.
4. Design a proportional attitude controller with angular-rate feedback.
5. Convert roll, pitch, and yaw commands into quaternions.
6. Connect the controller to the UAV Toolbox 3D UAV Vehicle block.
7. Validate the controller using commanded attitude changes.
8. Analyze the simulated quaternion response using Scope plots and 3D visualization.

---

## Software and Tools

- MATLAB
- Simulink
- UAV Toolbox
- Simulation 3D UAV Vehicle
- MATLAB Function blocks
- Simulink Scope

---

## Methodology

The overall simulation follows this structure:

```text
Desired Roll/Pitch/Yaw
          |
          v
 Euler Angles → Quaternion Reference (q_ref)
          |
          v
   Quaternion Controller
          |
          v
       Torque (τ)
          |
          v
   Angular Dynamics
          |
          v
   Angular Velocity (ω)
          |
          v
 Quaternion Dynamics
          |
          v
 Measured Quaternion (q_m)
          |
          +------> Feedback to Controller
          |
          v
 Quaternion → Yaw/Pitch/Roll
          |
          v
 Simulation 3D UAV Vehicle
```

### 1. Quaternion Representation

The UAV attitude is represented by a unit quaternion

$$
q = \begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{T}.
$$

The initial quaternion used in the simulation is

$$
q_0 = \begin{bmatrix}1 & 0 & 0 & 0\end{bmatrix}^{T},
$$

which represents zero initial rotation.

### 2. Quaternion Reference

A desired roll, pitch, and yaw command is converted into a reference quaternion before being supplied to the controller. This allows the user to specify attitude commands in intuitive angular units while keeping the controller quaternion-based.

For individual rotations, the corresponding quaternions are formed from half-angle sine and cosine terms and combined using quaternion multiplication.

### 3. Quaternion Error

The controller calculates the attitude error between the reference quaternion and measured quaternion. The error quaternion is used to obtain the vector component associated with the attitude error.

The controller implemented in the MATLAB Function block follows the form

$$
\tau = -K_q e_q - K_\omega \omega,
$$

where:

- $\tau$ is the control torque,
- $K_q$ is the quaternion-error gain,
- $e_q$ is the vector part of the quaternion error,
- $K_\omega$ is the angular-rate feedback gain,
- $\omega$ is the angular velocity vector.

### 4. Angular Dynamics

The rigid-body angular dynamics are modeled using

$$
I\dot{\omega}=\tau,
$$

for the simplified simulation model used in the project.

The inertia matrix used in the current model is

$$
I =
\begin{bmatrix}
1.4\times10^{-5} & 0 & 0\\
0 & 1.4\times10^{-5} & 0\\
0 & 0 & 2.2\times10^{-5}
\end{bmatrix}.
$$

Therefore,

$$
\dot{\omega}=I^{-1}\tau.
$$

### 5. Quaternion Dynamics

The angular velocity is supplied to the quaternion dynamics block. The resulting quaternion derivative is integrated to obtain the measured quaternion state.

The quaternion state is initialized as

$$
q(0)=\begin{bmatrix}1&0&0&0\end{bmatrix}^{T}.
$$

### 6. Quaternion-to-Euler Conversion

For visualization and interpretation, the measured quaternion is converted to yaw, pitch, and roll.

For $q=[w,x,y,z]^T$:

$$
\phi = \operatorname{atan2}\left(2(wx+yz),1-2(x^2+y^2)\right),
$$

$$
\theta = \arcsin\left(2(wy-zx)\right),
$$

$$
\psi = \operatorname{atan2}\left(2(wz+xy),1-2(y^2+z^2)\right).
$$

Here $\phi$, $\theta$, and $\psi$ represent roll, pitch, and yaw respectively.

---

## Simulink Model

The main Simulink model contains:

- Quaternion reference input
- P2 attitude controller
- Angular dynamics
- Angular velocity integrator
- Quaternion dynamics
- Quaternion state integrator
- Quaternion-to-yaw/pitch/roll conversion
- Simulation 3D Scene Configuration
- Simulation 3D UAV Vehicle
- Scope blocks for monitoring the response

### 3D UAV Visualization

The **Simulation 3D UAV Vehicle** block is configured as a **Quadrotor**. The translation input controls the vehicle position while the rotation input controls the UAV orientation.

The current simulation uses the UAV Toolbox 3D environment for visual validation of the attitude controller.

---

## Results

The controller was tested with different quaternion references.

### 90-Degree Roll Test

A reference corresponding to approximately 90 degrees of roll was supplied to the controller. The Scope response converged to the commanded attitude, and the Simulation 3D UAV visualization showed the quadrotor rotating by approximately 90 degrees.

This demonstrates that the quaternion reference, controller, angular dynamics, quaternion dynamics, and 3D UAV visualization are connected correctly for the tested roll command.

### Result Summary

| Test | Command | Observation | Status |
|---|---|---|---|
| Initial attitude | $q=[1,0,0,0]^T$ | UAV remains at initial orientation | Completed |
| Small attitude command | Quaternion reference | Controlled response observed | Completed |
| 90° roll | Roll = 90° | UAV rotates approximately 90° | Completed |

### Figures and Videos

Project screenshots, Scope plots, and simulation recordings will be stored in the `results/` directory.

---

## Repository Structure

```text
Drones_S5_CD5_UAV_Attitude_Control_Quaternion/
│
├── README.md
├── simulink/
│   ├── main_model.slx
│   └── matlab_functions/
│
├── src/
│   ├── euler_to_quaternion.m
│   ├── quat_to_ypr.m
│   ├── P2_Controller.m
│   └── Angular_Dynamics.m
│
├── results/
│   ├── figures/
│   └── videos/
│
├── docs/
│   ├── report/
│   └── images/
│
├── presentation/
│   └── project_presentation.pptx
│
└── references/
    └── references.md
```

---

## Future Work

- Extend the controller from attitude-only control to full position and attitude control.
- Include the complete nonlinear rigid-body rotational dynamics.
- Add realistic quadrotor actuator and motor dynamics.
- Test roll, pitch, and yaw commands systematically.
- Evaluate transient response, settling time, overshoot, and steady-state error.
- Integrate sensor models and measurement noise.
- Compare quaternion-based control with Euler-angle-based control.

---

## Conclusion

A quaternion-based attitude-control framework for a quadrotor UAV was developed in MATLAB/Simulink. The model combines quaternion representation, attitude-error feedback, angular dynamics, quaternion dynamics, and UAV Toolbox 3D visualization. The implemented simulation successfully demonstrated controlled attitude response, including a 90-degree roll command in the 3D UAV environment.

The project provides a foundation for extending the simulation toward more complete quadrotor dynamics, trajectory tracking, and autonomous UAV control.

---

## References

> Add the exact base paper and other sources used by the team here. Do not replace the team's selected base paper with unrelated references.

1. **Base Paper:** *To be added by the team with title, authors, venue, year, and link.*
2. MathWorks — UAV Toolbox documentation: https://www.mathworks.com/products/uav.html
3. MathWorks — Simulink documentation: https://www.mathworks.com/products/simulink.html
4. MathWorks — UAV Toolbox examples and documentation: https://www.mathworks.com/help/uav/

---

## LinkedIn

A brief LinkedIn project post will be added after the final repository contents, results, and team member GitHub/LinkedIn usernames are finalized.

---

## Project Status

- [x] GitHub repository created
- [x] Repository description added
- [x] Quaternion attitude-control simulation developed
- [x] UAV Toolbox 3D visualization connected
- [x] 90-degree roll test validated
- [ ] Euler-angle to quaternion input function finalized
- [ ] Final Simulink model uploaded
- [ ] Final MATLAB source files uploaded
- [ ] Scope figures uploaded
- [ ] Simulation video uploaded
- [ ] PPT uploaded
- [ ] Final team details added
- [ ] Base paper and references finalized
- [ ] Final report completed
- [ ] LinkedIn post published
