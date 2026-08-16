<p align="center">
  <img src="Images/logo-branding-amrita-universiy-2024.jpg" alt="Amrita Vishwa Vidyapeetham" width="900">
</p>

# Full Quaternion-Based Attitude Control of a Quadrotor UAV

## Team Members

| S. No. | Name | Roll Number | Email |
|---|---|---|---|
| 1 | Gudivada Geetham Rishi Kanth | cb.sc.u4aie24218 | cb.sc.u4aie24218@cb.students.amrita.edu |
| 2 | Nakka Saampotth Maddileti | cb.sc.u4aie24233 | cb.sc.u4aie24233@cb.students.amrita.edu |
| 3 | Namburi Rupesh | cb.sc.u4aie24234 | cb.sc.u4aie24234@cb.students.amrita.edu |
| 4 | Telapolu Bala Prasanna Kumar | cb.sc.u4aie24256 | cb.sc.u4aie24256@cb.students.amrita.edu |
| 5 | Uday Sri Yaramati | cb.sc.u4aie24260 | cb.sc.u4aie24260@cb.students.amrita.edu |

## Abstract

This project focuses on the development of a quaternion-based attitude control system for a quadrotor UAV. The project explores the use of quaternions for representing the UAV's orientation and developing an attitude control approach for roll, pitch, and yaw motion. The control system is being developed using MATLAB/Simulink and MuJoCo, providing simulation environments for studying the quadrotor's attitude dynamics and controller response.

The project is based on the quaternion-based attitude control approach presented in the selected base paper, with emphasis on quaternion representation, attitude error calculation, and controller development.  The current work focuses on developing and validating the simulation models and integrating the attitude control framework across the simulation environments.

# Introduction

Unmanned Aerial Vehicles (UAVs), particularly quadrotors with Vertical Take-Off and Landing (VTOL) capability, have gained significant attention because of their ability to perform complex aerial missions. Quadrotor control is generally divided into translation control and attitude control, where the position controller can generate desired attitude set-points for the attitude controller. Achieving accurate, smooth, and robust attitude stabilization remains an important challenge in quadrotor control. 

The attitude of a quadrotor is described by its roll, pitch, and yaw rotations. A major challenge in attitude control arises from the mathematical properties of three-dimensional rotations, since finite rotations are non-commutative and cannot be treated as ordinary vectors. Conventional approaches based on Euler angles are intuitive but suffer from singularities, particularly the well known gimbal-lock problem. Direction Cosine Matrices (DCMs) avoid this singularity but introduce a more complex representation and require maintaining orthogonality constraints. 

To overcome these limitations, the base paper, **“Full Quaternion Based Attitude Control for a Quadrotor” by Emil Fresk and George Nikolakopoulos**, proposes the use of quaternions for representing quadrotor attitude. A quaternion provides a compact four-parameter representation of three-dimensional orientation while avoiding the geometric singularities associated with Euler angles. The paper's key approach is to implement both the quadrotor attitude model and the nonlinear Proportional Squared (P2) controller directly in quaternion space, without requiring transformations through Euler angles or DCMs. 

A unit quaternion can be represented as

$$
q =
\begin{bmatrix}
q_0 & q_1 & q_2 & q_3
\end{bmatrix}^{T},
$$

with the unit-norm constraint

$$
q_0^2+q_1^2+q_2^2+q_3^2=1.
$$

Quaternions can therefore be used to represent the UAV's orientation and calculate attitude errors directly in quaternion space. Euler angles can still be used at the input or visualization level, allowing intuitive roll, pitch, and yaw commands to be converted into quaternion references. 

The proposed project follows this quaternion-based control philosophy and develops a simulation framework using MATLAB/Simulink and MuJoCo. The implementation focuses on quaternion attitude representation, quaternion reference generation, attitude-error calculation, P2 controller development, quadrotor rotational dynamics, and quaternion kinematics. The simulated attitude is also visualized to observe the UAV's response to desired roll, pitch, and yaw commands.

The overall control concept can be summarized as:

$$
q_{\mathrm{ref}}
\rightarrow
\text{Quaternion Error}
\rightarrow
\text{P2 Controller}
\rightarrow
\tau
\rightarrow
\text{Rotational Dynamics}
\rightarrow
q
$$

where $q_{\mathrm{ref}}$ is the desired attitude quaternion, $q$ is the measured attitude, and $\tau$ represents the control torque. This approach provides the foundation for investigating quaternion-based attitude control of a quadrotor while avoiding the singularities associated with conventional Euler-angle representations.
