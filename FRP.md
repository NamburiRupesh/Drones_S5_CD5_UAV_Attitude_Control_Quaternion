# Project Proposal (FRP)

## Project Title

**Quaternion-Based Attitude and Trajectory Control of a UAV using P² Controller**

## Team Members

| Name | Roll Number | Email |
|---|---|---|
| Gudivada Geetham Rishi Kanth | CB.SC.U4AIE24218 | cb.sc.u4aie24218@cb.students.amrita.edu |
| Nakka Saampotth Maddileti | CB.SC.U4AIE24233 | cb.sc.u4aie24233@cb.students.amrita.edu |
| Namburi Rupesh | CB.SC.U4AIE24234 | cb.sc.u4aie24234@cb.students.amrita.edu |
| Telapolu Bala Prasanna Kumar | CB.SC.U4AIE24256 | cb.sc.u4aie24256@cb.students.amrita.edu |
| Uday Sri Yaramati | CB.SC.U4AIE24260 | cb.sc.u4aie24260@cb.students.amrita.edu |

## Proposed Project Work

This project aims to develop a quaternion-based control framework for UAV attitude and trajectory control. The main focus is to design and simulate a quaternion-based P² attitude controller that can accurately control the rotational motion of a quadrotor.

The project uses quaternion representation for UAV orientation to overcome limitations of Euler-angle based approaches such as singularities and gimbal lock. The controller will generate required control torques to track desired roll, pitch, and yaw commands.

The developed system will be implemented and tested using MATLAB/Simulink and MuJoCo simulation environments.

## Objectives

- Model the rotational dynamics of a UAV.
- Implement quaternion-based attitude representation.
- Convert desired roll, pitch, and yaw commands into reference quaternions.
- Design and simulate a quaternion-based P² attitude controller.
- Analyze attitude tracking performance for different orientation commands.
- Validate the controller using Simulink and MuJoCo simulations.
- Extend the system towards UAV position and trajectory control.

## Methodology

1. Generate reference attitude using Euler angle to quaternion conversion.
2. Calculate quaternion error between desired and measured UAV orientation.
3. Use the P² controller to generate corrective torque commands.
4. Simulate UAV angular dynamics and quaternion evolution.
5. Visualize UAV attitude response using simulation environments.
6. Extend the work towards translation and trajectory generation using interpolation techniques.

## Expected Outcomes

- A working quaternion-based UAV attitude controller.
- Stable tracking of desired roll, pitch, and yaw commands.
- Simulation validation using MATLAB/Simulink and MuJoCo.
- Foundation for complete UAV trajectory control.

## Tools and Technologies

| Tool | Purpose |
|---|---|
| MATLAB/Simulink | UAV modelling and controller simulation |
| MuJoCo | Physics-based UAV simulation |
| Python | Simulation and controller implementation |
| GitHub | Project documentation and version control |

## Future Scope

Future extensions include translational dynamics modelling, position control, trajectory planning using LERP, and smooth attitude trajectory generation using SLERP.