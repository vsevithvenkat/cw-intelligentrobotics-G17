

\### 1. Statement of the Assumption: Collision-Free Reachable Workspace

While the nominal actuator limits defined in the coursework specification span $q\_4 \\in \[-0.20, 0.00]\\text{ m}$ for the Epson SCARA and $q\_2 \\in \[-0.8, 0.8]\\text{ rad}$, $q\_3 \\in \[0.0, 1.3]\\text{ rad}$ (elbow-up) for the Niryo Ned, not all independent combinations within these raw bounds are physically executable inside the Webots R2023b simulator due to ground plane and self-body contacts\[cite: 1].



\*\*Formal Assumption:\*\*  

We assume that the valid operational workspace for Forward Kinematics precision validation (Task 1.C), dataset generation (Task 2), and simulation-based evaluation (Task 4.B) is constrained to the \*\*collision-free reachable workspace\*\* of each manipulator within the default Webots scene\[cite: 1]. Concurrently, we retain the full nominal specification bounds for $\[-1.0, 1.0]$ joint normalisation and de-normalisation (Tasks 2 and 3) to ensure strict adherence to coursework requirements\[cite: 1].



\---



\### 2. Mathematical and Empirical Justification



\#### A. Epson SCARA (`ScaraT6`)

From the coursework specification, the Tool Centre Point (TCP) position in the `ScaraT6` base frame is\[cite: 1]:



$$\\mathbf{p} = \\begin{bmatrix} 0.060 \\\\ 0 \\\\ 0.220 \\end{bmatrix} + R\_z(q\_1) \\left( \\begin{bmatrix} 0.310 \\\\ 0.005 \\\\ -0.004 \\end{bmatrix} + R\_z(q\_2) \\begin{bmatrix} 0.2545 \\\\ -0.005 \\\\ q\_4 - 0.034 \\end{bmatrix} \\right)$$



Because $R\_z(q\_1)$ and $R\_z(q\_2)$ represent planar rotations strictly about the vertical Z-axis, the vertical TCP coordinate simplifies to:



$$z\_{\\text{TCP}} = 0.220 - 0.004 + (q\_4 - 0.034) = 0.182 + q\_4$$



\* \*\*Theoretical Boundary Discrepancy:\*\* Evaluating the nominal lower actuator stroke $q\_4 = -0.200\\text{ m}$ yields $z\_{\\text{TCP}} = -0.018\\text{ m}$ ($18\\text{ mm}$ below the $z = 0$ ground plane)\[cite: 1].

\* \*\*Empirical Webots Telemetry:\*\* In simulation, commanding $q\_4 < -0.1729\\text{ m}$ results in physical contact between the vertical shaft collision geometry and the floor plane at $q\_{4,\\text{stall}} = -0.1729\\text{ m}$ ($z\_{\\text{TCP}} = 0.0093\\text{ m}$).

\* \*\*Operational Constraint:\*\* Sampled vertical shaft extensions are restricted to:

&#x20; $$q\_4 \\in \[-0.170, 0.000]\\text{ m}$$

&#x20; This maintains a $2.9\\text{ mm}$ safety margin above the contact threshold while preventing stall-induced errors.



\#### B. Niryo Ned

For the Niryo Ned arm, the active controlled joints are `joint\_1` ($q\_1$), `joint\_2` ($q\_2$), and `joint\_3` ($q\_3$), while the wrist joints (`joint\_4`, `joint\_5`, `joint\_6`) are held fixed at $0.0\\text{ rad}$\[cite: 1].

\* \*\*Overhanging Geometry Discrepancy:\*\* The coursework specifies the TCP at the interface between the forearm link and the first wrist motor (`joint\_4`)\[cite: 1]. However, the physical wrist assembly and end-effector extend approximately $0.11\\text{ m}$ past this point. When the shoulder pitch ($q\_2$) and elbow pitch ($q\_3$) simultaneously rotate forward/downward, the physical gripper contacts the base structure or floor before reaching extreme target angles.

\* \*\*Operational Constraint:\*\* Two geometric boundary filters are applied:

&#x20; 1. \*\*Minimum TCP Height:\*\* $z\_{\\text{TCP}} \\ge 0.120\\text{ m}$ (guaranteeing that the end-effector clears the floor by $\\ge 10\\text{ mm}$).

&#x20; 2. \*\*Maximum Combined Pitch:\*\* $q\_2 + q\_3 \\le 1.15\\text{ rad}$ (preventing structural self-contact with the robot base).



\---



\### 3. Impact on Forward Kinematics Evaluation (Task 1.C)



Applying this collision-free operational assumption resolves physical stall outliers without requiring modifications to the `.wbt` world geometry, ensuring cross-platform reproducibility\[cite: 1]:



| Manipulator | Sampling Strategy | Mean Error (mm) | Std Error (mm) | Max Error (mm) |

| :--- | :--- | :---: | :---: | :---: |

| \*\*Epson SCARA\*\* | Unconstrained Limits ($q\_4 \\in \[-0.20, 0.00]$)\[cite: 1] | 1.4024 | 5.1722 | 25.8605 |

| \*\*Epson SCARA\*\* | \*\*Collision-Free Filter ($q\_4 \\in \[-0.17, 0.00]$)\*\* | \*\*0.1281\*\* | \*\*0.0644\*\* | \*\*0.2618\*\* |

| \*\*Niryo Ned\*\* | Unconstrained Limits ($q\_2 \\in \[-0.8, 0.8], q\_3 \\in \[0.0, 1.3]$)\[cite: 1] | 0.3310 | 1.0580 | 5.5071 |

| \*\*Niryo Ned\*\* | \*\*Collision-Free Filter ($z\_{\\text{TCP}} \\ge 0.12\\text{ m}, \\sum q\_i \\le 1.15$)\*\* | \*\*0.1193\*\* | \*\*0.0004\*\* | \*\*0.1197\*\* |



\---



\### 4. Normalisation Compliance (Tasks 2 \& 3)

To maintain compliance with the project specification, data normalisation and de-normalisation strictly utilize the official datasheet limits\[cite: 1]:



$$q\_{\\text{norm}} = 2\\frac{q - q\_{\\min}}{q\_{\\max} - q\_{\\min}} - 1 \\quad \\Longleftrightarrow \\quad q = q\_{\\min} + \\frac{q\_{\\text{norm}} + 1}{2}(q\_{\\max} - q\_{\\min})$$



\* \*\*SCARA Limits:\*\* $q\_1 \\in \[-0.73, 0.73]\\text{ rad}$, $q\_2 \\in \[0.00, 0.83]\\text{ rad}$, $q\_4 \\in \[-0.20, 0.00]\\text{ m}$\[cite: 1]

\* \*\*Ned Limits:\*\* $q\_1 \\in \[-2.80, 2.80]\\text{ rad}$, $q\_2 \\in \[-0.80, 0.80]\\text{ rad}$, $q\_3 \\in \[0.00, 1.30]\\text{ rad}$\[cite: 1]



This decoupling allows neural network outputs to map to the complete standard input range while guaranteeing that generated training coordinates reflect physically executable targets in simulation\[cite: 1].

