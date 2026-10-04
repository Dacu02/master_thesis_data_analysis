import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src', 'master_thesis_data_analysis', 'data'))

from framework.Movement import Movement
from framework.Stroke import Stroke
from framework.Trajectory import Trajectory

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    # Parameters from the exported table (theta = zenith, psi = azimuth)
    s1 = Stroke(D=0.00139827777164166, mu=-0.001174418462195, sigma=0.0948390265158332, t0=-0.877,
                theta_s=1.59860790486515, psi_s=-0.82984886724927,
                theta_e=1.54316360068371, psi_e=2.35422424746859)
    s2 = Stroke(D=0.189089176144639, mu=0.308634999515702, sigma=0.0984422580105803, t0=-0.715,
                theta_s=2.28149466118698, psi_s=0.342105325333695,
                theta_e=0.748887639041892, psi_e=1.03300701537757)
    s3 = Stroke(D=0.189771302440273, mu=0.0929475031450966, sigma=0.106336185517195, t0=-0.117,
                theta_s=1.28148172700849, psi_s=3.11066874575017,
                theta_e=2.25622120885911, psi_e=2.84069572481738)

    traj = Trajectory([s1, s2])
    traj.append(s3)
    print(traj, "| strokes:", len(traj), "| duration:", round(traj.duration(), 3))
    print(traj.get_stroke(1).get_parameters())

    traj.plot()
    plt.show()

    # ---- checks ----
    # 1) after the last stroke the position equals the end point of the last arc
    far = traj.position(np.array([50.0]))[0]
    print("final position vs end point:", np.linalg.norm(far - np.asarray(traj.get_end_point())))

    # 2) analytic velocity vs finite differences of the position
    t = np.linspace(traj.get_onset(), traj.end_time(), 4000)
    num = np.gradient(traj.position(t), t, axis=0)
    vel = traj.velocity(t)
    print("relative velocity error:", np.abs(num - vel).max() / np.abs(vel).max())   # ~1e-3 expected