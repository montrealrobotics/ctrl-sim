import numpy as np


class BicycleModel:
    def __init__(self, x: float = 0, y: float = 0, theta: float = 0, L: float = 1, vel: float = 0, dt: float = 0.1):
        """When running this in forward mode, init with first state. When running in backward mode,
        init with last state and final velocity and orientation.

        :param x: x starting pos
        :param y: y starting pos
        :param theta: starting angle (aligned with x axis i.e. car starts moving right)
        :param L: distance between front and rear axles - idk how long is an average car? Must be online somewhere.
        :param vel: starting velocity
        :param dt: default time step
        """
        super().__init__()

        self.x = x  # current position
        self.y = y  # current position
        self.theta = theta  # current angle
        self.L = L  # current axle between front and back wheel (doesn't change)
        self.vel = vel  # current velocity
        self.dt = dt

    def forward(self, accel, steer, dt=None):
        """Given accel and steer, update the state of the agent and return new pos, angle, and velocity
        :param accel: acceleration
        :param steer: steering angle
        :param dt: time step (only use this if it's different from the default)

        :return: x, y, theta, vel, (position x, position y, angle in world coord (rel to x axis), velocity)
        """

        if dt is None:
            # if dt not specified, use default
            dt = self.dt

        # update velocity
        self.vel += accel * dt

        # update pos and rotation
        theta = self.theta +self.vel * dt * np.tan(steer) / self.L
        self.x += self.vel * dt * np.cos(self.theta)
        self.y += self.vel * dt * np.sin(self.theta)
        print (f"old theta {self.theta}, new theta {theta}")

        self.theta = theta

        return self.x, self.y, self.theta, self.vel

    def backward(self, prev_pos, prev_theta=None, prev_vel=None, dt=None):
        """This will internally update the pos to be the previous_pos
        :param previous_pos: previous position
        :param dt: time step (only use this if it's different from the default)
        :return: steer angle, accel that brought the agent from the previous point to the current point
        """
        if dt is None:
            # if dt not specified, use default
            dt = self.dt

        x, y = prev_pos

        if prev_vel is None:
            # distance to previous pos
            dist = np.sqrt((x - self.x) ** 2 + (self.y - y) ** 2)
            # velocity between current point (self.x/y) and previous point (x/y
            vel = dist / dt
        else:
            vel = prev_vel

        # diff in velocity gives us acceleration
        accel = (self.vel - vel) / dt

        if prev_theta is None:
            # we only need x or y to calculate new theta
            theta = np.arccos((self.x - x) / (vel * dt))
        else:
            theta = prev_theta

        # steer comes from difference in theta
        # steer = np.arctan((self.theta - theta) / (vel * dt))
        # print (f"x {x}, self.x {self.x}, theta {theta}, self.theta {self.theta}, steer {steer}")
        def angle_sub(current_angle, target_angle) -> int:
            """Subtract two angles to find the minimum angle between them."""
            # Subtract the angles, constraining the value to [0, 2 * np.pi)
            diff = (target_angle - current_angle) % (2 * np.pi)

            # If we are more than np.pi we're taking the long way around.
            # Let's instead go in the shorter, negative direction
            if diff > np.pi:
                diff = -(2 * np.pi - diff)
            return diff
        # w = (self.theta - theta) / dt
        w = angle_sub(theta, self.theta) / dt
        C = 2.0 * self.L * w / (self.vel + vel + 1e-10)
        steer = np.arctan(2.0 * C / np.sqrt(4 - C**2))
        if np.isnan(steer):
            steer = 0.0

        steer = np.clip(steer, a_min=-0.7, a_max=0.7)
        # accel = np.clip(accel, a_min=-10.0, a_max=10.0)

        # update pos and rotation
        self.x = x
        self.y = y
        self.theta = theta
        self.vel = vel

        return accel, steer, self.vel, self.theta


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # run this for 5 steps

    # x = 0
    # y = 0
    # theta = 1 ( # starting angle
    # L = 1 # length of the car (axle to axle)
    # vel = 2 (m/s) # starting vel
    bm = BicycleModel(0, 0, 1, 1, 2)

    # hardcoded rollout with some fixed accel and steer
    p1 = [0, 0, 1, 2] # start, copied from init
    p2 = bm.forward(1, 0)
    p3 = bm.forward(0, 0.2)
    p4 = bm.forward(2, 0.4)
    p5 = bm.forward(0, 0)
    p6 = bm.forward(4, 0.6)

    accel_gt = np.array([1, 0, 2, 0, 4, 0])
    steer_gt = np.array([0, 0.2, 0.4, 0, 0.6, 0])

    steps_fwd = np.array([p1, p2, p3, p4, p5, p6])

    c = np.linspace(0, 1, len(steps_fwd))

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.scatter(steps_fwd[:, 0], steps_fwd[:, 1], c=c)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1.2)
    plt.title("x/y pos over time (yellow = newest)")

    plt.subplot(2, 3, 2)
    plt.plot(np.arange(len(steps_fwd)), steps_fwd[:, 2])
    plt.title("angle over time")

    plt.subplot(2, 3, 3)
    plt.plot(np.arange(len(steps_fwd)), steps_fwd[:, 3])
    plt.title("vel over time")

    # plt.show()

    final_angle = steps_fwd[-1, 2]
    print ("final angle", final_angle)
    final_vel = steps_fwd[-1, 3]
    final_pos = steps_fwd[-1, :2]
    print("final pos", final_pos)
    bm_inv = BicycleModel(final_pos[0], final_pos[1], final_angle, 1, final_vel)
    steps_bwd = []

    steps_fwd = steps_fwd[::-1]  # reverse
    for i in range(len(steps_fwd) - 1):
        last_point = steps_fwd[i+1, :2]  # get last point (inverse order)
        steps_bwd.append(bm_inv.backward(last_point))

    steps_bwd = np.array(steps_bwd)[::-1]
    accel_inferred = steps_bwd[:, 0]
    steer_inferred = steps_bwd[:, 1]

    # steer needs to be padded with zero at the end
    steer_inferred = np.append(steer_inferred, 0)
    # accel needs to be padded with zero at the beginning
    accel_inferred = np.insert(accel_inferred, 0, 0)

    x = np.arange(len(steps_bwd)+1)
    plt.subplot(2, 3, 6)
    plt.scatter(x, accel_inferred, alpha=0.5, marker="x", label="inferred")
    plt.scatter(x, accel_gt, alpha=0.5, label="GT")
    plt.legend()
    plt.title("accel at each step (inferred)")

    plt.subplot(2, 3, 5)
    plt.scatter(x, steer_inferred, alpha=0.5, marker="x", label="inferred")
    plt.scatter(x, steer_gt, alpha=0.5, label="GT")
    plt.legend()
    plt.title("steer at each step (inferred)")


    plt.show()