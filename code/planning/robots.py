import autograd.numpy as np  # Thinly-wrapped numpy

class RobotCartpole:
  def __init__(self):
    self.mass_cart = 1.0
    self.mass_pole = 0.1
    self.length_pole = 0.5
    self.g = 9.81

    self.n = 4
    self.m = 1

    dt = 0.05

    self.stateDim = 4
    self.ctrlDim = 1
    self.is2D = True
    self.statePosIdx = 0
    # idx in csv input file
    self.idxState = 0
    self.idxCtrl = 4
    self.x_min = np.array([-3, -np.radians(360), -5/dt, -np.radians(180)/dt])
    self.x_max = -self.x_min
    self.u_min = [-10]
    self.u_max = [10]

  def f(self, x, u):
    # parameters
    m_p = self.mass_pole
    m_c = self.mass_cart
    l = self.length_pole
    g = self.g

    # s = [q,qdot], q = [x,th]
    u = np.reshape(u,(self.m,1))
    q = np.reshape(x[0:2],(2,1))
    qdot = np.reshape(x[2:],(2,1))
    th = x[1]
    thdot = x[3]

    # EOM from learning+control@caltech
    # D = np.array([[m_c+m_p,m_p*l*np.cos(th)],[m_p*l*np.cos(th),m_p*(l**2)]])
    a = m_c+m_p
    b = m_p*l*np.cos(th)
    c = m_p*l*np.cos(th)
    d = m_p*(l**2)
    Dinv = 1/(a*d-b*c) * np.array([
      [d, -b],
      [-c, a]])

    C = np.array([[0,-m_p*l*thdot*np.sin(th)],[0,0]])
    G = np.array([[0],[-m_p*g*l*np.sin(th)]])
    B = np.array([[1],[0]])
    qdotdot = np.dot(Dinv, np.dot(B,u) - np.dot(C,qdot) - G)

    res = np.array([qdot[0][0], qdot[1][0], qdotdot[0][0], qdotdot[1][0]])
    # print(res)
    return res

class RobotDubinsCar:
  def __init__(self, v, k):
    self.v = v
    self.k = k

    self.stateDim = 3
    self.ctrlDim = 1
    self.is2D = True
    self.statePosIdx = 0
    # idx in csv input file
    self.idxState = 7
    self.idxCtrl = 11
    self.x_min = [0, 0, -np.pi]
    self.x_max = [60, 60, np.pi]
    self.u_min = [-2]
    self.u_max = [2]

  def f(self, x, u):
    return np.array([
      self.v * np.cos(x[2]),
      self.v * np.sin(x[2]),
      self.k * u[0]])


class RobotDoubleIntegrator:
  def __init__(self):
    self.stateDim = 4
    self.ctrlDim = 2
    self.is2D = True
    self.statePosIdx = 0
    # idx in csv input file
    self.idxState = 7
    self.idxCtrl = 12
    self.x_min = [-6, -6, -10, -10]
    self.x_max = [6, 6, 10, 10]
    self.u_min = [-2, -2]
    self.u_max = [2, 2]

  def f(self, x, u):
    return np.array([
      x[2],
      x[3],
      u[0],
      u[1]])


class RobotAirplane:
  def __init__(self):
    self.mass = 1
    self.rho = 1.225
    self.area = 0.7
    self.Cd0 = 0.015
    self.Kd = 0.025
    self.v_min = 3
    self.v_max = 10
    self.g = 9.81

    self.stateDim = 8
    self.ctrlDim = 3
    self.is2D = False
    self.statePosIdx = 0
    # idx in csv input file
    self.idxState = 7
    self.idxCtrl = 16

    self.x_min = [-11, -11, -11, -3.14159, 3, -0.523599, -0.785398, -3.14159]
    self.x_max = [11, 11, 11, 3.14159, 10, 0.523599, 0.785398, 3.14159]
    self.u_min = [-8.05686, -2.0944, -1.0472]
    self.u_max = [8.05686, 2.0944, 1.0472]

  # x dot = f(x, u)
  def f(self, x, u):
    _,_,_,psi,v,gamma,phi,alpha = x
    Fl = np.pi * self.rho * self.area * v * v * alpha
    Fd = self.rho*self.area * v *v * (self.Cd0 + 4 * np.pi * np.pi * self.Kd * alpha * alpha)
    return np.array([
      v * np.cos(psi) * np.cos(gamma),
      v * np.sin(psi) * np.cos(gamma),
      v * np.sin(gamma),
      -Fl * np.sin(phi) / (self.mass * v * np.cos(gamma)),
      u[0] - Fd/self.mass - self.g * np.sin(gamma),
      Fl * np.cos(phi)/(self.mass*v) - self.g * np.cos(gamma)/v,
      u[1],
      u[2]])