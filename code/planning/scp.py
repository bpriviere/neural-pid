import cvxpy as cp
# import numpy as np
import autograd.numpy as np  # Thinly-wrapped numpy
import scipy
import traceback
from autograd import grad, elementwise_grad, jacobian

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf


def scp(param, env, xf = None):

  partialFx = jacobian(env.f_scp, 0)
  partialFu = jacobian(env.f_scp, 1)

  def constructA(xbar, ubar):
    return partialFx(xbar, ubar)

  def constructB(xbar, ubar):
    return partialFu(xbar, ubar)

  data = np.loadtxt(param.rrt_fn, delimiter=',', ndmin=2)

  xprev = data[:,0:env.n]
  uprev = data[:,env.n:env.n + env.m]
  T = data.shape[0]
  dt = param.sim_dt

  if xf is None:
    goalState = param.ref_trajectory[:,-1]
  else:
    goalState = xf

  x0 = xprev[0]

  if param.scp_pdf_fn is not None:
    pdf = matplotlib.backends.backend_pdf.PdfPages(param.scp_pdf_fn)

  objectiveValues = []
  xChanges = []
  uChanges = []
  try:
    obj = 'minimizeError' # 'minimizeError', 'minimizeX'

    for iteration in range(10):

      x = cp.Variable((T, env.n))
      u = cp.Variable((T, env.m))

      if obj == 'minimizeError':
        delta = cp.Variable()
        objective = cp.Minimize(delta)
      elif obj == 'minimizeX':
        objective = cp.Minimize(cp.sum_squares(x[:,0:2]))
      else:
        # objective = cp.Minimize(cp.sum_squares(u) + 10 * cp.sum_squares(x[:,3:5]))
        objective = cp.Minimize(cp.sum_squares(u))
        # objective = cp.Minimize(cp.sum_squares(u) + 10 * cp.sum(x[:,4]))
      constraints = [
        x[0] == x0, # initial state constraint
      ]

      if obj == 'minimizeError':
        if goalState is not None:
          constraints.append( cp.abs(x[-1] - goalState) <= delta )
        else:
          constraints.append(cp.abs(x[-1,0:2] - goalPos) <= delta )
      elif obj == 'minimizeX':
        pass
      else:
        if goalState is not None:
          constraints.append( x[-1] == goalState )
        else:
          constraints.append( x[-1,0:2] == goalPos )

      # trust region
      for t in range(0, T):
        constraints.append(
          cp.abs(x[t] - xprev[t]) <= 2 #0.1
        )
        constraints.append(
          cp.abs(u[t] - uprev[t]) <= 2 #0.1
        )

      # dynamics constraints
      for t in range(0, T-1):
        xbar = xprev[t]
        ubar = uprev[t]
        A = constructA(xbar, ubar)
        B = constructB(xbar, ubar)
        # print(xbar, ubar, A, B)
        # print(robot.f(xbar, ubar))
        # # simple version:
        constraints.append(
          x[t+1] == x[t] + dt * (env.f_scp(xbar, ubar) + A @ (x[t] - xbar) + B @ (u[t] - ubar))
          )
        # # discretized zero-order hold
        # F = scipy.linalg.expm(A * dt)
        # G = np.zeros(B.shape)
        # H = np.zeros(xbar.shape)
        # for tau in np.linspace(0, dt, 10):
        #   G += (scipy.linalg.expm(A * tau) @ B) * dt / 10
        #   H += (scipy.linalg.expm(A * tau) @ (robot.f(xbar, ubar) - A @ xbar - B @ ubar)) * dt / 10
        # constraints.append(
        #   x[t+1] == F @ x[t] + G @ u[t] + H
        #   )

      # bounds (x and u)
      for t in range(0, T):
        constraints.extend([
          env.s_min <= x[t],
          x[t] <= env.s_max,
          env.a_min <= u[t],
          u[t] <= env.a_max
          ])

      prob = cp.Problem(objective, constraints)

      # The optimal objective value is returned by `prob.solve()`.
      try:
        result = prob.solve(verbose=True,solver=cp.GUROBI, BarQCPConvTol=1e-8)
      except cp.error.SolverError:
        return

      if result is None:
        return

      objectiveValues.append(result)
      xChanges.append(np.linalg.norm(x.value - xprev))
      uChanges.append(np.linalg.norm(u.value - uprev))

      if obj == 'minimizeError' and result < 1e-6:
        obj = 'minimizeU'

      # The optimal value for x is stored in `x.value`.
      # print(x.value)
      # print(u.value)

      # compute forward propagated value
      xprop = np.empty(x.value.shape)
      xprop[0] = x0
      for t in range(0, T-1):
        xprop[t+1] = xprop[t] + dt * env.f_scp(xprop[t], u.value[t])

      # print(xprop)
      if param.scp_pdf_fn is not None:
        fig, ax = plt.subplots()
        ax.plot(xprev[:,0], xprev[:,1], label="input")
        ax.plot(x.value[:,0], x.value[:,1], label="opt")
        ax.plot(xprop[:,0], xprop[:,1], label="opt forward prop")

        plt.legend()
        # plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(u.value[:,0], label="u0")
        ax.plot(u.value[:,1], label="u1")
        ax.plot(u.value[:,2], label="u2")
        ax.plot(u.value[:,3], label="u3")

        plt.legend()
        # plt.show()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(x.value[:,0], label="x")
        ax.plot(x.value[:,1], label="y")
        ax.plot(x.value[:,2], label="z")

        plt.legend()
        # plt.show()
        pdf.savefig(fig)
        plt.close(fig)

      xprev = np.array(x.value)
      uprev = np.array(u.value)

    if True: #obj == 'minimizeU' or obj == 'minimizeX':
      result = np.hstack([xprev, uprev])
      np.savetxt(param.scp_fn, result, delimiter=',')
      return xprev, uprev

  except Exception as e:
    print(e)
    traceback.print_tb(e.__traceback__)
  finally:
    # print(xprev)
    # print(uprev)
    if param.scp_pdf_fn is not None:
      fig, ax = plt.subplots()
      try:
        ax.plot(np.arange(1,len(objectiveValues)+1), objectiveValues, '*-', label='cost')
        ax.plot(np.arange(1,len(objectiveValues)+1), xChanges, '*-', label='|x-xp|')
        ax.plot(np.arange(1,len(objectiveValues)+1), uChanges, '*-', label='|u-up|')
      except:
        print("Error during plotting!")
      plt.legend()
      pdf.savefig(fig)
      pdf.close()

