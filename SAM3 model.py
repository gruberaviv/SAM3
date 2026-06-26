import mosek
import cvxpy as cp
import numpy as np

print(cp.installed_solvers())


class SamSetup():


    # input parameters from the parameters table
    def __init__(self, J, _lambda, Tij, Tb, T0, Tr, h, pai, ship_cost, z, S0_tilda_val):
        # number of fsl
        self.J = int(J)
        # Planning horizon
        self.T0 = T0
        # transportation time from warehouse to FSL
        self.Tr = list(Tr)  # vector(1,2)
        # penalty.
        self.h = np.reshape(np.array(list(h)), (1, J))
        # item shortage cost at FSL
        self.pai = np.reshape(np.array(list(pai)), (1, J))
        # poisson parameter. constant failure rate
        self._lambda = _lambda
        # items available at FSL
        self.S0_tilda = S0_tilda_val * np.ones((T0 + Tb, 1))

        # ================new variables=================
        # shipment cost from the depot to FSL
        # /c_ij/ the shipment cost of item from the depot to FSL j.
        self.ship_cost = ship_cost
        # /z_ij/ the incremental cost of repairing item at FSL j instead of at the depot.
        self.z = np.array(z)
        # emmision lead time (time to make dicision)
        self.Tb = Tb
        # base-repair lead time until available at FSL
        self.Tij = list(Tij)  # vector(1,2)

        # size of matrix for item availability in warehouse
        self.X_size = max(min(self.Tij), min(self.Tr) + self.T0 + self.Tb), self.J
        self.S_tild_size = self.X_size

        # items available at warehouse
        self.S_tilda = 2 * np.ones(self.S_tild_size)
        self.cases = ''

    def cp_setup(self):

        self.y = cp.Variable((self.T0 + self.Tb, self.J), integer=True)
        self.f = cp.Variable()
        self.S = cp.Variable(self.X_size)
        self.G = cp.Variable(self.X_size)
        self.cum_y = cp.Variable((self.y.shape), integer=True)

        # ============new setup variables=======================
        # Decision variables to send to repair !from central depot! at :
        self.S0 = cp.Variable(self.T0 + self.Tb)
        # cummulative num of items at FSL for which a decision, where to send for repair has not been taken up to time t.
        self.S_tild_F = cp.Variable((self.Tb, self.J))

        self.x_b = cp.Variable((self.Tb, self.J), integer=True)  # the same FSL
        self.x_w = cp.Variable((self.Tb, self.J), integer=True)  # central depot
        self.cum_x_b = cp.Variable((self.Tb, self.J))
        self.cum_x_w = cp.Variable((self.Tb, self.J))

        self.constraints = [self.f == sum(self.G) + sum(self.ship_cost * self.cum_y) + sum(self.z * self.x_b),
                            self.y >= 0,
                            self.cum_y >= 0,
                            self.x_b >= 0,
                            self.x_w >= 0,
                            self.cum_x_b >= 0,
                            self.cum_x_w >= 0,
                            self.S >= 0,
                            self.S0 >= 0,
                            self.G >= 0]
        return self.y, self.f, self.S, self.G, self.cum_y, self.constraints, self.x_b, self.x_w, self.S_tild_F

    def cp_setup(self):

        self.y = cp.Variable((self.T0 + self.Tb, self.J), integer=True)
        self.f = cp.Variable()
        self.S = cp.Variable(self.X_size)
        self.G = cp.Variable(self.X_size)
        self.cum_y = cp.Variable((self.T0 + self.Tb, self.J), integer=True)

        # ============new setup variables=======================
        # Decision variables to send to repair !from central depot! at :
        self.S0 = cp.Variable(self.T0 + self.Tb)
        # cummulative num of items at FSL for which a decision, where to send for repair has not been taken up to time t.
        self.S_tild_F = cp.Variable((self.Tb, self.J))

        self.x_b = cp.Variable((self.Tb, self.J), integer=True)  # the same FSL
        self.x_w = cp.Variable((self.Tb, self.J), integer=True)  # central depot
        self.cum_x_b = cp.Variable((self.Tb, self.J))
        self.cum_x_w = cp.Variable((self.Tb, self.J))

        self.constraints = [self.f == sum(self.G) + sum(self.ship_cost * self.cum_y) + sum(self.z * self.x_b),
                            self.y >= 0,
                            self.cum_y >= 0,
                            self.x_b >= 0,
                            self.x_w >= 0,
                            self.cum_x_b >= 0,
                            self.cum_x_w >= 0,
                            self.S >= 0,
                            self.S0 >= 0,
                            self.G >= 0]
        return self.y, self.f, self.S, self.G, self.cum_y, self.constraints, self.x_b, self.x_w, self.S_tild_F

    def sam_problem(self):
        self.cp_setup()

        def constraints_count():
            # cum_x_b, cum_x_w
            for j in range(self.J):
                for t in range(self.Tb):
                    self.constraints += [self.cum_x_b[t, j] == cp.sum(self.x_b[0:t, j])]
                    self.constraints += [self.cum_x_w[t, j] == cp.sum(self.x_w[0:t, j])]
            # constraint 2!!!
            self.constraints += [self.cum_x_b @ np.ones((self.cum_x_b.shape[1], 1)) +
                                 self.cum_x_w @ np.ones((self.cum_x_w.shape[1], 1)) <= self.S_tild_F]
            # create cum_y
            # constraint 3!!
            for j in range(self.J):
                for t in range(self.Tb + self.T0):
                    self.constraints += [self.cum_y[t, j] == cp.sum(self.y[0:t, j])]
            self.constraints += [
                self.cum_y @ np.ones((self.cum_y.shape[1], 1)) <= self.S0 @ np.ones((self.S0.shape[0], 1))]
            return constraints_count

        """Sould I loop over J????? """

        def constraints_S0():
            print(self.T0, self.Tb, self.y.shape)
            # constraint 4!!
            for j in range(self.J):
                for t in range(self.T0):
                    self.constraints += [self.S0 == self.S0_tilda[t, 0]]
            for j in range(self.J):
                for t in range(self.T0, self.T0 + self.Tb):
                    self.constraints += [self.S0 == self.S0_tilda[self.T0 - 1, 0] + sum(self.cum_x_w[t - self.T0, :])]
            self.constraints += [
                self.cum_y @ np.ones((self.cum_y.shape[1], 1)) <= self.S0 @ np.ones((self.S0.shape[0], 1))]
            return constraints_S0

        """Cases functions"""

        def S_case_1(j):
            print("***** CASE 1 *****")
            for t in range(self.Tij[j]):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j]]

            for t in range(self.Tij[j], self.Tij[j] + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_x_b[t - self.Tij[j], j]]
            #             if self.Tr[j] != (self.Tij[j] + self.Tb):

            for t in range(self.Tij[j] + self.Tb, self.Tr[j]):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_x_b[self.Tb - 1, j]]

            for t in range(self.Tr[j], self.Tr[j] + self.T0 + self.Tb + 1):
                self.constraints += [self.S[t, j] == self.S_tilda[self.Tr[j] - 1, j] + self.cum_x_b[self.Tb, j]
                                     + self.cum_y[t - self.Tr[j], j]]
            return self.constraints

        def S_case_2(j):
            print("***** CASE 2 *****")
            for t in range(self.Tij[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j]]
            if self.Tij[j] != self.Tr[j]:
                for t in range(self.Tij[j], self.Tr[j] - 1):
                    self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_x_b[t - self.Tij[j], j]]
            for t in range(self.Tr[j], self.Tij[j] + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[self.Tr[j] - 1, j] + self.cum_x_b[t - self.Tij[j], j]
                                     + self.cum_y[t - self.Tr[j], j]]
            for t in range(self.Tij[j] + self.Tb, self.Tr[j] + self.T0 + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_x_b[self.Tb - 1, j]
                                     + self.cum_y[t - self.Tr[j], j]]
            return self.constraints

        def S_case_3(j):
            print("***** CASE 3 *****")
            for t in range(self.Tr[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j]]
            for t in range(self.Tr[j], self.Tij[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_y[t - self.Tr[j], j]]
            for t in range(self.Tij[j], self.Tij[j] + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[self.Tij[j] - 1, j] + self.cum_x_b[t - self.Tij[j], j]
                                     + self.cum_y[t - self.Tr[j], j]]
            for t in range(self.Tij[j] + self.Tb, self.Tr[j] + self.T0 + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[self.Tij[j] - 1, j] + self.cum_x_b[self.Tb - 1, j]
                                     + self.cum_y[t - self.Tr[j], j]]
            return self.constraints

        def S_case_4(j):
            print("***** CASE 4 *****")
            for t in range(self.Tr[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j]]
            for t in range(self.Tr[j], self.Tr[j] + self.T0 + self.Tb + 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_y[t - self.Tr[j], j]]
            for t in range(self.Tr[j] + self.T0 + self.Tb, self.Tij[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_y[self.T0 + self.Tb, j]]
            #             if self.Tij[j] != self.Tr[j]+self.T0+self.Tb:
            #                 for t in range(self.Tij[j], self.Tr[j]+self.T0+self.Tb):
            #                     self.constraints += [self.S[t,j] == self.S_tilda[self.Tij[j]-1,j]+ self.cum_w_b[t-self.Tij[j],j]
            #                                          + self.cum_y[t-self.Tr[j],j]]
            for t in range(self.Tij[j], self.Tij[j] + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[self.Tij[j] - 1, j] + self.cum_x_b[t - self.Tij[j], j]
                                     + self.cum_y[self.T0 + self.Tb, j]]
            return self.constraints

        def S_case_5(j):
            print("***** CASE 5 *****")
            for t in range(self.Tr[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j]]
            for t in range(self.Tr[j], self.Tr[j] + self.T0 + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_y[t - self.Tr[j], j]]
            for t in range(self.Tr[j] + self.T0 + self.Tb, self.Tij[j] - 1):
                self.constraints += [self.S[t, j] == self.S_tilda[t, j] + self.cum_y[self.T0 + self.Tb - 1, j]]
            for t in range(self.Tij[j], self.Tij[j] + self.Tb):
                self.constraints += [self.S[t, j] == self.S_tilda[self.Tij[j] - 1, j] + self.cum_x_b[t - self.Tij[j], j]
                                     + self.cum_y[self.T0 + self.Tb - 1, j]]
            return self.constraints

        def G_case_1(j):
            for t in range(self.Tij[j], self.Tr[j] + self.T0 + self.Tb):
                self.constraints += [self.G[t, j] >= self.h[0, j] * (self.S[t, j] - self._lambda * (t + 1)),
                                     self.G[t, j] >= self.pai[0, j] * (self._lambda * (t + 1) - self.S[t, j])]
            return self.constraints

        def G_case_2(j):
            for t in range(self.Tij[j], self.Tr[j] + self.T0 + self.Tb):
                self.constraints += [self.G[t, j] >= self.h[0, j] * (self.S[t, j] - self._lambda * (t + 1)),
                                     self.G[t, j] >= self.pai[0, j] * (self._lambda * (t + 1) - self.S[t, j])]
            return self.constraints

        def G_case_3(j):
            for t in range(self.Tr[j], self.Tr[j] + self.T0 + self.Tb):
                self.constraints += [self.G[t, j] >= self.h[0, j] * (self.S[t, j] - self._lambda * (t + 1)),
                                     self.G[t, j] >= self.pai[0, j] * (self._lambda * (t + 1) - self.S[t, j])]
            return self.constraints

        def G_case_4(j):
            for t in range(self.Tr[j], self.Tij[j] + self.Tb):
                self.constraints += [self.G[t, j] >= self.h[0, j] * (self.S[t, j] - self._lambda * (t + 1)),
                                     self.G[t, j] >= self.pai[0, j] * (self._lambda * (t + 1) - self.S[t, j])]
            return self.constraints

        def G_case_5(j):
            for t in range(self.Tr[j], self.Tij[j] + self.Tb):
                self.constraints += [self.G[t, j] >= self.h[0, j] * (self.S[t, j] - self._lambda * (t + 1)),
                                     self.G[t, j] >= self.pai[0, j] * (self._lambda * (t + 1) - self.S[t, j])]
            return self.constraints

        # Calculating supply and demand constraints
        def constraints_of_S():
            for j in range(self.J):
                if ((self.Tr[j] + self.T0) >= self.Tij[j]) and ((self.Tij[j] + self.Tb) <= self.Tr[j]):
                    S_case_1(j)
                    self.cases = 'case 1'
                elif ((self.Tr[j] + self.T0) >= self.Tij[j]) and (
                        (self.Tij[j] <= self.Tr[j]) and (self.Tr[j] < (self.Tij[j] + self.Tb))):
                    S_case_2(j)
                    self.cases = 'case 2'
                elif ((self.Tr[j] + self.T0) >= self.Tij[j]) and (
                        (self.Tij[j] > self.Tr[j]) and (self.Tij[j] <= (self.Tr[j] + self.Tb + self.T0))):
                    S_case_3(j)
                    self.cases = 'case 3'
                elif ((self.Tr[j] + self.T0) < self.Tij[j]) and (self.Tij[j] > (self.Tr[j] + self.T0 + self.Tb)):
                    S_case_4(j)
                    self.cases = 'case 4'
                elif ((self.Tr[j] + self.T0) < self.Tij[j]) and (
                        (self.Tij[j] > self.Tr[j]) and (self.Tij[j] <= (self.Tr[j] + self.Tb + self.T0))):
                    S_case_5(j)
                    self.cases = 'case 5'
            return constraints_of_S

            # Calculating cost function

        def constraints_of_G():
            for j in range(self.J):
                if ((self.Tr[j] + self.T0) >= self.Tij[j]) and ((self.Tij[j] + self.Tb) <= self.Tr[j]):
                    G_case_1(j)
                elif ((self.Tr[j] + self.T0) >= self.Tij[j]) and (
                        (self.Tij[j] <= self.Tr[j]) & (self.Tr[j] < (self.Tij[j] + self.Tb))):
                    G_case_2(j)
                elif ((self.Tr[j] + self.T0) >= self.Tij[j]) and (
                        (self.Tij[j] > self.Tr[j]) & (self.Tij[j] <= (self.Tr[j] + self.Tb + self.T0))):
                    G_case_3(j)
                elif ((self.Tr[j] + self.T0) < self.Tij[j]) and (self.Tij[j] > (self.Tr[j] + self.T0 + self.Tb)):
                    G_case_4(j)
                elif ((self.Tr[j] + self.T0) < self.Tij[j]) and (
                        (self.Tij[j] > self.Tr[j]) & (self.Tij[j] <= (self.Tr[j] + self.Tb + self.T0))):
                    G_case_5(j)
            return constraints_of_G

        obj = cp.Minimize(self.f)
        constraints_count()
        constraints_S0()
        constraints_of_S()
        constraints_of_G()
        prob = cp.Problem(obj, self.constraints)
        prob.solve(solver=cp.MOSEK)

        return self.x_b.value, self.x_w.value, self.x_b.value.sum(), self.x_w.value.sum(), self.G.value, self.f.value, self.y.value, self.G.value.sum(), self.y.value.sum(), self.cases

    def __repr__(self):
        return f"{self.J}, {self._lambda},{self.T0},{self.Tr},{self.Tb},{self.Tij},{self.h},{self.pai},{self.ship_cost},{self.z},{self.S0_tilda},{self.S_tilda},{self.x_b},{self.x_w},{self.f.value}"

    def variablesData(self):
        return {'J': self.J, 'lambda': self._lambda,
                'T0': self.T0, 'Tr': tuple(self.Tr), 'Tb': self.Tb, 'Tij': tuple(self.Tij),
                'h': tuple(self.h[0]), 'pai': tuple(self.pai[0]), 'ship_cost': self.ship_cost, 'z': self.z,
                'S0tilda': self.S0_tilda, 'Stilda': self.S_tilda,
                'x_b': self.x_b.value, 'x_w': self.x_w.value, 'x_b_sum': self.x_b.value.sum(),
                'x_w_sum': self.x_w.value.sum(),
                'f_value': self.f.value,
                'G_value': self.G.value, 'G_sum': self.G.value.sum(),
                'y_value': self.y.value, 'y_sum': self.y.value.sum(), 'Cases': self.cases}

    pass