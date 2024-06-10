from utils import *
import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(wd)


class iteration:
    def __init__(self, parameter_model):
        self.parameter_model = parameter_model

    def iterate(self, parameter_numeric, parameter_iteration, initialization):
        self.parameter_numeric = parameter_numeric
        self.parameter_iteration = parameter_iteration
        self.value = initialization
        self.iteration = []

        print('start: '+time.asctime(time.localtime(time.time())))

        for i in range(self.parameter_iteration[0]):
            value = self.map()
            self.update(value)
            self.save()
            if i > 0:
                convergence_indicator = self.convergence_indicator()
                self.log(i, convergence_indicator)
                if convergence_indicator < parameter_iteration[4]:
                    break

    def map(self):
        x = np.zeros((self.parameter_numeric[1], self.parameter_numeric[2]))
        x[0] = np.random.rand(self.parameter_numeric[2])
        chi = np.zeros(
            (self.parameter_numeric[1], self.parameter_numeric[1], self.parameter_numeric[2]))
        for i in range(self.parameter_numeric[1]-1):
            chi[i+1, i] = np.ones(self.parameter_numeric[2])
        R_a = np.zeros(
            (self.parameter_numeric[1], self.parameter_numeric[1], self.parameter_numeric[2]))
        def phi(x): return np.tanh(x)
        def phi_derivative(x): return (1/np.cosh(x))**2

        gamma = np.random.multivariate_normal(np.zeros(
            self.parameter_numeric[1]), self.parameter_model[0]**2*self.value[0]+self.parameter_model[2]**2/self.parameter_numeric[0]*np.eye(self.parameter_numeric[1]), self.parameter_numeric[2]).T
        for i in range(self.parameter_numeric[1]-1):
            x[i+1] = (1-self.parameter_numeric[0])*x[i]\
                + self.parameter_numeric[0]*gamma[i]\
                + self.parameter_model[1] * self.parameter_model[0]**2 * \
                self.parameter_numeric[0]**2 * \
                np.dot(self.value[1][i, :i], phi(x)[:i])
        C = np.dot(phi(x), phi(x).T)/self.parameter_numeric[2]
        for j in range(self.parameter_numeric[1]):
            for i in range(j+1, self.parameter_numeric[1]):
                R_a[i, j] = phi_derivative(x[i])*chi[i, j]
                if i < self.parameter_numeric[1]-1:
                    chi[i+1, j] = (1-self.parameter_numeric[0])*chi[i, j]\
                        + self.parameter_model[1] * self.parameter_model[0]**2*self.parameter_numeric[0]**2 * np.dot(
                            self.value[1][i, j:i], R_a[j:i, j])
        R = np.mean(R_a, axis=2)

        Cx = np.dot(x, x.T)/self.parameter_numeric[2]
        m = np.mean(phi(x), axis=1)

        return [C, R, Cx, m]

    def update(self, value):
        for i in range(len(self.value)):
            self.value[i] = self.parameter_iteration[1] * self.value[i]\
                + (1-self.parameter_iteration[1]) * value[i]

    def save(self):
        self.iteration.append(self.value.copy())
        with open(self.parameter_iteration[2]+'iteration.pkl', 'wb') as file:
            pickle.dump(self.iteration, file)

    def convergence_indicator(self):
        difference_Cx = self.iteration[-1][2] - self.iteration[-2][2]
        norm_difference_Cx = np.trace(
            np.dot(difference_Cx, difference_Cx.T))/self.parameter_numeric[1]**2
        return norm_difference_Cx

    def log(self, i, convergence_indicator):
        if i % self.parameter_iteration[3] == 0:
            print(str(i)+' '+time.asctime(time.localtime(time.time())))
            print(convergence_indicator)
