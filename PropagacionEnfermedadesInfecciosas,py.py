import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo SIR
beta = 0.3  # Tasa de contagio
gamma = 0.1  # Tasa de recuperación

# Sistema de ecuaciones diferenciales
def sir_model(t, y):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Condiciones iniciales
S0, I0, R0 = 0.99, 0.01, 0  # 99% Susceptibles, 1% Infectados, 0% Recuperados
y0 = [S0, I0, R0]
t_span = (0, 100)  # Simulación por 100 días
t_eval = np.linspace(*t_span, 1000)

# Resolviendo con Runge-Kutta
sol = solve_ivp(sir_model, t_span, y0, t_eval=t_eval)

# Graficamos los resultados
plt.plot(sol.t, sol.y[0], label="Susceptibles", color="blue")
plt.plot(sol.t, sol.y[1], label="Infectados", color="red")
plt.plot(sol.t, sol.y[2], label="Recuperados", color="green")
plt.xlabel("Días")
plt.ylabel("Población")
plt.title("Modelo SIR - Propagación de Enfermedad")
plt.legend()
plt.show()
