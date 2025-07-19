import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Datos del Problema ---
# Características: [Área (x1), Habitaciones (x2), Distancia (x3)]
X = np.array([
    [150, 3, 5],
    [120, 2, 10],
    [200, 4, 2],
    [180, 3, 3],
    [100, 2, 12]
])
# Variable a predecir: Precio (y)
y = np.array([300, 250, 450, 350, 200])

#Entrenar el Modelo de Regresión Lineal ---
# Usamos scikit-learn para obtener los coeficientes óptimos de forma rápida
model = LinearRegression()
model.fit(X, y)

# Parámetros del modelo entrenado
theta_0 = model.intercept_
theta_1, theta_2, theta_3 = model.coef_

print(f"Modelo entrenado: Precio = {theta_0:.2f} + {theta_1:.2f}*Área + {theta_2:.2f}*Habitaciones + {theta_3:.2f}*Distancia")

#Generación de Gráficos ---

# Gráfico 1: Dispersión 3D de Datos y Plano de Regresión
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

ax1.scatter(X[:, 0], X[:, 2], y, c='r', marker='o', s=50, label='Datos Reales')

x1_surf = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
x3_surf = np.linspace(X[:, 2].min(), X[:, 2].max(), 10)
x1_surf, x3_surf = np.meshgrid(x1_surf, x3_surf)

y_surf = theta_0 + theta_1 * x1_surf + theta_2 * X[:, 1].mean() + theta_3 * x3_surf

# Dibujar el plano
ax1.plot_surface(x1_surf, x3_surf, y_surf, color='b', alpha=0.3, label='Plano de Regresión')

ax1.set_xlabel('Área construida (x1)')
ax1.set_ylabel('Distancia al centro (x3)')
ax1.set_zlabel('Precio (y)')
ax1.set_title('Datos y Plano de Regresión')
plt.savefig('graficos/dispersion_3d.png')
print("Gráfico guardado como 'graficos/dispersion_3d.png'")


# Gráfico 2: Curvas de Nivel de la Función de Coste
def cost_function(X, y, theta):
    m = len(y)
    predictions = X @ theta
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

X_b = np.c_[np.ones((len(X), 1)), X] # Añadir x0=1 para theta0

theta1_vals = np.linspace(theta_1 - 2, theta_1 + 2, 100)
theta3_vals = np.linspace(theta_3 - 15, theta_3 + 15, 100)
J_vals = np.zeros((len(theta1_vals), len(theta3_vals)))

for i, t1 in enumerate(theta1_vals):
    for j, t3 in enumerate(theta3_vals):
        t = np.array([theta_0, t1, theta_2, t3])
        J_vals[i, j] = cost_function(X_b, y, t)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.contour(theta1_vals, theta3_vals, J_vals.T, levels=np.logspace(0, 5, 20), cmap='viridis')
ax2.plot(theta_1, theta_3, 'rX', markersize=10, label='Mínimo encontrado')
ax2.set_xlabel('Theta 1 (Peso del Área)')
ax2.set_ylabel('Theta 3 (Peso de la Distancia)')
ax2.set_title('Curvas de Nivel de la Función de Coste')
ax2.legend()
plt.savefig('graficos/curvas_de_nivel.png')
print("Gráfico guardado como 'graficos/curvas_de_nivel.png'")

# Gráfico 3: Convergencia del Coste (Descenso de Gradiente)
alpha = 0.00001
iterations = 100
theta_gd = np.zeros(X_b.shape[1])
cost_history = []

for i in range(iterations):
    predictions = X_b @ theta_gd
    errors = predictions - y
    gradients = (1/len(y)) * X_b.T @ errors
    theta_gd -= alpha * gradients
    cost_history.append(cost_function(X_b, y, theta_gd))

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(range(iterations), cost_history, 'b-')
ax3.set_xlabel('Número de Iteraciones')
ax3.set_ylabel('Función de Coste J(θ)')
ax3.set_title('Convergencia del Descenso de Gradiente')
ax3.grid(True)
plt.savefig('graficos/convergencia.png')
print("Gráfico guardado como 'graficos/convergencia.png'")


# Gráfico 4: Predicciones vs. Valores Reales
y_pred = model.predict(X)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(y, y_pred, edgecolors=(0, 0, 0))
ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax4.set_xlabel('Precios Reales')
ax4.set_ylabel('Precios Predichos')
ax4.set_title('Predicciones vs. Valores Reales')
ax4.grid(True)
plt.savefig('graficos/predicciones_vs_reales.png')
print("Gráfico guardado como 'graficos/predicciones_vs_reales.png'")


# Gráfico 5: Superficie 3D de la Función de Coste (Paraboloide)
fig5 = plt.figure(figsize=(10, 8))
ax5 = fig5.add_subplot(111, projection='3d')

T1, T3 = np.meshgrid(theta1_vals, theta3_vals)

ax5.plot_surface(T1, T3, J_vals.T, cmap='viridis', alpha=0.7)
ax5.set_xlabel('Theta 1 (Peso del Área)')
ax5.set_ylabel('Theta 3 (Peso de la Distancia)')
ax5.set_zlabel('Coste J(θ)')
ax5.set_title('Superficie 3D de la Función de Coste')
plt.savefig('graficos/paraboloide_coste.png')
print("Gráfico guardado como 'graficos/paraboloide_coste.png'")
