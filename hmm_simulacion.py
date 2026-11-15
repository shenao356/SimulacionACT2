"""
Simulación y Evaluación Estadística de un Modelo Oculto de Markov (HMM)
para el Monitoreo de Condición de Maquinaria Industrial mediante Vibraciones.

Actividad didáctica 2-M2 · IU Digital de Antioquia · Simulación
Estándar de Reproducibilidad y Rigor Científico Q1 (Scopus / WoS)

Estados ocultos (S) : Sano (S0), Desgaste (S1), Crítico (S2)
Observaciones (V)   : Vibración Baja (V0), Media (V1), Alta (V2)

Incluye:
1. Simulación estocástica de cadenas de Markov ocultas con semillas controladas.
2. Problema 1 de Rabiner (Evaluación y Filtrado): Algoritmo Forward-Backward escalado.
3. Problema 2 de Rabiner (Decodificación óptima): Algoritmo de Viterbi en escala logarítmica.
4. Problema 3 de Rabiner (Aprendizaje no supervisado): Algoritmo de Baum-Welch (EM).
5. Evaluación experimental Multi-Escenario (4 escenarios x 500 réplicas Monte Carlo)
   con Media, Desviación Estándar, Mínimo, Máximo, Intervalos de Confianza al 95% (IC 95%),
   Matriz de Confusión por estado y Retardo de Detección de Fallas.
6. Generación de figuras de calidad publicable a 300 DPI.

Uso:
    python hmm_simulacion.py --pasos 200 --semilla 42 --corridas 500
"""
import argparse
import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ESTADOS = ["Sano", "Desgaste", "Crítico"]
ESTADOS_ASCII = ["Sano", "Desgaste", "Critico"]
OBSERV = ["Baja", "Media", "Alta"]

# ==============================================================================
# PARÁMETROS DEL ESCENARIO 1 (BASE - OPERACIÓN NOMINAL)
# ==============================================================================
PI_BASE = np.array([0.80, 0.18, 0.02])

A_BASE = np.array([
    [0.92, 0.07, 0.01],
    [0.05, 0.85, 0.10],
    [0.40, 0.10, 0.50],
])

B_BASE = np.array([
    [0.75, 0.20, 0.05],
    [0.20, 0.55, 0.25],
    [0.05, 0.25, 0.70],
])

# ==============================================================================
# DEFINICIÓN DE LOS 4 ESCENARIOS EXPERIMENTALES (RESPUESTA A P2 Y P3)
# ==============================================================================
ESCENARIOS = {
    "E1_Base_Nominal": {
        "nombre": "E1: Base (Operación Nominal)",
        "descripcion": "Condición inicial sana (π=[0.80,0.18,0.02]), degradación progresiva estándar y sensor comercial.",
        "PI": PI_BASE.copy(),
        "A": A_BASE.copy(),
        "B": B_BASE.copy(),
    },
    "E2_Inicio_Critico": {
        "nombre": "E2: Inicio Crítico (Variación de π)",
        "descripcion": "Máquina inicia deteriorada (π=[0.05,0.25,0.70]) con igual dinámica A y B. Evalúa ergodicidad.",
        "PI": np.array([0.05, 0.25, 0.70]),
        "A": A_BASE.copy(),
        "B": B_BASE.copy(),
    },
    "E3_Degradacion_Severa": {
        "nombre": "E3: Sobrecarga Industrial (Variación de A)",
        "descripcion": "Régimen severo de operación: desgaste acelerado y menor tasa de reparación correctiva en A.",
        "PI": PI_BASE.copy(),
        "A": np.array([
            [0.82, 0.15, 0.03],
            [0.03, 0.77, 0.20],
            [0.25, 0.15, 0.60],
        ]),
        "B": B_BASE.copy(),
    },
    "E4_Sensor_Alta_Precision": {
        "nombre": "E4: Sensor Calibrado (Variación de B)",
        "descripcion": "Sensor piezoeléctrico de baja ambigüedad (menor solapamiento espectral en matriz B).",
        "PI": PI_BASE.copy(),
        "A": A_BASE.copy(),
        "B": np.array([
            [0.90, 0.08, 0.02],
            [0.08, 0.82, 0.10],
            [0.02, 0.10, 0.88],
        ]),
    },
}


def validar(pi=PI_BASE, A=A_BASE, B=B_BASE):
    """Verifica las restricciones de estocasticidad del modelo HMM."""
    assert np.isclose(pi.sum(), 1.0), "El vector inicial PI debe sumar 1.0"
    assert np.allclose(A.sum(axis=1), 1.0), "Las filas de A deben sumar 1.0"
    assert np.allclose(B.sum(axis=1), 1.0), "Las filas de B deben sumar 1.0"
    assert (pi >= 0).all() and (A >= 0).all() and (B >= 0).all(), "Probabilidades negativas no permitidas"


def simular(T, rng, pi=PI_BASE, A=A_BASE, B=B_BASE):
    """Genera trayectorias (estados ocultos z, observaciones x) con distribuciones categóricas."""
    N = len(pi)
    M = B.shape[1]
    z = np.empty(T, dtype=int)
    x = np.empty(T, dtype=int)
    z[0] = rng.choice(N, p=pi)
    x[0] = rng.choice(M, p=B[z[0]])
    for t in range(1, T):
        z[t] = rng.choice(N, p=A[z[t - 1]])
        x[t] = rng.choice(M, p=B[z[t]])
    return z, x


def forward(x, pi=PI_BASE, A=A_BASE, B=B_BASE):
    """Algoritmo Forward escalado: calcula log-verosimilitud log P(X|λ) y filtrado P(z_t | x_{1:t})."""
    T = len(x)
    N = len(pi)
    alfa = np.zeros((T, N))
    c = np.zeros(T)
    a = pi * B[:, x[0]]
    c[0] = a.sum()
    alfa[0] = a / c[0]
    for t in range(1, T):
        a = (alfa[t - 1] @ A) * B[:, x[t]]
        c[t] = a.sum()
        alfa[t] = a / c[t]
    return np.log(c).sum(), alfa, c


def backward(x, c, A=A_BASE, B=B_BASE):
    """Algoritmo Backward escalado para suavizado y aprendizaje Baum-Welch."""
    T = len(x)
    N = A.shape[0]
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, x[t + 1]] * beta[t + 1])) / c[t + 1]
    return beta


def viterbi(x, pi=PI_BASE, A=A_BASE, B=B_BASE):
    """Algoritmo de Viterbi en log-espacio: decodifica la secuencia globalmente más probable."""
    T = len(x)
    N = len(pi)
    eps = 1e-15
    d = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    d[0] = np.log(pi + eps) + np.log(B[:, x[0]] + eps)
    logA = np.log(A + eps)
    logB = np.log(B + eps)
    for t in range(1, T):
        m = d[t - 1][:, None] + logA
        psi[t] = m.argmax(axis=0)
        d[t] = m.max(axis=0) + logB[:, x[t]]
    z = np.zeros(T, dtype=int)
    z[-1] = d[-1].argmax()
    for t in range(T - 2, -1, -1):
        z[t] = psi[t + 1, z[t + 1]]
    return z


def baum_welch(secuencias_x, A_init, B_init, pi_init, max_iter=35, tol=1e-4):
    """
    Algoritmo de Baum-Welch (Expectation-Maximization) sobre múltiples secuencias de observación
    para estimar parámetros (A, B, pi) de manera no supervisada (Problema 3 de Rabiner).
    """
    A_est = A_init.copy()
    B_est = B_init.copy()
    pi_est = pi_init.copy()
    N, M = B_est.shape
    historial_ll = []

    for it in range(max_iter):
        pi_num = np.zeros(N)
        A_num = np.zeros((N, N))
        A_den = np.zeros(N)
        B_num = np.zeros((N, M))
        B_den = np.zeros(N)
        ll_total = 0.0

        for x in secuencias_x:
            T = len(x)
            ll, alfa, c = forward(x, pi_est, A_est, B_est)
            beta = backward(x, c, A_est, B_est)
            ll_total += ll

            gamma = alfa * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            pi_num += gamma[0]
            B_den += gamma.sum(axis=0)
            A_den += gamma[:-1].sum(axis=0)

            for k in range(M):
                mask = (x == k)
                if np.any(mask):
                    B_num[:, k] += gamma[mask].sum(axis=0)

            for t in range(T - 1):
                xi_t = (alfa[t][:, None] * A_est) * (B_est[:, x[t + 1]] * beta[t + 1])[None, :]
                xi_t /= xi_t.sum()
                A_num += xi_t

        pi_est = pi_num / len(secuencias_x)
        A_est = A_num / A_den[:, None]
        B_est = B_num / B_den[:, None]
        historial_ll.append(ll_total / len(secuencias_x))

        if it > 0 and abs(historial_ll[-1] - historial_ll[-2]) < tol:
            break

    return pi_est, A_est, B_est, historial_ll


def estacionaria(A=A_BASE):
    """Distribución estacionaria teórica de la cadena ergódica (autovector izquierdo asociado a λ=1)."""
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return p / p.sum()


def calcular_retardo_deteccion(z_real, z_pred):
    """
    Calcula el retardo promedio (en pasos) con que Viterbi detecta la entrada
    a un nuevo episodio de deterioro (Desgaste o Crítico).
    """
    retardos = []
    T = len(z_real)
    t = 1
    while t < T:
        if z_real[t] != z_real[t - 1] and z_real[t] in (1, 2):
            estado_objetivo = z_real[t]
            fin_episodio = t
            while fin_episodio < T and z_real[fin_episodio] == estado_objetivo:
                fin_episodio += 1
            detectado = np.where(z_pred[t:fin_episodio] == estado_objetivo)[0]
            if len(detectado) > 0:
                retardos.append(int(detectado[0]))
            t = fin_episodio
        else:
            t += 1
    return np.mean(retardos) if len(retardos) > 0 else 0.0


def estadisticos_ic95(valores):
    """Retorna media, desviación estándar muestral, mínimo, máximo y límites del IC al 95%."""
    arr = np.asarray(valores, dtype=float)
    n = len(arr)
    media = float(np.mean(arr))
    desv = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = desv / np.sqrt(n) if n > 0 else 0.0
    ic_inf = media - 1.96 * sem
    ic_sup = media + 1.96 * sem
    return {
        "media": media,
        "desv": desv,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ic95_inf": ic_inf,
        "ic95_sup": ic_sup,
    }


def evaluar_escenario(clave, cfg, pasos, corridas, semilla_base):
    """Ejecuta simulación Monte Carlo completa para un escenario y retorna métricas detalladas."""
    pi, A, B = cfg["PI"], cfg["A"], cfg["B"]
    validar(pi, A, B)
    rng = np.random.default_rng(semilla_base)

    # Consumir primera corrida si es el escenario base para mantener idéntica la semilla 42 de David
    if clave == "E1_Base_Nominal":
        _ = simular(pasos, rng, pi, A, B)

    accs_viterbi = []
    accs_forward = []
    lls = []
    retardos = []
    frec_est = np.zeros(3)
    frec_obs = np.zeros(3)
    trans = np.zeros((3, 3))
    emis = np.zeros((3, 3))
    conf_matrix = np.zeros((3, 3), dtype=int)

    for _ in range(corridas):
        zz, xx = simular(pasos, rng, pi, A, B)
        zv = viterbi(xx, pi, A, B)
        ll, alfa, _ = forward(xx, pi, A, B)
        zf = alfa.argmax(axis=1)

        accs_viterbi.append((zv == zz).mean())
        accs_forward.append((zf == zz).mean())
        lls.append(ll)
        retardos.append(calcular_retardo_deteccion(zz, zv))

        frec_est += np.bincount(zz, minlength=3)
        frec_obs += np.bincount(xx, minlength=3)
        for a_, b_ in zip(zz[:-1], zz[1:]):
            trans[a_, b_] += 1
        for s_, o_ in zip(zz, xx):
            emis[s_, o_] += 1
        for r_, p_ in zip(zz, zv):
            conf_matrix[r_, p_] += 1

    frec_est /= frec_est.sum()
    frec_obs /= frec_obs.sum()
    A_hat = trans / trans.sum(axis=1, keepdims=True)
    B_hat = emis / emis.sum(axis=1, keepdims=True)
    pi_teo = estacionaria(A)
    obs_teo = pi_teo @ B

    recall_por_estado = np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=1), 1)

    return {
        "clave": clave,
        "nombre": cfg["nombre"],
        "descripcion": cfg["descripcion"],
        "viterbi_stats": estadisticos_ic95(accs_viterbi),
        "forward_stats": estadisticos_ic95(accs_forward),
        "ll_stats": estadisticos_ic95(lls),
        "retardo_stats": estadisticos_ic95(retardos),
        "accs_viterbi_raw": np.array(accs_viterbi),
        "accs_forward_raw": np.array(accs_forward),
        "frec_est": frec_est,
        "pi_teo": pi_teo,
        "frec_obs": frec_obs,
        "obs_teo": obs_teo,
        "A_hat": A_hat,
        "B_hat": B_hat,
        "err_A": float(np.abs(A_hat - A).max()),
        "err_B": float(np.abs(B_hat - B).max()),
        "conf_matrix": conf_matrix,
        "recall_estados": recall_por_estado,
    }


def generar_diagrama_metodologico(ruta_salida):
    """Genera una figura compuesta a 300 DPI con el grafo del HMM y el flujo metodológico F1-F7."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- Panel Izquierdo: Grafo de Estados Ocultos y Emisiones ---
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis("off")
    ax1.set_title("A) Topología del HMM: Estados Ocultos (S) y Emisiones (V)", fontsize=11, fontweight="bold", pad=10)

    nodos_s = [(2.0, 5.8, "Sano\n(S₀)\nπ=0.80", "#c8e6c9", "#2e7d32"),
               (5.0, 5.8, "Desgaste\n(S₁)\nπ=0.18", "#fff9c4", "#f57f17"),
               (8.0, 5.8, "Crítico\n(S₂)\nπ=0.02", "#ffcdd2", "#c62828")]

    nodos_v = [(2.0, 1.6, "Vibración\nBaja (V₀)", "#e3f2fd", "#1565c0"),
               (5.0, 1.6, "Vibración\nMedia (V₁)", "#e3f2fd", "#1565c0"),
               (8.0, 1.6, "Vibración\nAlta (V₂)", "#e3f2fd", "#1565c0")]

    for x, y, lbl, fc, ec in nodos_s:
        circle = plt.Circle((x, y), 0.92, facecolor=fc, edgecolor=ec, lw=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(x, y, lbl, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)

    for x, y, lbl, fc, ec in nodos_v:
        box = FancyBboxPatch((x - 0.95, y - 0.65), 1.9, 1.3, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, lw=1.8, zorder=3)
        ax1.add_patch(box)
        ax1.text(x, y, lbl, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)

    # Flechas de transición entre estados ocultos
    ax1.annotate("", xy=(4.05, 6.05), xytext=(2.95, 6.05),
                 arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    ax1.text(3.5, 6.3, "a₀₁=0.07", ha="center", fontsize=7.5)

    ax1.annotate("", xy=(2.95, 5.55), xytext=(4.05, 5.55),
                 arrowprops=dict(arrowstyle="->", lw=1.3, color="#555", ls="--"))
    ax1.text(3.5, 5.15, "a₁₀=0.05", ha="center", fontsize=7.5)

    ax1.annotate("", xy=(7.05, 6.05), xytext=(5.95, 6.05),
                 arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    ax1.text(6.5, 6.3, "a₁₂=0.10", ha="center", fontsize=7.5)

    ax1.annotate("", xy=(2.3, 6.62), xytext=(7.7, 6.62),
                 arrowprops=dict(arrowstyle="->", lw=1.6, color="#1b5e20", connectionstyle="arc3,rad=0.32"))
    ax1.text(5.0, 7.62, "Mantenimiento correctivo: a₂₀ = 0.40", ha="center", fontsize=8.2, color="#1b5e20", fontweight="bold")

    # Autotransiciones
    ax1.text(2.0, 4.55, "a₀₀=0.92", ha="center", fontsize=7.5, color="#2e7d32", fontweight="bold")
    ax1.text(5.0, 4.55, "a₁₁=0.85", ha="center", fontsize=7.5, color="#f57f17", fontweight="bold")
    ax1.text(8.0, 4.55, "a₂₂=0.50", ha="center", fontsize=7.5, color="#c62828", fontweight="bold")

    # Emisiones dominantes y cruzadas
    for sx, vx, prob, col in [
        (2.0, 2.0, "b₀₀=0.75", "#2e7d32"),
        (5.0, 5.0, "b₁₁=0.55", "#f57f17"),
        (8.0, 8.0, "b₂₂=0.70", "#c62828"),
    ]:
        ax1.annotate("", xy=(vx, 2.3), xytext=(sx, 4.4),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color=col))
        ax1.text((sx + vx) / 2 + 0.48, 3.35, prob, ha="center", fontsize=7.5, color=col, fontweight="bold")

    # Emisiones cruzadas (ruido)
    for sx, vx in [(2.0, 5.0), (5.0, 2.0), (5.0, 8.0), (8.0, 5.0)]:
        ax1.annotate("", xy=(vx, 2.3), xytext=(sx, 4.4),
                     arrowprops=dict(arrowstyle="->", lw=0.9, color="#888", ls=":"))

    # --- Panel Derecho: Flujo Metodológico de Simulación y Validación ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis("off")
    ax2.set_title("B) Flujo Metodológico y Diseño Experimental (F1–F7)", fontsize=11, fontweight="bold", pad=10)

    pasos_flujo = [
        (7.1, "1. Especificación Formal λ = (S, V, A, B, π) y Validación Estocástica", "#e8eaf6", "#283593"),
        (5.8, "2. Generador Categórico Monte Carlo (Semilla Fija = 42, T = 200)", "#e0f2f1", "#00695c"),
        (4.5, "3. Inferencia: Filtrado Forward Escalado P(z_t|x_1:t) y Decodificación Viterbi", "#fff3e0", "#e65100"),
        (3.2, "4. Aprendizaje No Supervisado Baum-Welch (EM) sobre Observaciones", "#f3e5f5", "#6a1b9a"),
        (1.9, "5. Evaluación Multi-Escenario (E1–E4, N = 500 réplicas c/u)", "#fce4ec", "#ad1457"),
        (0.6, "6. Estadística Q1: Media, DE, Mín, Máx, IC 95%, Matriz de Confusión", "#e8f5e9", "#1b5e20"),
    ]

    for idx_f, (y, texto, fc, ec) in enumerate(pasos_flujo):
        box = FancyBboxPatch((0.35, y - 0.42), 9.3, 0.84, boxstyle="round,pad=0.08",
                             facecolor=fc, edgecolor=ec, lw=1.6, zorder=3)
        ax2.add_patch(box)
        ax2.text(5.0, y, texto, ha="center", va="center", fontsize=7.8, fontweight="bold", color="#111", zorder=4)
        if idx_f < len(pasos_flujo) - 1:
            y_next = pasos_flujo[idx_f + 1][0]
            ax2.annotate("", xy=(5.0, y_next + 0.45), xytext=(5.0, y - 0.45),
                         arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"))

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Simulación HMM Multi-Escenario con Estándar Q1")
    ap.add_argument("--pasos", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--corridas", type=int, default=500)
    ap.add_argument("--salida", default="results")
    args = ap.parse_args()

    validar()
    os.makedirs(args.salida, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    # ==========================================================================
    # 1) SIMULACIÓN INDIVIDUAL DETALLADA (SEMILLA 42, ESCENARIO BASE)
    # ==========================================================================
    z, x = simular(args.pasos, rng, PI_BASE, A_BASE, B_BASE)
    ll, alfa, _ = forward(x, PI_BASE, A_BASE, B_BASE)
    zv = viterbi(x, PI_BASE, A_BASE, B_BASE)
    acc_v = (zv == z).mean()
    acc_f = (alfa.argmax(axis=1) == z).mean()
    retardo_ind = calcular_retardo_deteccion(z, zv)

    print("=== 1. Simulación Individual (Escenario Base, Semilla 42) ===")
    print("Estados ocultos (primeros 30):", " ".join(ESTADOS_ASCII[i][0] for i in z[:30]))
    print("Observaciones   (primeras 30):", " ".join(OBSERV[i][0] for i in x[:30]))
    print("Viterbi         (primeros 30):", " ".join(ESTADOS_ASCII[i][0] for i in zv[:30]))
    print(f"Log-verosimilitud de la secuencia: {ll:.2f}")
    print(f"Precisión Viterbi: {acc_v:.3f} | Precisión filtrado forward: {acc_f:.3f} | Retardo medio detección: {retardo_ind:.2f} pasos")

    ruta_secuencia_csv = os.path.join(args.salida, "secuencia_simulada.csv")
    with open(ruta_secuencia_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "estado_oculto", "observacion", "viterbi", "P_sano", "P_desgaste", "P_critico"])
        for t in range(args.pasos):
            w.writerow([t, ESTADOS_ASCII[z[t]], OBSERV[x[t]], ESTADOS_ASCII[zv[t]], *np.round(alfa[t], 4)])

    # ==========================================================================
    # 2) EVALUACIÓN EXPERIMENTAL MULTI-ESCENARIO (4 ESCENARIOS x 500 RÉPLICAS)
    # ==========================================================================
    resultados_escenarios = {}
    for clave, cfg in ESCENARIOS.items():
        res = evaluar_escenario(clave, cfg, args.pasos, args.corridas, args.semilla)
        resultados_escenarios[clave] = res

    res_base = resultados_escenarios["E1_Base_Nominal"]
    vs_b = res_base["viterbi_stats"]
    fs_b = res_base["forward_stats"]

    print(f"\n=== 2. Análisis Monte Carlo Escenario Base ({args.corridas} corridas x {args.pasos} pasos) ===")
    print(f"Precisión Viterbi : Media={vs_b['media']:.4f} | DE={vs_b['desv']:.4f} | Mín={vs_b['min']:.4f} | Máx={vs_b['max']:.4f} | IC95%=[{vs_b['ic95_inf']:.4f}, {vs_b['ic95_sup']:.4f}]")
    print(f"Precisión Forward : Media={fs_b['media']:.4f} | DE={fs_b['desv']:.4f} | Mín={fs_b['min']:.4f} | Máx={fs_b['max']:.4f} | IC95%=[{fs_b['ic95_inf']:.4f}, {fs_b['ic95_sup']:.4f}]")
    print("Frecuencia empírica de estados :", dict(zip(ESTADOS_ASCII, [round(float(v), 4) for v in res_base["frec_est"]])))
    print("Distribución estacionaria teórica:", dict(zip(ESTADOS_ASCII, [round(float(v), 4) for v in res_base["pi_teo"]])))
    print("Frecuencia empírica de observ. :", dict(zip(OBSERV, [round(float(v), 4) for v in res_base["frec_obs"]])))
    print("Observaciones teóricas         :", dict(zip(OBSERV, [round(float(v), 4) for v in res_base["obs_teo"]])))
    print("Error máx. |A_hat - A| =", round(res_base["err_A"], 4))
    print("Error máx. |B_hat - B| =", round(res_base["err_B"], 4))

    print("\n=== 3. Comparación Estadística Multi-Escenario (Estándar Q1) ===")
    filas_comp = []
    for clave, r in resultados_escenarios.items():
        v = r["viterbi_stats"]
        fw = r["forward_stats"]
        ll_s = r["ll_stats"]
        ret_s = r["retardo_stats"]
        print(
            f"{r['nombre']:<42} | Viterbi: {v['media']*100:5.2f}% ± {v['desv']*100:4.2f}% "
            f"(Mín {v['min']*100:5.1f}%, Máx {v['max']*100:5.1f}%, IC95% [{v['ic95_inf']*100:5.2f}%, {v['ic95_sup']*100:5.2f}%]) "
            f"| Forward: {fw['media']*100:5.2f}% | P_est(Crítico): {r['frec_est'][2]*100:4.2f}%"
        )
        filas_comp.append({
            "escenario": r["nombre"],
            "viterbi_media": round(v["media"], 4),
            "viterbi_desv": round(v["desv"], 4),
            "viterbi_min": round(v["min"], 4),
            "viterbi_max": round(v["max"], 4),
            "viterbi_ic95_inf": round(v["ic95_inf"], 4),
            "viterbi_ic95_sup": round(v["ic95_sup"], 4),
            "forward_media": round(fw["media"], 4),
            "forward_desv": round(fw["desv"], 4),
            "forward_min": round(fw["min"], 4),
            "forward_max": round(fw["max"], 4),
            "forward_ic95_inf": round(fw["ic95_inf"], 4),
            "forward_ic95_sup": round(fw["ic95_sup"], 4),
            "log_verosimilitud_media": round(ll_s["media"], 2),
            "retardo_deteccion_pasos": round(ret_s["media"], 2),
            "frec_sano_sim": round(float(r["frec_est"][0]), 4),
            "frec_desgaste_sim": round(float(r["frec_est"][1]), 4),
            "frec_critico_sim": round(float(r["frec_est"][2]), 4),
            "est_sano_teo": round(float(r["pi_teo"][0]), 4),
            "est_desgaste_teo": round(float(r["pi_teo"][1]), 4),
            "est_critico_teo": round(float(r["pi_teo"][2]), 4),
            "recall_sano": round(float(r["recall_estados"][0]), 4),
            "recall_desgaste": round(float(r["recall_estados"][1]), 4),
            "recall_critico": round(float(r["recall_estados"][2]), 4),
            "error_max_A": round(r["err_A"], 4),
            "error_max_B": round(r["err_B"], 4),
        })

    df_comp = pd.DataFrame(filas_comp)
    df_comp.to_csv(os.path.join(args.salida, "comparacion_escenarios.csv"), index=False, encoding="utf-8")

    # ==========================================================================
    # 3) ESTIMACIÓN NO SUPERVISADA BAUM-WELCH (PROBLEMA 3 DE RABINER)
    # ==========================================================================
    rng_bw = np.random.default_rng(args.semilla + 100)
    secuencias_bw = [simular(args.pasos, rng_bw, PI_BASE, A_BASE, B_BASE)[1] for _ in range(40)]
    A_init_pert = np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.75, 0.15],
        [0.30, 0.20, 0.50],
    ])
    B_init_pert = np.array([
        [0.65, 0.25, 0.10],
        [0.25, 0.50, 0.25],
        [0.10, 0.30, 0.60],
    ])
    pi_init_pert = np.array([0.70, 0.20, 0.10])
    pi_bw, A_bw, B_bw, hist_ll = baum_welch(secuencias_bw, A_init_pert, B_init_pert, pi_init_pert, max_iter=35)
    err_A_bw = float(np.abs(A_bw - A_BASE).max())
    err_B_bw = float(np.abs(B_bw - B_BASE).max())
    print(f"\n=== 4. Aprendizaje No Supervisado Baum-Welch (40 secuencias x {args.pasos} pasos) ===")
    print(f"Iteraciones ejecutadas: {len(hist_ll)} | Log-verosimilitud inicial: {hist_ll[0]:.2f} -> final: {hist_ll[-1]:.2f}")
    print(f"Error máx. post-entrenamiento |A_bw - A| = {err_A_bw:.4f} | Error máx. |B_bw - B| = {err_B_bw:.4f}")

    # Guardar resumen_montecarlo.csv enriquecido
    with open(os.path.join(args.salida, "resumen_montecarlo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerow(["precision_viterbi_media", round(vs_b["media"], 4)])
        w.writerow(["precision_viterbi_desv", round(vs_b["desv"], 4)])
        w.writerow(["precision_viterbi_min", round(vs_b["min"], 4)])
        w.writerow(["precision_viterbi_max", round(vs_b["max"], 4)])
        w.writerow(["precision_viterbi_ic95_inf", round(vs_b["ic95_inf"], 4)])
        w.writerow(["precision_viterbi_ic95_sup", round(vs_b["ic95_sup"], 4)])
        w.writerow(["precision_forward_media", round(fs_b["media"], 4)])
        w.writerow(["precision_forward_desv", round(fs_b["desv"], 4)])
        for n, e, t_val in zip(ESTADOS_ASCII, res_base["frec_est"], res_base["pi_teo"]):
            w.writerow([f"frec_empirica_{n}", round(float(e), 4)])
            w.writerow([f"estacionaria_teorica_{n}", round(float(t_val), 4)])
        w.writerow(["error_max_A", round(res_base["err_A"], 4)])
        w.writerow(["error_max_B", round(res_base["err_B"], 4)])
        w.writerow(["error_max_A_baum_welch", round(err_A_bw, 4)])
        w.writerow(["error_max_B_baum_welch", round(err_B_bw, 4)])

    # ==========================================================================
    # 4) GENERACIÓN DE FIGURAS DE CALIDAD PUBLICABLE (300 DPI)
    # ==========================================================================
    N = min(80, args.pasos)
    t = np.arange(N)

    # Figura 1: Secuencia real, observaciones y decodificación de Viterbi (300 DPI)
    fig, ax = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    ax[0].step(t, z[:N], where="mid", color="#1565c0", lw=1.8)
    ax[0].set_yticks(range(3))
    ax[0].set_yticklabels(ESTADOS)
    ax[0].set_title("A) Estado Oculto Real de la Máquina $z_t$ (Primeros 80 pasos, Semilla 42)", fontweight="bold")
    ax[0].grid(True, linestyle=":", alpha=0.6)

    ax[1].step(t, x[:N], where="mid", color="#ef6c00", lw=1.8)
    ax[1].set_yticks(range(3))
    ax[1].set_yticklabels(OBSERV)
    ax[1].set_title("B) Observación Visible del Sensor de Vibración $x_t$", fontweight="bold")
    ax[1].grid(True, linestyle=":", alpha=0.6)

    ax[2].step(t, z[:N], where="mid", color="#90caf9", lw=2.5, alpha=0.55, label="Estado Real")
    ax[2].step(t, zv[:N], where="mid", color="#2e7d32", lw=1.8, label="Viterbi Inferido")
    errores_idx = np.where(z[:N] != zv[:N])[0]
    if len(errores_idx) > 0:
        ax[2].scatter(errores_idx, zv[errores_idx], color="#c62828", zorder=5, s=28, label="Desfase/Error de transición")
    ax[2].set_yticks(range(3))
    ax[2].set_yticklabels(ESTADOS)
    ax[2].set_title(f"C) Trayectoria Óptima Decodificada por Viterbi (Precisión en corrida: {acc_v:.1%})", fontweight="bold")
    ax[2].set_xlabel("Paso de tiempo discreto ($t$)")
    ax[2].legend(loc="upper right", fontsize=8.5)
    ax[2].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.salida, "secuencia.png"), dpi=300)
    plt.close()

    # Figura 2: Filtrado Forward (300 DPI)
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.stackplot(t, alfa[:N].T, labels=ESTADOS, colors=["#4caf50", "#ffb300", "#e53935"], alpha=0.88)
    ax.set_title("Evolución Temporal de la Probabilidad Filtrada Forward $P(z_t \\mid x_{1:t})$", fontweight="bold")
    ax.set_xlabel("Paso de tiempo discreto ($t$)")
    ax.set_ylabel("Probabilidad a posteriori")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.salida, "filtrado.png"), dpi=300)
    plt.close()

    # Figura 3: Monte Carlo Escenario Base (300 DPI)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    idx = np.arange(3)
    wd = 0.36
    ax[0].bar(idx - wd / 2, res_base["frec_est"], wd, label="Frecuencia Simulada (MC)", color="#1976d2")
    ax[0].bar(idx + wd / 2, res_base["pi_teo"], wd, label=r"Estacionaria Teórica ($\pi^*$)", color="#ff9800")
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels(ESTADOS)
    ax[0].set_ylabel("Proporción de tiempo")
    ax[0].set_title("A) Convergencia Empírica vs. Distribución Estacionaria", fontweight="bold", fontsize=10)
    ax[0].legend()
    ax[0].grid(axis="y", linestyle=":", alpha=0.6)

    accs_b = res_base["accs_viterbi_raw"]
    ax[1].hist(accs_b, bins=22, color="#2e7d32", edgecolor="white", alpha=0.88)
    ax[1].axvline(vs_b["media"], color="#b71c1c", linestyle="--", lw=2, label=f"Media: {vs_b['media']:.1%}")
    ax[1].axvline(vs_b["ic95_inf"], color="#1565c0", linestyle=":", lw=1.6, label=f"IC 95%: [{vs_b['ic95_inf']:.1%}, {vs_b['ic95_sup']:.1%}]")
    ax[1].axvline(vs_b["ic95_sup"], color="#1565c0", linestyle=":", lw=1.6)
    ax[1].set_title("B) Distribución de Precisión de Viterbi (500 Réplicas)", fontweight="bold", fontsize=10)
    ax[1].set_xlabel("Precisión de clasificación")
    ax[1].set_ylabel("Frecuencia de corridas")
    ax[1].legend(fontsize=8.5)
    ax[1].grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(args.salida, "montecarlo.png"), dpi=300)
    plt.close()

    # Figura 4: Comparación Multi-Escenario (E1 a E4) + Matriz de Confusión (300 DPI)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    nombres_cortos = ["E1: Base\nNominal", "E2: Inicio\nCrítico (π)", "E3: Sobrecarga\nSevera (A)", "E4: Sensor\nCalibrado (B)"]
    claves_ord = ["E1_Base_Nominal", "E2_Inicio_Critico", "E3_Degradacion_Severa", "E4_Sensor_Alta_Precision"]

    # Panel A: Precisión Media Viterbi vs Forward con IC 95% y Rango Mín-Máx
    x_pos = np.arange(len(claves_ord))
    v_means = [resultados_escenarios[k]["viterbi_stats"]["media"] * 100 for k in claves_ord]
    v_stds = [resultados_escenarios[k]["viterbi_stats"]["desv"] * 100 for k in claves_ord]
    f_means = [resultados_escenarios[k]["forward_stats"]["media"] * 100 for k in claves_ord]
    f_stds = [resultados_escenarios[k]["forward_stats"]["desv"] * 100 for k in claves_ord]

    axes[0, 0].bar(x_pos - 0.18, v_means, 0.34, yerr=v_stds, capsize=4, label="Viterbi (Global ± DE)", color="#2e7d32", alpha=0.9)
    axes[0, 0].bar(x_pos + 0.18, f_means, 0.34, yerr=f_stds, capsize=4, label="Forward (Puntual ± DE)", color="#1565c0", alpha=0.9)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(nombres_cortos, fontsize=8.5)
    axes[0, 0].set_ylabel("Precisión de Inferencia (%)")
    axes[0, 0].set_ylim(50, 105)
    axes[0, 0].set_title("A) Precisión Viterbi vs. Filtrado Forward por Escenario (±1 DE)", fontweight="bold", fontsize=10)
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Distribución Estacionaria Empírica de Estados por Escenario
    f_sano = [resultados_escenarios[k]["frec_est"][0] * 100 for k in claves_ord]
    f_desg = [resultados_escenarios[k]["frec_est"][1] * 100 for k in claves_ord]
    f_crit = [resultados_escenarios[k]["frec_est"][2] * 100 for k in claves_ord]

    axes[0, 1].bar(x_pos, f_sano, 0.48, label="Sano", color="#4caf50")
    axes[0, 1].bar(x_pos, f_desg, 0.48, bottom=f_sano, label="Desgaste", color="#ffb300")
    axes[0, 1].bar(x_pos, f_crit, 0.48, bottom=np.array(f_sano) + np.array(f_desg), label="Crítico", color="#e53935")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(nombres_cortos, fontsize=8.5)
    axes[0, 1].set_ylabel("Proporción de Tiempo Simulado (%)")
    axes[0, 1].set_title("B) Sensibilidad de la Distribución de Estados ante Cambios en π, A y B", fontweight="bold", fontsize=10)
    axes[0, 1].legend(loc="upper right", fontsize=8.5)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel C: Boxplot comparativo de las 500 corridas de Viterbi en los 4 escenarios
    datos_box = [resultados_escenarios[k]["accs_viterbi_raw"] * 100 for k in claves_ord]
    bp = axes[1, 0].boxplot(datos_box, tick_labels=nombres_cortos, patch_artist=True, showmeans=True)
    colores_box = ["#c8e6c9", "#bbdefb", "#ffe0b2", "#e1bee7"]
    for patch, col in zip(bp["boxes"], colores_box):
        patch.set_facecolor(col)
    axes[1, 0].set_ylabel("Precisión de Viterbi (%)")
    axes[1, 0].set_title("C) Dispersión, Rango (Mín–Máx) y Consistencia en 500 Réplicas", fontweight="bold", fontsize=10)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel D: Matriz de Confusión Normalizada (%) en el Escenario Base
    cm = res_base["conf_matrix"].astype(float)
    cm_norm = (cm / cm.sum(axis=1, keepdims=True)) * 100
    im = axes[1, 1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(ESTADOS)
    axes[1, 1].set_yticklabels(ESTADOS)
    axes[1, 1].set_xlabel("Estado Inferido por Viterbi")
    axes[1, 1].set_ylabel("Estado Oculto Real")
    axes[1, 1].set_title("D) Matriz de Confusión Normalizada (% por Estado Real - E1)", fontweight="bold", fontsize=10)
    for i in range(3):
        for j in range(3):
            color_txt = "white" if cm_norm[i, j] > 50 else "black"
            axes[1, 1].text(j, i, f"{cm_norm[i, j]:.1f}%\n({int(cm[i, j])})",
                            ha="center", va="center", color=color_txt, fontsize=8.5, fontweight="bold")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(args.salida, "comparacion_escenarios.png"), dpi=300)
    plt.close()

    # Figura 5: Diagrama de estados HMM y Flujo Metodológico (300 DPI)
    generar_diagrama_metodologico(os.path.join(args.salida, "diagrama_modelo_hmm.png"))

    # Copiar figuras principales también a la raíz del proyecto para compatibilidad total con el repositorio original
    for fig_name in ["secuencia.png", "filtrado.png", "montecarlo.png", "comparacion_escenarios.png", "diagrama_modelo_hmm.png"]:
        src = os.path.join(args.salida, fig_name)
        dst = fig_name
        if os.path.exists(src):
            with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                f_out.write(f_in.read())


if __name__ == "__main__":
    main()
