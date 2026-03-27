"""
Diseño con Margen de Shadowing y Cobertura al 95%
==================================================
Escenario: cobertura LTE para terminales de inventario y telefonía
corporativa en naves y muelles de carga.  Se garantiza 95 % de cobertura
en borde de celda y se estima la cobertura de área resultante.

Ejecutar::

    python diseno_shadowing_cobertura.py

Los resultados se imprimen por consola y las figuras se guardan como PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate

# ── Reproducibilidad ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Estilo de figuras ─────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.4})


# =============================================================================
# 1. Parámetros del sistema
# =============================================================================

PIRE_dBm   = 46.0   # PIRE de la estación base (dBm)
sens_dBm   = -95.0  # Sensibilidad del terminal LTE (dBm)
L_penet_dB = 20.0   # Pérdida de penetración en nave industrial (dB)

# Presupuesto máximo de pérdidas de propagación (MAPL) sin margen de shadowing
MAPL_dB = PIRE_dBm - sens_dBm - L_penet_dB

# Modelo ETSI TR 36.942 macrocelda: PL(d) = 128.1 + 37.6·log10(d_km)
PL_A = 128.1   # término constante (dB)
PL_B = 37.6    # coeficiente de pendiente (dB/dec)

# Radio nominal de celda sin margen
d0_km = 10 ** ((MAPL_dB - PL_A) / PL_B)

# Objetivo de cobertura en borde y rango de sigma
p_borde   = 0.95
sigmas_dB = np.array([6, 7, 8, 9, 10], dtype=float)


# =============================================================================
# 2. Aproximación numérica del percentil gaussiano (Abramowitz & Stegun 26.2.17)
# =============================================================================

def ppf_approx(p: float) -> float:
    """Aproximación racional del percentil gaussiano.

    Válida para 0.5 <= p < 1 con error máximo < 4.5e-4.
    """
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    num = c[0] + c[1] * t + c[2] * t**2
    den = 1.0 + d[0] * t + d[1] * t**2 + d[2] * t**3
    return t - num / den


z95_exacto = stats.norm.ppf(p_borde)
z95_aprox  = ppf_approx(p_borde)

print("=" * 60)
print("2. PERCENTIL GAUSSIANO")
print("=" * 60)
print(f"  Φ⁻¹(0.95) exacto  = {z95_exacto:.6f}")
print(f"  Φ⁻¹(0.95) aprox.  = {z95_aprox:.6f}")
print(f"  Error absoluto     = {abs(z95_exacto - z95_aprox):.2e}")

# Figura: error de la aproximación
p_range = np.linspace(0.50, 0.999, 500)
error   = np.abs(np.vectorize(ppf_approx)(p_range) - stats.norm.ppf(p_range))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.semilogy(p_range, error, color="steelblue")
ax.axvline(0.95, color="crimson", linestyle="--", label="p = 0.95")
ax.set_xlabel("Probabilidad p")
ax.set_ylabel("Error absoluto")
ax.set_title("Error de la aproximación racional de Φ⁻¹(p)")
ax.legend()
plt.tight_layout()
plt.savefig("fig_error_ppf.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 3. Margen de shadowing para distintos valores de σ
# =============================================================================

margenes_dB = z95_exacto * sigmas_dB

print("\n" + "=" * 60)
print("3. MARGEN DE SHADOWING  (objetivo borde 95 %)")
print("=" * 60)
print(f"  z_(0.95) = {z95_exacto:.4f}")
print()
print(f"  {'σ (dB)':>8} | {'M_s (dB)':>10} | {'MAPL efectivo (dB)':>20}")
print("  " + "-" * 44)
for s, m in zip(sigmas_dB, margenes_dB):
    print(f"  {s:>8.0f} | {m:>10.2f} | {MAPL_dB - m:>20.2f}")


# =============================================================================
# 4. Radio útil de celda con margen
# =============================================================================

MAPL_ef_dB = MAPL_dB - margenes_dB
radios_km  = 10 ** ((MAPL_ef_dB - PL_A) / PL_B)
radios_m   = radios_km * 1000

print("\n" + "=" * 60)
print("4. RADIO ÚTIL DE CELDA")
print("=" * 60)
print(f"  Radio sin margen (d₀) = {d0_km * 1000:.0f} m\n")
print(f"  {'σ (dB)':>8} | {'M_s (dB)':>10} | {'Radio (m)':>10} | {'Reducción (%)':>14}")
print("  " + "-" * 50)
for s, m, r in zip(sigmas_dB, margenes_dB, radios_m):
    red = (1 - r / (d0_km * 1000)) * 100
    print(f"  {s:>8.0f} | {m:>10.2f} | {r:>10.0f} | {red:>13.1f}%")

# Figura: radio vs sigma
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(sigmas_dB, radios_m, width=0.6, color="steelblue", alpha=0.8, label="Radio útil")
ax1.axhline(d0_km * 1000, color="crimson", linestyle="--",
            label=f"Radio sin margen ({d0_km*1000:.0f} m)")
ax1.set_xlabel("Desviación típica σ (dB)")
ax1.set_ylabel("Radio útil de celda (m)")
ax1.set_title("Radio útil de celda vs σ  (cobertura borde 95 %)")
ax1.legend()
ax1.set_ylim(0, d0_km * 1000 * 1.2)
ax2 = ax1.twinx()
ax2.plot(sigmas_dB, margenes_dB, "o--", color="darkorange", label="Margen M_s")
ax2.set_ylabel("Margen de shadowing M_s (dB)", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig_radio_vs_sigma.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 5. Cobertura de área  (integración numérica)
# =============================================================================

def cobertura_area(Ms_dB: float, sigma_dB: float, gamma: float = PL_B / 10) -> float:
    """Fracción del área de la celda con cobertura suficiente.

    Integra la probabilidad de cobertura sobre el disco unitario:
        C_area = ∫₀¹ 2ρ · Φ((M_s + 10γ·log10(1/ρ))/σ) dρ
    """
    def integrand(rho: float) -> float:
        if rho == 0:
            return 0.0
        arg = (Ms_dB + 10 * gamma * np.log10(1.0 / rho)) / sigma_dB
        return 2 * rho * stats.norm.cdf(arg)

    result, _ = integrate.quad(integrand, 1e-6, 1.0, limit=200)
    return result


coberturas_area = np.array([
    cobertura_area(m, s) for m, s in zip(margenes_dB, sigmas_dB)
])

print("\n" + "=" * 60)
print("5. COBERTURA DE ÁREA")
print("=" * 60)
print(f"  {'σ (dB)':>8} | {'M_s (dB)':>10} | {'Cob. borde':>12} | {'Cob. área':>12}")
print("  " + "-" * 50)
for s, m, ca in zip(sigmas_dB, margenes_dB, coberturas_area):
    print(f"  {s:>8.0f} | {m:>10.2f} | {p_borde*100:>10.1f} % | {ca*100:>10.2f} %")


# =============================================================================
# 6. CDF de la potencia recibida en borde de celda
# =============================================================================

N_muestras = 50_000
fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
colores    = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigmas_dB)))

for sigma, margen, radio_km, color in zip(sigmas_dB, margenes_dB, radios_km, colores):
    PL_borde = PL_A + PL_B * np.log10(radio_km)
    P_media  = PIRE_dBm - PL_borde - L_penet_dB
    X        = rng.normal(0, sigma, N_muestras)
    P_rx     = P_media + X
    P_sorted = np.sort(P_rx)
    cdf      = np.arange(1, N_muestras + 1) / N_muestras
    label    = f"σ = {sigma:.0f} dB  (M_s = {margen:.1f} dB)"
    axes[0].plot(P_sorted, cdf * 100, color=color, label=label)
    axes[1].plot(P_sorted, cdf * 100, color=color, label=label)

for ax in axes:
    ax.axvline(sens_dBm, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Umbral = {sens_dBm} dBm")
    ax.axhline(p_borde * 100, color="gray", linestyle=":", linewidth=1.2,
               label=f"{p_borde*100:.0f} % objetivo")
    ax.set_xlabel("Potencia recibida (dBm)")
    ax.set_ylabel("CDF (%)")
    ax.legend(fontsize=8)

axes[0].set_title("CDF de potencia recibida en borde de celda")
axes[1].set_xlim(sens_dBm - 5 * max(sigmas_dB), sens_dBm + 5 * max(sigmas_dB))
axes[1].set_ylim(0, 100)
axes[1].set_title("Zoom entorno al umbral de sensibilidad")
plt.tight_layout()
plt.savefig("fig_cdf_potencia.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 7. Mapa de calor de cobertura de área vs σ y p_borde
# =============================================================================

p_valores   = np.linspace(0.80, 0.99, 20)
sig_valores = np.linspace(4, 12, 17)
mapa = np.zeros((len(sig_valores), len(p_valores)))
for i, sig in enumerate(sig_valores):
    for j, pv in enumerate(p_valores):
        mapa[i, j] = cobertura_area(stats.norm.ppf(pv) * sig, sig) * 100

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.contourf(p_valores * 100, sig_valores, mapa, levels=20, cmap="RdYlGn")
cb = fig.colorbar(im, ax=ax)
cb.set_label("Cobertura de área (%)")
cs = ax.contour(p_valores * 100, sig_valores, mapa,
                levels=[90, 95, 97, 99], colors="k", linewidths=0.8)
ax.clabel(cs, fmt="%.0f%%", inline=True, fontsize=8)
ax.plot(95, 8, "w*", markersize=14, label="Punto nominal (σ=8 dB, 95%)")
ax.set_xlabel("Objetivo de cobertura en borde (%)")
ax.set_ylabel("Desviación típica shadowing σ (dB)")
ax.set_title("Cobertura de área (%) en función de σ y objetivo de borde")
ax.legend()
plt.tight_layout()
plt.savefig("fig_mapa_cobertura.png", bbox_inches="tight")
plt.close()


# =============================================================================
# 8. Tabla resumen
# =============================================================================

print("\n" + "=" * 75)
print("8. TABLA RESUMEN: MARGEN vs σ")
print("=" * 75)
header = f"  {'σ':>5} | {'M_s':>8} | {'MAPL_ef':>9} | {'Radio':>7} | {'Área':>9} | {'C.borde':>8} | {'C.área':>8}"
print(header)
print(f"  {'(dB)':>5} | {'(dB)':>8} | {'(dB)':>9} | {'(m)':>7} | {'(km²)':>9} | {'':>8} | {'':>8}")
print("  " + "-" * 70)
for s, m, r_m, ca in zip(sigmas_dB, margenes_dB, radios_m, coberturas_area):
    area_km2 = np.pi * (r_m / 1000) ** 2
    print(
        f"  {s:>5.0f} | {m:>8.2f} | {MAPL_dB - m:>9.1f} | {r_m:>7.0f} | "
        f"{area_km2:>9.4f} | {p_borde*100:>7.1f}% | {ca*100:>7.2f}%"
    )


# =============================================================================
# 9. Extensión: coste de celda adicional
# =============================================================================

COSTE_CELDA_EUR = 80_000
N_celdas        = (d0_km / radios_km) ** 2
coste_extra     = (N_celdas - 1) * COSTE_CELDA_EUR

print("\n" + "=" * 60)
print("9. EXTENSIÓN: COSTE DE DENSIFICACIÓN")
print("=" * 60)
print(f"  Coste de referencia por celda: {COSTE_CELDA_EUR:,} €\n")
print(f"  {'σ (dB)':>8} | {'M_s (dB)':>10} | {'N celdas':>10} | {'Coste extra (€)':>16}")
print("  " + "-" * 52)
for s, m, nc, ce in zip(sigmas_dB, margenes_dB, N_celdas, coste_extra):
    print(f"  {s:>8.0f} | {m:>10.2f} | {nc:>10.2f} | {ce:>16,.0f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(sigmas_dB, coste_extra / 1e3, width=0.6, color="darkorange", alpha=0.8)
ax.set_xlabel("Desviación típica σ (dB)")
ax.set_ylabel("Coste extra de densificación (k€)")
ax.set_title("Coste adicional de infraestructura por efecto del shadowing")
ax2 = ax.twinx()
ax2.plot(sigmas_dB, N_celdas, "s--", color="steelblue", label="Nº celdas necesarias")
ax2.set_ylabel("Nº de celdas equivalentes", color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")
ax2.legend(loc="upper left")
plt.tight_layout()
plt.savefig("fig_coste_celdas.png", bbox_inches="tight")
plt.close()

print("\nFiguras guardadas: fig_error_ppf.png, fig_radio_vs_sigma.png,")
print("                   fig_cdf_potencia.png, fig_mapa_cobertura.png,")
print("                   fig_coste_celdas.png")
