import scipy.io
import numpy as np
import matplotlib.pyplot as plt




# 1. Charger le fichier MATLAB
data = scipy.io.loadmat(r'C:\Users\user\Desktop\Donnees_KVI\Kelvin_Helmholtz_Instabilities_B_0\MHD_data_kelvin_helmholtz_B_0\MHD_data_kelvin_helmholtz_B_0_t0.mat')

# Analyser les champs numériques (pas params qui est une structure)
numerical_fields = ['J', 'u1', 'u2', 'w', 'B1', 'B2']

print("=== ANALYSE DES DONNÉES MHD ===")
print(f"Résolution spatiale: 512 x 512")
print(f"Temps: t = 0 (condition initiale)")

print("\nChamps disponibles:")
for field in numerical_fields:
    var = data[field]
    print(f"\n{field}:")
    print(f"  Shape: {var.shape}")
    print(f"  Min: {np.min(var):.6f}")
    print(f"  Max: {np.max(var):.6f}")
    print(f"  Mean: {np.mean(var):.6f}")
    print(f"  Std: {np.std(var):.6f}")

# Essayer d'extraire quelques paramètres de la structure params
print("\n=== PARAMÈTRES DE SIMULATION ===")
try:
    params_struct = data['params'][0,0]
    print(f"Équation: {params_struct['eqname'][0] if len(params_struct['eqname']) > 0 else 'N/A'}")
    print(f"Champ magnétique B0: {params_struct['B0'][0][0] if len(params_struct['B0']) > 0 else 'N/A'}")
    print(f"Domaine Lx: {params_struct['Lx'][0][0] if len(params_struct['Lx']) > 0 else 'N/A'}")
    print(f"Domaine Ly: {params_struct['Ly'][0][0] if len(params_struct['Ly']) > 0 else 'N/A'}")
    print(f"Temps final: {params_struct['T_end'][0][0] if len(params_struct['T_end']) > 0 else 'N/A'}")
    print(f"Pas de temps: {params_struct['dt'][0][0] if len(params_struct['dt']) > 0 else 'N/A'}")
except:
    print("Impossible d'extraire les paramètres détaillés")

## Le PNG montre 4 champs différents : vorticity, current density, velocity, et magnetic field
## u1, u2 : Composantes de vitesse (u1 = vitesse horizontale, u2 = vitesse verticale)
## w : Vorticité (tourbillon du fluide)
## J : Densité de courant électrique
## B1, B2 : Composantes du champ magnétique
##  params : Paramètres de simulation
## Résolution spatiale/temporelle :
## Spatiale : 512 × 512 points
## Domaine : 6.28 × 6.28 (≈ 2π × 2π)
## Temporelle : dt = 0.001, simulation jusqu'à T = 50
## 35 fichiers → des snapshots temporels
## B = 0 : C'est bien le cas idéal (pas de champ magnétique)
## J = 0 et B1 = B2 = 0 : Cohérent avec B₀ = 0
## Instabilité KH : On voit la couche de cisaillement dans la vorticité
## Condition initiale : t = 0, l'instabilité va se développer dans le temps

## Singularités potentielles à chercher :
## - Zones de fort gradient de vorticité (interfaces entre fluides)
## - Points de stagnation dans le champ de vitesse
## - Structures tourbillonnaires qui vont apparaître

# 2 Analyse des coupes 1D de la vorticité

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt


# Charger les données

w = data['w']  # Vorticité

# Créer la grille spatiale
Lx, Ly = 2*np.pi, 2*np.pi  # Domaine 2π x 2π
x = np.linspace(0, Lx, 512)
y = np.linspace(0, Ly, 512)

# Faire plusieurs coupes 1D intéressantes
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Coupe horizontale au milieu (y = π)
y_mid = 256  # milieu en indices
w_horizontal = w[y_mid, :]

axes[0,0].plot(x, w_horizontal, 'b-', linewidth=2)
axes[0,0].set_title('Coupe horizontale de la vorticité (y = π)')
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('ω(x, π)')
axes[0,0].grid(True, alpha=0.3)

# 2. Coupe verticale au milieu (x = π)
x_mid = 256  # milieu en indices
w_vertical = w[:, x_mid]

axes[0,1].plot(y, w_vertical, 'r-', linewidth=2)
axes[0,1].set_title('Coupe verticale de la vorticité (x = π)')
axes[0,1].set_xlabel('y')
axes[0,1].set_ylabel('ω(π, y)')
axes[0,1].grid(True, alpha=0.3)

# 3. Champ 2D complet pour référence
im = axes[1,0].imshow(w, extent=[0, 2*np.pi, 0, 2*np.pi], 
                      cmap='RdBu_r', origin='lower')
axes[1,0].axhline(y=np.pi, color='blue', linestyle='--', alpha=0.7, label='Coupe horizontale')
axes[1,0].axvline(x=np.pi, color='red', linestyle='--', alpha=0.7, label='Coupe verticale')
axes[1,0].set_title('Champ de vorticité 2D')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('y')
axes[1,0].legend()
plt.colorbar(im, ax=axes[1,0])

# 4. Statistiques des coupes
axes[1,1].hist(w_horizontal, bins=30, alpha=0.7, label='Coupe horizontale', color='blue')
axes[1,1].hist(w_vertical, bins=30, alpha=0.7, label='Coupe verticale', color='red')
axes[1,1].set_title('Distribution des valeurs de vorticité')
axes[1,1].set_xlabel('ω')
axes[1,1].set_ylabel('Fréquence')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=== ANALYSE DES COUPES 1D ===")
print(f"Coupe horizontale - Min: {np.min(w_horizontal):.4f}, Max: {np.max(w_horizontal):.4f}")
print(f"Coupe verticale - Min: {np.min(w_vertical):.4f}, Max: {np.max(w_vertical):.4f}")
print(f"Résolution spatiale: Δx = Δy = {2*np.pi/512:.4f}")

## Observations :
## Coupe horizontale : Presque constante (≈ -1.59) → pas de variation dans cette direction
## Coupe verticale : Transition nette de -1.59 à 0 → c'est là qu'est l'instabilité !
## Gradient maximal à y ≈ 2.94 → zone de cisaillement de l'instabilité KH
## C'est typique de Kelvin-Helmholtz : une interface entre deux fluides avec des vitesses différentes.

## 3. exploration d'autres coupes 1D (diagonales, autres positions)

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
w = data['w']  # Vorticité

# Créer la grille spatiale
Lx, Ly = 2*np.pi, 2*np.pi
x = np.linspace(0, Lx, 512)
y = np.linspace(0, Ly, 512)

fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# 1. Coupes horizontales à différentes hauteurs
y_positions = [128, 256, 384]  # y ≈ π/2, π, 3π/2
colors = ['green', 'blue', 'orange']
labels = ['y ≈ π/2', 'y ≈ π', 'y ≈ 3π/2']

for i, (y_pos, color, label) in enumerate(zip(y_positions, colors, labels)):
    w_h = w[y_pos, :]
    axes[0,0].plot(x, w_h, color=color, linewidth=2, label=label)
    print(f"Coupe {label}: Min={np.min(w_h):.4f}, Max={np.max(w_h):.4f}, Std={np.std(w_h):.6f}")

axes[0,0].set_title('Coupes horizontales à différentes hauteurs')
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('ω(x, y)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Coupes verticales à différentes positions x
x_positions = [128, 256, 384]  # x ≈ π/2, π, 3π/2
for i, (x_pos, color, label) in enumerate(zip(x_positions, colors, ['x ≈ π/2', 'x ≈ π', 'x ≈ 3π/2'])):
    w_v = w[:, x_pos]
    axes[0,1].plot(y, w_v, color=color, linewidth=2, label=label)
    print(f"Coupe {label}: Min={np.min(w_v):.4f}, Max={np.max(w_v):.4f}, Std={np.std(w_v):.6f}")

axes[0,1].set_title('Coupes verticales à différentes positions x')
axes[0,1].set_xlabel('y')
axes[0,1].set_ylabel('ω(x, y)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Coupes diagonales
# Diagonale principale (0,0) -> (2π, 2π)
diag_indices = np.arange(512)
w_diag_main = w[diag_indices, diag_indices]
diag_coord = np.linspace(0, 2*np.pi, 512)

axes[1,0].plot(diag_coord, w_diag_main, 'purple', linewidth=2, label='Diagonale principale')

# Diagonale secondaire (0, 2π) -> (2π, 0)
w_diag_sec = w[511-diag_indices, diag_indices]
axes[1,0].plot(diag_coord, w_diag_sec, 'brown', linewidth=2, label='Diagonale secondaire')

axes[1,0].set_title('Coupes diagonales')
axes[1,0].set_xlabel('Position le long de la diagonale')
axes[1,0].set_ylabel('ω')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

print(f"Diagonale principale: Min={np.min(w_diag_main):.4f}, Max={np.max(w_diag_main):.4f}")
print(f"Diagonale secondaire: Min={np.min(w_diag_sec):.4f}, Max={np.max(w_diag_sec):.4f}")

# 4. Analyse des gradients pour toutes les coupes
gradients_h = [np.max(np.abs(np.gradient(w[y_pos, :]))) for y_pos in y_positions]
gradients_v = [np.max(np.abs(np.gradient(w[:, x_pos]))) for x_pos in x_positions]
grad_diag_main = np.max(np.abs(np.gradient(w_diag_main)))
grad_diag_sec = np.max(np.abs(np.gradient(w_diag_sec)))

bar_labels = ['y≈π/2', 'y≈π', 'y≈3π/2', 'x≈π/2', 'x≈π', 'x≈3π/2', 'Diag 1', 'Diag 2']
bar_values = gradients_h + gradients_v + [grad_diag_main, grad_diag_sec]
bar_colors = ['green', 'blue', 'orange', 'green', 'blue', 'orange', 'purple', 'brown']

axes[1,1].bar(bar_labels, bar_values, color=bar_colors, alpha=0.7)
axes[1,1].set_title('Gradients maximaux par coupe')
axes[1,1].set_ylabel('|∇ω|_max')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

# 5. Champ 2D avec toutes les coupes superposées
im = axes[2,0].imshow(w, extent=[0, 2*np.pi, 0, 2*np.pi], 
                      cmap='RdBu_r', origin='lower', alpha=0.8)

# Tracer les lignes de coupe
for i, y_pos in enumerate(y_positions):
    axes[2,0].axhline(y=y_pos*2*np.pi/512, color=colors[i], linestyle='-', alpha=0.8, linewidth=2)
for i, x_pos in enumerate(x_positions):
    axes[2,0].axvline(x=x_pos*2*np.pi/512, color=colors[i], linestyle='-', alpha=0.8, linewidth=2)

# Diagonales
axes[2,0].plot([0, 2*np.pi], [0, 2*np.pi], 'purple', linewidth=2, alpha=0.8, label='Diag principale')
axes[2,0].plot([0, 2*np.pi], [2*np.pi, 0], 'brown', linewidth=2, alpha=0.8, label='Diag secondaire')

axes[2,0].set_title('Champ 2D avec toutes les coupes')
axes[2,0].set_xlabel('x')
axes[2,0].set_ylabel('y')
plt.colorbar(im, ax=axes[2,0])

# 6. Profil détaillé de la zone de transition
# Focus sur la zone où le gradient est maximal (autour de y ≈ 3)
y_focus = np.linspace(2.5, 3.5, 100)
y_indices = (y_focus * 512 / (2*np.pi)).astype(int)
y_indices = np.clip(y_indices, 0, 511)

w_focus = w[y_indices, 256]  # coupe verticale au milieu

axes[2,1].plot(y_focus, w_focus, 'red', linewidth=3, marker='o', markersize=3)
axes[2,1].set_title('Zoom sur la zone de transition')
axes[2,1].set_xlabel('y')
axes[2,1].set_ylabel('ω(π, y)')
axes[2,1].grid(True, alpha=0.3)

# Calculer la largeur de la zone de transition
transition_indices = np.where((w_focus > -1.5) & (w_focus < -0.1))[0]
if len(transition_indices) > 0:
    transition_width = y_focus[transition_indices[-1]] - y_focus[transition_indices[0]]
    print(f"\nLargeur de la zone de transition: {transition_width:.4f}")
    axes[2,1].axvspan(y_focus[transition_indices[0]], y_focus[transition_indices[-1]], 
                      alpha=0.3, color='yellow', label=f'Zone transition (Δy={transition_width:.3f})')
    axes[2,1].legend()

plt.tight_layout()
plt.show()

## Découvertes importantes :
# # 1. Structure spatiale claire :
# # Couches horizontales : y ≈ π/2 et 3π/2 → vorticité ≈ 0 (fluide au repos)
# # Couche centrale : y ≈ π → vorticité ≈ -1.59 (fluide en mouvement)
# # Toutes les coupes verticales sont identiques → symétrie parfaite en x

# # 2. Zone de transition :
# # Largeur : ~0.99 ≈ π/3 (assez large pour t=0)
# # Gradients maximaux : ~0.048 dans les coupes verticales
# # Transition douce : pas de discontinuité brutale

# # 3. Géométrie de l'instabilité :
# # Invariance en x : l'instabilité n'a pas encore développé de structures 2D
# # Profil en marche d'escalier : transition entre deux états de vorticité
# # Diagonales : montrent la même transition → confirme la structure en couches

# # Implications pour les ondelettes :
# # Signal 1D optimal : Les coupes verticales (n'importe laquelle !)
# # Singularités attendues : Aux interfaces y ≈ 2.5 et y ≈ 3.5
# # Échelles caractéristiques : Largeur de transition ~1, domaine total 2π

# 4. analyser la coupe verticale avec la transformée en ondelettes continue (CWT) et l'ondelette de Morlet

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Charger les données
w = data['w']

# Prendre une coupe verticale (toutes sont identiques)
y = np.linspace(0, 2*np.pi, 512)
signal = w[:, 256]  # coupe verticale au milieu

print("=== ANALYSE PAR ONDELETTES CONTINUES ===")
print(f"Signal: {len(signal)} points")
print(f"Domaine: [0, 2π] ≈ [0, {2*np.pi:.3f}]")
print(f"Résolution: Δy = {2*np.pi/512:.4f}")

# Définir les échelles pour la CWT
# Échelles de 1 à 64 points (couvrant différentes résolutions)
scales = np.arange(1, 65)
dt = 2*np.pi/512  # pas spatial

# Transformée en ondelettes continue avec Morlet
print("\nCalcul de la CWT avec ondelette de Morlet...")
coefficients, frequencies = pywt.cwt(signal, scales, 'cmor1.5-1.0', dt)

print(f"Matrice des coefficients: {coefficients.shape}")
print(f"Échelles: {len(scales)} de {scales[0]} à {scales[-1]}")
print(f"Fréquences: de {frequencies[0]:.4f} à {frequencies[-1]:.4f}")

# Créer la visualisation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Signal original
axes[0,0].plot(y, signal, 'b-', linewidth=2)
axes[0,0].set_title('Signal original (coupe verticale de vorticité)')
axes[0,0].set_xlabel('y')
axes[0,0].set_ylabel('ω(π, y)')
axes[0,0].grid(True, alpha=0.3)

# Marquer les zones de transition
transition_zones = [2.5, 3.5]  # approximativement
for tz in transition_zones:
    axes[0,0].axvline(x=tz, color='red', linestyle='--', alpha=0.7, 
                      label=f'Transition ≈ {tz}')
axes[0,0].legend()

# 2. Scalogramme (coefficients CWT)
im1 = axes[0,1].imshow(np.abs(coefficients), extent=[0, 2*np.pi, scales[0], scales[-1]], 
                       cmap='jet', aspect='auto', origin='lower')
axes[0,1].set_title('Scalogramme |CWT(y, échelle)|')
axes[0,1].set_xlabel('y')
axes[0,1].set_ylabel('Échelle')
plt.colorbar(im1, ax=axes[0,1])

# Marquer les zones de transition sur le scalogramme
for tz in transition_zones:
    axes[0,1].axvline(x=tz, color='white', linestyle='--', alpha=0.8, linewidth=2)

# 3. Coefficients à différentes échelles
selected_scales = [2, 8, 16, 32]
colors = ['red', 'green', 'blue', 'orange']

for scale_idx, color in zip(selected_scales, colors):
    if scale_idx < len(scales):
        coeff_at_scale = np.abs(coefficients[scale_idx-1, :])  # -1 car indices commencent à 0
        axes[1,0].plot(y, coeff_at_scale, color=color, linewidth=2, 
                       label=f'Échelle {scale_idx}')

axes[1,0].set_title('Coefficients d\'ondelettes à différentes échelles')
axes[1,0].set_xlabel('y')
axes[1,0].set_ylabel('|CWT|')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. Détection de singularités
# Les maxima locaux des coefficients indiquent des singularités
max_coeffs_per_position = np.max(np.abs(coefficients), axis=0)

axes[1,1].plot(y, max_coeffs_per_position, 'purple', linewidth=3, label='Max |CWT| par position')
axes[1,1].set_title('Détection de singularités')
axes[1,1].set_xlabel('y')
axes[1,1].set_ylabel('max_échelles |CWT(y, échelle)|')
axes[1,1].grid(True, alpha=0.3)

# Trouver les pics (singularités potentielles)
from scipy.signal import find_peaks
peaks, properties = find_peaks(max_coeffs_per_position, height=np.max(max_coeffs_per_position)*0.3)

print(f"\n=== SINGULARITÉS DÉTECTÉES ===")
print(f"Nombre de pics détectés: {len(peaks)}")
for i, peak in enumerate(peaks):
    y_pos = y[peak]
    intensity = max_coeffs_per_position[peak]
    print(f"Singularité {i+1}: y = {y_pos:.3f}, intensité = {intensity:.4f}")
    axes[1,1].plot(y_pos, intensity, 'ro', markersize=8, 
                   label=f'Singularité {i+1}' if i < 3 else "")

axes[1,1].legend()

plt.tight_layout()
plt.show()

# Analyse quantitative des singularités
print(f"\n=== ANALYSE QUANTITATIVE ===")
print(f"Coefficient maximal: {np.max(np.abs(coefficients)):.4f}")
print(f"Position du maximum global: y = {y[np.argmax(max_coeffs_per_position)]:.3f}")
print(f"Échelle optimale: {scales[np.unravel_index(np.argmax(np.abs(coefficients)), coefficients.shape)[0]]}")


## Observations :
# # scalogramme pour visualiser les coefficients à toutes les échelles
# # Détecter automatiquement les singularités via les maxima des coefficients
# # Analyser quantitativement les positions et intensités des singularités
# # Ce qu'on va découvrir :
# # Localisation précise des zones de transition (singularités)
# # Échelles caractéristiques de l'instabilité
# # Intensité relative des différentes singularités
# # Comparaison avec notre analyse manuelle précédente

## 5 Evolution temporelle de l'instabilité KH

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

# Chemin correct vers vos données
data_path = r'C:\Users\user\Desktop\Donnees_KVI\Kelvin_Helmholtz_Instabilities_B_0\MHD_data_kelvin_helmholtz_B_0'

print(f"=== ANALYSE TEMPORELLE DE L'INSTABILITÉ KH ===")
print(f"Chemin des données: {data_path}")

# Créer la liste des fichiers de t=0 à t=50
mat_files = []
times = []

for t in range(51):  # 0 à 50 inclus
    filename = f'MHD_data_kelvin_helmholtz_B_0_t{t}.mat'
    filepath = os.path.join(data_path, filename)
    mat_files.append(filepath)
    times.append(float(t))

print(f"Fichiers à analyser: {len(mat_files)} (t=0 à t=50)")

# Sélectionner des temps représentatifs pour l'analyse détaillée
selected_times_indices = [0, 5, 10, 20, 30, 40, 50]  # t = 0, 5, 10, 20, 30, 40, 50
selected_files = [mat_files[i] for i in selected_times_indices]
selected_times = [times[i] for i in selected_times_indices]

print(f"\nTemps sélectionnés pour analyse détaillée:")
for i, time in enumerate(selected_times):
    print(f"  {i+1}. t = {time:.0f}")

# Analyser l'évolution des propriétés
evolution_data = []
y = np.linspace(0, 2*np.pi, 512)

print(f"\nAnalyse des propriétés pour chaque temps sélectionné...")

# Fonction pour analyser un fichier
def analyze_file(filepath, time):
    try:
        if not os.path.exists(filepath):
            print(f"  ⚠️  Fichier manquant: t = {time}")
            return None
            
        data = scipy.io.loadmat(filepath)
        w = data['w']
        
        # Coupe verticale au milieu
        signal = w[:, 256]
        
        # Calculs des propriétés
        signal_std = np.std(signal)
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        signal_range = signal_max - signal_min
        
        # Analyse des gradients
        gradient = np.gradient(signal)
        max_gradient = np.max(np.abs(gradient))
        mean_abs_gradient = np.mean(np.abs(gradient))
        
        # Détection des zones de forte variation
        high_grad_threshold = max_gradient * 0.3
        high_grad_indices = np.where(np.abs(gradient) > high_grad_threshold)[0]
        n_transition_zones = len(high_grad_indices)
        
        # Analyse 2D - variance spatiale
        w_variance_2d = np.var(w)
        w_mean_2d = np.mean(w)
        
        # Asymétrie en x (mesure du développement 2D)
        x_profiles = [w[:, i] for i in [128, 256, 384]]  # 3 coupes verticales
        x_profile_stds = [np.std(profile) for profile in x_profiles]
        x_asymmetry = np.std(x_profile_stds)  # Si > 0, développement 2D
        
        return {
            'time': time,
            'signal_std': signal_std,
            'signal_range': signal_range,
            'max_gradient': max_gradient,
            'mean_gradient': mean_abs_gradient,
            'n_transitions': n_transition_zones,
            'variance_2d': w_variance_2d,
            'x_asymmetry': x_asymmetry,
            'signal': signal,
            'w_2d': w
        }
        
    except Exception as e:
        print(f"  ❌ Erreur pour t = {time}: {e}")
        return None

# Analyser les temps sélectionnés
for i, (filepath, time) in enumerate(zip(selected_files, selected_times)):
    result = analyze_file(filepath, time)
    if result:
        evolution_data.append(result)
        print(f"  ✅ t = {time:.0f}: std = {result['signal_std']:.4f}, max_grad = {result['max_gradient']:.4f}")

print(f"\nAnalyse terminée pour {len(evolution_data)} temps.")

# Créer les visualisations
if len(evolution_data) > 1:
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Extraire les données pour les plots
    times_plot = [d['time'] for d in evolution_data]
    stds = [d['signal_std'] for d in evolution_data]
    ranges = [d['signal_range'] for d in evolution_data]
    max_grads = [d['max_gradient'] for d in evolution_data]
    mean_grads = [d['mean_gradient'] for d in evolution_data]
    n_trans = [d['n_transitions'] for d in evolution_data]
    var_2d = [d['variance_2d'] for d in evolution_data]
    x_asym = [d['x_asymmetry'] for d in evolution_data]
    
    # 1. Évolution de la variance (complexité)
    axes[0,0].plot(times_plot, stds, 'bo-', linewidth=2, markersize=6)
    axes[0,0].set_title('Évolution de la variance de vorticité')
    axes[0,0].set_xlabel('Temps')
    axes[0,0].set_ylabel('σ(ω)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Évolution du gradient maximal
    axes[0,1].plot(times_plot, max_grads, 'ro-', linewidth=2, markersize=6)
    axes[0,1].set_title('Gradient maximal (singularités)')
    axes[0,1].set_xlabel('Temps')
    axes[0,1].set_ylabel('max|∇ω|')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Développement 2D (asymétrie en x)
    axes[0,2].plot(times_plot, x_asym, 'go-', linewidth=2, markersize=6)
    axes[0,2].set_title('Développement 2D (asymétrie)')
    axes[0,2].set_xlabel('Temps')
    axes[0,2].set_ylabel('Asymétrie en x')
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Variance 2D totale
    axes[1,0].plot(times_plot, var_2d, 'mo-', linewidth=2, markersize=6)
    axes[1,0].set_title('Variance 2D totale')
    axes[1,0].set_xlabel('Temps')
    axes[1,0].set_ylabel('Var(ω) 2D')
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Nombre de zones de transition
    axes[1,1].plot(times_plot, n_trans, 'co-', linewidth=2, markersize=6)
    axes[1,1].set_title('Zones de forte variation')
    axes[1,1].set_xlabel('Temps')
    axes[1,1].set_ylabel('Nombre de zones')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Range des valeurs
    axes[1,2].plot(times_plot, ranges, 'yo-', linewidth=2, markersize=6)
    axes[1,2].set_title('Étendue des valeurs')
    axes[1,2].set_xlabel('Temps')
    axes[1,2].set_ylabel('max(ω) - min(ω)')
    axes[1,2].grid(True, alpha=0.3)
    
    # 7. Évolution des profils 1D
    colors = plt.cm.viridis(np.linspace(0, 1, len(evolution_data)))
    for i, (data_point, color) in enumerate(zip(evolution_data, colors)):
        axes[2,0].plot(y, data_point['signal'], color=color, linewidth=2, 
                      label=f't = {data_point["time"]:.0f}')
    
    axes[2,0].set_title('Évolution des profils de vorticité')
    axes[2,0].set_xlabel('y')
    axes[2,0].set_ylabel('ω(π, y)')
    axes[2,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2,0].grid(True, alpha=0.3)
    
    # 8. Champ 2D initial
    im1 = axes[2,1].imshow(evolution_data[0]['w_2d'], extent=[0, 2*np.pi, 0, 2*np.pi], 
                          cmap='RdBu_r', origin='lower')
    axes[2,1].set_title(f'Vorticité à t = {evolution_data[0]["time"]:.0f}')
    axes[2,1].set_xlabel('x')
    axes[2,1].set_ylabel('y')
    
    # 9. Champ 2D final
    im2 = axes[2,2].imshow(evolution_data[-1]['w_2d'], extent=[0, 2*np.pi, 0, 2*np.pi], 
                          cmap='RdBu_r', origin='lower')
    axes[2,2].set_title(f'Vorticité à t = {evolution_data[-1]["time"]:.0f}')
    axes[2,2].set_xlabel('x')
    axes[2,2].set_ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    # Résumé quantitatif
    print(f"\n=== RÉSUMÉ DE L'ÉVOLUTION ===")
    print(f"Variance initiale: {stds[0]:.4f} → finale: {stds[-1]:.4f} (×{stds[-1]/stds[0]:.1f})")
    print(f"Gradient max initial: {max_grads[0]:.4f} → final: {max_grads[-1]:.4f} (×{max_grads[-1]/max_grads[0]:.1f})")
    print(f"Asymétrie 2D initiale: {x_asym[0]:.6f} → finale: {x_asym[-1]:.6f}")
    print(f"Développement 2D: {'OUI' if x_asym[-1] > x_asym[0]*10 else 'FAIBLE'}")

else:
    print("Pas assez de données pour créer les visualisations.")

## Commentaires :
# 1. Évolution des singularités (Gradient maximal) :
# t=0 → t=20 : Croissance lente (×2) - phase linéaire
# t=20 → t=50 : EXPLOSION (×13.7 total !) - phase non-linéaire
# Point critique : t ≈ 20 où l'instabilité devient turbulente

# 2. Développement 2D spectaculaire :
# t=0 : Asymétrie = 0 (structure 1D parfaite)
# t=50 : Asymétrie = 0.21 → Développement 2D confirmé !
# Transition : t ≈ 20-30 où la symétrie se brise

# 3. Évolution des profils 1D :
# t=0-10 : Profil lisse (marche d'escalier)
# t=20+ : Oscillations complexes → structures tourbillonnaires !
# t=50 : Profil très chaotique avec multiples singularités

# 4. Champs 2D - Transformation dramatique :
# t=0 : Interface horizontale simple
# t=50 : Structure tourbillonnaire complexe → vortex de KH !

# Implications pour le Machine Learning :
# Caractéristiques temporelles identifiées :
# Phase linéaire (t=0-20) : Croissance exponentielle douce
# Transition critique (t≈20) : Seuil de non-linéarité
# Phase turbulente (t=20-50) : Développement 2D explosif

# Features ML potentielles :
# Variance : Indicateur de complexité
# Gradient maximal : Détecteur de singularités
# Asymétrie 2D : Mesure du développement spatial
# Profils 1D : Signatures temporelles

## 6 analyse par ondelettes de toute la série temporelle 

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from scipy.signal import find_peaks

print("=== ANALYSE TEMPORELLE COMPLÈTE PAR ONDELETTES ===")

# Chemin vers vos données
data_path = r'C:\Users\user\Desktop\Donnees_KVI\Kelvin_Helmholtz_Instabilities_B_0\MHD_data_kelvin_helmholtz_B_0'

# Paramètres d'analyse
y = np.linspace(0, 2*np.pi, 512)
scales = np.arange(1, 65)  # Échelles pour CWT
dt = 2*np.pi/512

# Structures pour stocker les résultats
all_times = []
all_singularities = []
all_max_coeffs = []
all_scalograms = []
evolution_metrics = []

print("Chargement et analyse de tous les fichiers...")

# Analyser tous les fichiers de t=0 à t=50
for t in range(51):
    filename = f'MHD_data_kelvin_helmholtz_B_0_t{t}.mat'
    filepath = os.path.join(data_path, filename)
    
    if os.path.exists(filepath):
        try:
            # Charger les données
            data = scipy.io.loadmat(filepath)
            w = data['w']
            
            # Coupe verticale au milieu
            signal = w[:, 256]
            
            # CWT avec Morlet
            coefficients, frequencies = pywt.cwt(signal, scales, 'cmor1.5-1.0', dt)
            
            # Détection de singularités
            max_coeffs_per_position = np.max(np.abs(coefficients), axis=0)
            peaks, properties = find_peaks(max_coeffs_per_position, 
                                         height=np.max(max_coeffs_per_position)*0.2)
            
            # Stocker les résultats
            singularities_positions = [y[peak] for peak in peaks]
            singularities_intensities = [max_coeffs_per_position[peak] for peak in peaks]
            
            all_times.append(t)
            all_singularities.append({
                'positions': singularities_positions,
                'intensities': singularities_intensities,
                'n_singularities': len(peaks)
            })
            all_max_coeffs.append(max_coeffs_per_position)
            all_scalograms.append(np.abs(coefficients))
            
            # Métriques d'évolution
            evolution_metrics.append({
                'time': t,
                'max_coeff_global': np.max(np.abs(coefficients)),
                'mean_coeff': np.mean(np.abs(coefficients)),
                'n_singularities': len(peaks),
                'strongest_singularity': np.max(singularities_intensities) if len(singularities_intensities) > 0 else 0,
                'signal_energy': np.sum(signal**2),
                'wavelet_energy': np.sum(np.abs(coefficients)**2)
            })
            
            if t % 10 == 0:
                print(f"  ✅ t = {t}: {len(peaks)} singularités, coeff_max = {np.max(np.abs(coefficients)):.4f}")
                
        except Exception as e:
            print(f"  ❌ Erreur pour t = {t}: {e}")
    else:
        print(f"  ⚠️  Fichier manquant: t = {t}")

print(f"\nAnalyse terminée pour {len(all_times)} temps.")

# Créer les visualisations spectaculaires
fig = plt.figure(figsize=(20, 16))

# 1. Évolution du nombre de singularités
ax1 = plt.subplot(3, 4, 1)
n_sing_evolution = [m['n_singularities'] for m in evolution_metrics]
plt.plot(all_times, n_sing_evolution, 'bo-', linewidth=2, markersize=4)
plt.title('Évolution du nombre de singularités')
plt.xlabel('Temps')
plt.ylabel('Nombre de singularités')
plt.grid(True, alpha=0.3)

# 2. Évolution de l'intensité maximale
ax2 = plt.subplot(3, 4, 2)
max_intensities = [m['strongest_singularity'] for m in evolution_metrics]
plt.plot(all_times, max_intensities, 'ro-', linewidth=2, markersize=4)
plt.title('Intensité de la singularité la plus forte')
plt.xlabel('Temps')
plt.ylabel('Intensité maximale')
plt.grid(True, alpha=0.3)

# 3. Évolution de l'énergie des ondelettes
ax3 = plt.subplot(3, 4, 3)
wavelet_energies = [m['wavelet_energy'] for m in evolution_metrics]
plt.plot(all_times, wavelet_energies, 'go-', linewidth=2, markersize=4)
plt.title('Énergie totale des ondelettes')
plt.xlabel('Temps')
plt.ylabel('Énergie CWT')
plt.grid(True, alpha=0.3)

# 4. Coefficient maximal global
ax4 = plt.subplot(3, 4, 4)
max_coeffs_global = [m['max_coeff_global'] for m in evolution_metrics]
plt.plot(all_times, max_coeffs_global, 'mo-', linewidth=2, markersize=4)
plt.title('Coefficient CWT maximal')
plt.xlabel('Temps')
plt.ylabel('max|CWT|')
plt.grid(True, alpha=0.3)

# 5. Diagramme spatio-temporel des singularités
ax5 = plt.subplot(3, 4, (5, 6))
for i, (t, sing_data) in enumerate(zip(all_times, all_singularities)):
    if sing_data['n_singularities'] > 0:
        positions = sing_data['positions']
        intensities = sing_data['intensities']
        # Taille des points proportionnelle à l'intensité
        sizes = [50 * (intensity / max(max_intensities)) for intensity in intensities]
        colors = [intensity for intensity in intensities]
        scatter = plt.scatter([t] * len(positions), positions, s=sizes, c=colors, 
                            cmap='hot', alpha=0.7, edgecolors='black', linewidth=0.5)

plt.title('Diagramme spatio-temporel des singularités')
plt.xlabel('Temps')
plt.ylabel('Position y')
plt.colorbar(scatter, label='Intensité')
plt.grid(True, alpha=0.3)

# 6. Scalogrammes à différents temps
selected_times_for_scalogram = [0, 10, 20, 30, 40, 50]
for i, t_sel in enumerate(selected_times_for_scalogram):
    if t_sel in all_times:
        idx = all_times.index(t_sel)
        ax = plt.subplot(3, 6, 13 + i)
        im = plt.imshow(all_scalograms[idx], extent=[0, 2*np.pi, scales[0], scales[-1]], 
                       cmap='jet', aspect='auto', origin='lower')
        plt.title(f't = {t_sel}')
        plt.xlabel('y')
        if i == 0:
            plt.ylabel('Échelle')
        else:
            plt.ylabel('')

# 7. Évolution des profils de coefficients maximaux
ax7 = plt.subplot(3, 4, (7, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_times_for_scalogram)))
for i, t_sel in enumerate(selected_times_for_scalogram):
    if t_sel in all_times:
        idx = all_times.index(t_sel)
        plt.plot(y, all_max_coeffs[idx], color=colors[i], linewidth=2, 
                label=f't = {t_sel}', alpha=0.8)

plt.title('Évolution des profils de coefficients maximaux')
plt.xlabel('y')
plt.ylabel('max_échelles |CWT|')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyse quantitative détaillée
print(f"\n=== ANALYSE QUANTITATIVE COMPLÈTE ===")

# Phases de développement
early_phase = [m for m in evolution_metrics if m['time'] <= 10]
middle_phase = [m for m in evolution_metrics if 10 < m['time'] <= 30]
late_phase = [m for m in evolution_metrics if m['time'] > 30]

print(f"\n📊 PHASES DE DÉVELOPPEMENT:")
print(f"Phase précoce (t=0-10):")
print(f"  - Singularités moyennes: {np.mean([m['n_singularities'] for m in early_phase]):.1f}")
print(f"  - Intensité moyenne: {np.mean([m['strongest_singularity'] for m in early_phase]):.4f}")

print(f"Phase intermédiaire (t=10-30):")
print(f"  - Singularités moyennes: {np.mean([m['n_singularities'] for m in middle_phase]):.1f}")
print(f"  - Intensité moyenne: {np.mean([m['strongest_singularity'] for m in middle_phase]):.4f}")

print(f"Phase tardive (t>30):")
print(f"  - Singularités moyennes: {np.mean([m['n_singularities'] for m in late_phase]):.1f}")
print(f"  - Intensité moyenne: {np.mean([m['strongest_singularity'] for m in late_phase]):.4f}")

# Détection des transitions
print(f"\n🔍 DÉTECTION DES TRANSITIONS:")
for i in range(1, len(evolution_metrics)):
    curr = evolution_metrics[i]
    prev = evolution_metrics[i-1]
    
    # Changement significatif du nombre de singularités
    if curr['n_singularities'] > prev['n_singularities'] * 1.5:
        print(f"  📈 Augmentation singularités à t = {curr['time']}: {prev['n_singularities']} → {curr['n_singularities']}")
    
    # Changement significatif d'intensité
    if curr['strongest_singularity'] > prev['strongest_singularity'] * 2:
        print(f"  🚀 Explosion intensité à t = {curr['time']}: {prev['strongest_singularity']:.4f} → {curr['strongest_singularity']:.4f}")

# Statistiques finales
print(f"\n📈 STATISTIQUES GLOBALES:")
print(f"Temps total analysé: {len(all_times)} points")
print(f"Singularités max simultanées: {max(n_sing_evolution)}")
print(f"Intensité maximale atteinte: {max(max_intensities):.4f}")
print(f"Croissance énergie ondelettes: ×{wavelet_energies[-1]/wavelet_energies[0]:.1f}")


## Observations finales :
# # 1. Explosion des singularités - Cascade turbulente :
# # t=0-17 : 1 singularité (phase stable)
# # t=17-23 : EXPLOSION 1 → 21 singularités
# # t=23-35 : Croissance continue → 72 singularités
# # t=35-50 : Saturation turbulente → 88 singularités max

# # 2. Transitions critiques identifiées :
# # t=17 : Premier dédoublement (1→2)
# # t=20 : Multiplication (1→6)
# # t=23 : AVALANCHE (2→21) - Point critique !
# # t=35 : Explosion finale (41→72)

# # 3. Diagramme spatio-temporel révélateur :
# # t<20 : Singularité unique à y≈π (interface KH)
# # t=20-30 : Bifurcation - nouvelles singularités
# # t>30 : Cascade complète - tout l'espace rempli

# # 4. Évolution des scalogrammes :
# # t=0 : Structure simple, échelle unique
# # t=20 : Apparition multi-échelles
# # t=50 : Turbulence développée - toutes échelles actives

# # Implications pour le Machine Learning :

# # Features temporelles identifiées :
# # Nombre de singularités : Indicateur de complexité
# # Intensité maximale : Mesure de l'énergie
# # Distribution spatiale : Caractérisation des patterns
# # Énergie ondelettes : Signature énergétique

# # Phases distinctes pour classification :
# # Phase I (t=0-17) : Linéaire, 1 singularité
# # Phase II (t=17-23) : Transition, multiplication
# # Phase III (t=23-35) : Non-linéaire, cascade
# # Phase IV (t>35) : Turbulence saturée