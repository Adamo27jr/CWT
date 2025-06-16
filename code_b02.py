import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from scipy.signal import find_peaks

def analyze_kh_case_with_plots(case_name, data_path, max_time=50):
    """
    Analyse complète d'un cas Kelvin-Helmholtz AVEC TOUTES LES VISUALISATIONS
    """
    print(f"\n{'='*60}")
    print(f"=== ANALYSE COMPLÈTE DU CAS {case_name} ===")
    print(f"{'='*60}")
    print(f"Chemin des données: {data_path}")
    print(f"Temps analysé: t=0 à t={max_time}")

    # 1. ANALYSE INITIALE (t=0)
    print(f"\n=== 1. ANALYSE DES DONNÉES INITIALES (t=0) ===")

    # CORRECTION DES PATTERNS DE FICHIERS POUR B_0_2 (même logique que B_0_1)
    if case_name == 'B_0_2x':
        filename_t0 = 'MHD_data_kelvin_helmholtz_B_0_2_t0.mat'  # SANS x
    elif case_name == 'B_0_2y':
        filename_t0 = 'MHD_data_kelvin_helmholtz_B_0_2y_t0.mat'  # AVEC y
    else:
        filename_t0 = f'MHD_data_kelvin_helmholtz_{case_name}_t0.mat'

    filepath_t0 = os.path.join(data_path, filename_t0)

    if not os.path.exists(filepath_t0):
        print(f"❌ Fichier t=0 manquant: {filepath_t0}")
        return None

    # Charger et analyser t=0
    data_t0 = scipy.io.loadmat(filepath_t0)
    numerical_fields = ['J', 'u1', 'u2', 'w', 'B1', 'B2']

    print(f"Résolution spatiale: 512 x 512")
    print(f"Champs disponibles:")

    for field in numerical_fields:
        if field in data_t0:
            var = data_t0[field]
            print(f"\n{field}:")
            print(f"  Shape: {var.shape}")
            print(f"  Min: {np.min(var):.6f}")
            print(f"  Max: {np.max(var):.6f}")
            print(f"  Mean: {np.mean(var):.6f}")
            print(f"  Std: {np.std(var):.6f}")

    # Paramètres de simulation
    print(f"\n=== PARAMÈTRES DE SIMULATION ===")
    try:
        params_struct = data_t0['params'][0,0]
        print(f"Équation: {params_struct['eqname'][0] if len(params_struct['eqname']) > 0 else 'N/A'}")
        print(f"Champ magnétique B0: {params_struct['B0'][0][0] if len(params_struct['B0']) > 0 else 'N/A'}")
        print(f"Domaine Lx: {params_struct['Lx'][0][0] if len(params_struct['Lx']) > 0 else 'N/A'}")
        print(f"Domaine Ly: {params_struct['Ly'][0][0] if len(params_struct['Ly']) > 0 else 'N/A'}")
        print(f"Temps final: {params_struct['T_end'][0][0] if len(params_struct['T_end']) > 0 else 'N/A'}")
        print(f"Pas de temps: {params_struct['dt'][0][0] if len(params_struct['dt']) > 0 else 'N/A'}")
    except:
        print("Impossible d'extraire les paramètres détaillés")

    # 2. VISUALISATION DES COUPES 1D
    print(f"\n=== 2. ANALYSE DES COUPES 1D ===")

    w = data_t0['w']
    Lx, Ly = 2*np.pi, 2*np.pi
    x = np.linspace(0, Lx, 512)
    y = np.linspace(0, Ly, 512)

    # PLOT 1: Coupes 1D principales
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Analyse des coupes 1D - Cas {case_name} (t=0)', fontsize=16)

    # Coupes principales
    y_mid = 256
    x_mid = 256
    w_horizontal = w[y_mid, :]
    w_vertical = w[:, x_mid]

    axes[0,0].plot(x, w_horizontal, 'b-', linewidth=2)
    axes[0,0].set_title('Coupe horizontale de la vorticité (y = π)')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('ω(x, π)')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(y, w_vertical, 'r-', linewidth=2)
    axes[0,1].set_title('Coupe verticale de la vorticité (x = π)')
    axes[0,1].set_xlabel('y')
    axes[0,1].set_ylabel('ω(π, y)')
    axes[0,1].grid(True, alpha=0.3)

    # Champ 2D
    im = axes[1,0].imshow(w, extent=[0, 2*np.pi, 0, 2*np.pi], 
                          cmap='RdBu_r', origin='lower')
    axes[1,0].axhline(y=np.pi, color='blue', linestyle='--', alpha=0.7, label='Coupe horizontale')
    axes[1,0].axvline(x=np.pi, color='red', linestyle='--', alpha=0.7, label='Coupe verticale')
    axes[1,0].set_title('Champ de vorticité 2D')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    axes[1,0].legend()
    plt.colorbar(im, ax=axes[1,0])

    # Histogrammes
    axes[1,1].hist(w_horizontal, bins=30, alpha=0.7, label='Coupe horizontale', color='blue')
    axes[1,1].hist(w_vertical, bins=30, alpha=0.7, label='Coupe verticale', color='red')
    axes[1,1].set_title('Distribution des valeurs de vorticité')
    axes[1,1].set_xlabel('ω')
    axes[1,1].set_ylabel('Fréquence')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Coupe horizontale - Min: {np.min(w_horizontal):.4f}, Max: {np.max(w_horizontal):.4f}")
    print(f"Coupe verticale - Min: {np.min(w_vertical):.4f}, Max: {np.max(w_vertical):.4f}")

    # PLOT 2: Coupes multiples et diagonales
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'Exploration des coupes 1D - Cas {case_name} (t=0)', fontsize=16)

    # Coupes horizontales multiples
    y_positions = [128, 256, 384]
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

    # Coupes verticales multiples
    x_positions = [128, 256, 384]
    for i, (x_pos, color, label) in enumerate(zip(x_positions, colors, ['x ≈ π/2', 'x ≈ π', 'x ≈ 3π/2'])):
        w_v = w[:, x_pos]
        axes[0,1].plot(y, w_v, color=color, linewidth=2, label=label)
        print(f"Coupe {label}: Min={np.min(w_v):.4f}, Max={np.max(w_v):.4f}, Std={np.std(w_v):.6f}")

    axes[0,1].set_title('Coupes verticales à différentes positions x')
    axes[0,1].set_xlabel('y')
    axes[0,1].set_ylabel('ω(x, y)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Diagonales
    diag_indices = np.arange(512)
    w_diag_main = w[diag_indices, diag_indices]
    w_diag_sec = w[511-diag_indices, diag_indices]
    diag_coord = np.linspace(0, 2*np.pi, 512)

    axes[1,0].plot(diag_coord, w_diag_main, 'purple', linewidth=2, label='Diagonale principale')
    axes[1,0].plot(diag_coord, w_diag_sec, 'brown', linewidth=2, label='Diagonale secondaire')
    axes[1,0].set_title('Coupes diagonales')
    axes[1,0].set_xlabel('Position le long de la diagonale')
    axes[1,0].set_ylabel('ω')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    print(f"Diagonale principale: Min={np.min(w_diag_main):.4f}, Max={np.max(w_diag_main):.4f}")
    print(f"Diagonale secondaire: Min={np.min(w_diag_sec):.4f}, Max={np.max(w_diag_sec):.4f}")

    # Gradients
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

    # Champ 2D avec coupes
    im = axes[2,0].imshow(w, extent=[0, 2*np.pi, 0, 2*np.pi], 
                          cmap='RdBu_r', origin='lower', alpha=0.8)

    for i, y_pos in enumerate(y_positions):
        axes[2,0].axhline(y=y_pos*2*np.pi/512, color=colors[i], linestyle='-', alpha=0.8, linewidth=2)
    for i, x_pos in enumerate(x_positions):
        axes[2,0].axvline(x=x_pos*2*np.pi/512, color=colors[i], linestyle='-', alpha=0.8, linewidth=2)

    axes[2,0].plot([0, 2*np.pi], [0, 2*np.pi], 'purple', linewidth=2, alpha=0.8, label='Diag principale')
    axes[2,0].plot([0, 2*np.pi], [2*np.pi, 0], 'brown', linewidth=2, alpha=0.8, label='Diag secondaire')
    axes[2,0].set_title('Champ 2D avec toutes les coupes')
    axes[2,0].set_xlabel('x')
    axes[2,0].set_ylabel('y')
    plt.colorbar(im, ax=axes[2,0])

    # Zone de transition
    y_focus = np.linspace(2.5, 3.5, 100)
    y_indices = (y_focus * 512 / (2*np.pi)).astype(int)
    y_indices = np.clip(y_indices, 0, 511)
    w_focus = w[y_indices, 256]

    axes[2,1].plot(y_focus, w_focus, 'red', linewidth=3, marker='o', markersize=3)
    axes[2,1].set_title('Zoom sur la zone de transition')
    axes[2,1].set_xlabel('y')
    axes[2,1].set_ylabel('ω(π, y)')
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 3. ANALYSE PAR ONDELETTES (t=0) AVEC PLOTS
    print(f"\n=== 3. ANALYSE PAR ONDELETTES CONTINUES (t=0) ===")

    signal = w[:, 256]
    scales = np.arange(1, 65)
    dt = 2*np.pi/512

    print(f"Signal: {len(signal)} points")
    print(f"Domaine: [0, 2π] ≈ [0, {2*np.pi:.3f}]")
    print(f"Résolution: Δy = {dt:.4f}")

    # CWT
    coefficients, frequencies = pywt.cwt(signal, scales, 'cmor1.5-1.0', dt)
    print(f"Matrice des coefficients: {coefficients.shape}")

    # PLOT 3: Analyse par ondelettes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Analyse par ondelettes continues - Cas {case_name} (t=0)', fontsize=16)

    # Signal original
    axes[0,0].plot(y, signal, 'b-', linewidth=2)
    axes[0,0].set_title('Signal original (coupe verticale de vorticité)')
    axes[0,0].set_xlabel('y')
    axes[0,0].set_ylabel('ω(π, y)')
    axes[0,0].grid(True, alpha=0.3)

    # Scalogramme
    im1 = axes[0,1].imshow(np.abs(coefficients), extent=[0, 2*np.pi, scales[0], scales[-1]], 
                           cmap='jet', aspect='auto', origin='lower')
    axes[0,1].set_title('Scalogramme |CWT(y, échelle)|')
    axes[0,1].set_xlabel('y')
    axes[0,1].set_ylabel('Échelle')
    plt.colorbar(im1, ax=axes[0,1])

    # Coefficients à différentes échelles
    selected_scales = [2, 8, 16, 32]
    colors_scales = ['red', 'green', 'blue', 'orange']

    for scale_idx, color in zip(selected_scales, colors_scales):
        if scale_idx < len(scales):
            coeff_at_scale = np.abs(coefficients[scale_idx-1, :])
            axes[1,0].plot(y, coeff_at_scale, color=color, linewidth=2, 
                           label=f'Échelle {scale_idx}')

    axes[1,0].set_title('Coefficients d\'ondelettes à différentes échelles')
    axes[1,0].set_xlabel('y')
    axes[1,0].set_ylabel('|CWT|')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Détection de singularités
    max_coeffs_per_position = np.max(np.abs(coefficients), axis=0)
    peaks, properties = find_peaks(max_coeffs_per_position, 
                                 height=np.max(max_coeffs_per_position)*0.2)

    axes[1,1].plot(y, max_coeffs_per_position, 'purple', linewidth=3, label='Max |CWT| par position')
    axes[1,1].set_title('Détection de singularités')
    axes[1,1].set_xlabel('y')
    axes[1,1].set_ylabel('max_échelles |CWT(y, échelle)|')
    axes[1,1].grid(True, alpha=0.3)

    print(f"\n=== SINGULARITÉS DÉTECTÉES (t=0) ===")
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

    # 4. ÉVOLUTION TEMPORELLE AVEC PLOTS
    print(f"\n=== 4. ANALYSE TEMPORELLE DE L'INSTABILITÉ ===")

    # Trouver les fichiers disponibles avec les bons patterns
    available_times = []
    for t in range(max_time + 1):
        # CORRECTION DES PATTERNS DE FICHIERS POUR B_0_2 (même logique que B_0_1)
        if case_name == 'B_0_2x':
            filename = f'MHD_data_kelvin_helmholtz_B_0_2_t{t}.mat'  # SANS x
        elif case_name == 'B_0_2y':
            filename = f'MHD_data_kelvin_helmholtz_B_0_2y_t{t}.mat'  # AVEC y
        else:
            filename = f'MHD_data_kelvin_helmholtz_{case_name}_t{t}.mat'

        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            available_times.append(t)

    print(f"Fichiers disponibles: {len(available_times)} (t=0 à t={max(available_times) if available_times else 'N/A'})")

    if not available_times:
        print("❌ Aucun fichier temporel trouvé")
        return None

    # Sélectionner des temps représentatifs
    if len(available_times) > 7:
        step = len(available_times) // 7
        selected_times = available_times[::step][:7]
    else:
        selected_times = available_times

    print(f"Temps sélectionnés pour analyse détaillée: {selected_times}")

    # Analyser l'évolution
    evolution_data = []

    def analyze_time_point(t):
        # CORRECTION DES PATTERNS DE FICHIERS POUR B_0_2 (même logique que B_0_1)
        if case_name == 'B_0_2x':
            filename = f'MHD_data_kelvin_helmholtz_B_0_2_t{t}.mat'  # SANS x
        elif case_name == 'B_0_2y':
            filename = f'MHD_data_kelvin_helmholtz_B_0_2y_t{t}.mat'  # AVEC y
        else:
            filename = f'MHD_data_kelvin_helmholtz_{case_name}_t{t}.mat'

        filepath = os.path.join(data_path, filename)

        try:
            data = scipy.io.loadmat(filepath)
            w = data['w']
            signal = w[:, 256]

            # Propriétés du signal
            signal_std = np.std(signal)
            signal_range = np.max(signal) - np.min(signal)
            gradient = np.gradient(signal)
            max_gradient = np.max(np.abs(gradient))

            # Analyse 2D
            w_variance_2d = np.var(w)
            x_profiles = [w[:, i] for i in [128, 256, 384]]
            x_profile_stds = [np.std(profile) for profile in x_profiles]
            x_asymmetry = np.std(x_profile_stds)

            return {
                'time': t,
                'signal_std': signal_std,
                'signal_range': signal_range,
                'max_gradient': max_gradient,
                'variance_2d': w_variance_2d,
                'x_asymmetry': x_asymmetry,
                'signal': signal,
                'w_2d': w
            }
        except Exception as e:
            print(f"  ❌ Erreur pour t = {t}: {e}")
            return None

    print(f"\nAnalyse des propriétés pour chaque temps sélectionné...")
    for t in selected_times:
        result = analyze_time_point(t)
        if result:
            evolution_data.append(result)
            print(f"  ✅ t = {t}: std = {result['signal_std']:.4f}, max_grad = {result['max_gradient']:.4f}")

    # PLOT 4: Évolution temporelle
    if len(evolution_data) > 1:
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Évolution temporelle de l\'instabilité - Cas {case_name}', fontsize=16)

        # Extraire les données
        times_plot = [d['time'] for d in evolution_data]
        stds = [d['signal_std'] for d in evolution_data]
        ranges = [d['signal_range'] for d in evolution_data]
        max_grads = [d['max_gradient'] for d in evolution_data]
        var_2d = [d['variance_2d'] for d in evolution_data]
        x_asym = [d['x_asymmetry'] for d in evolution_data]

        # Plots d'évolution
        axes[0,0].plot(times_plot, stds, 'bo-', linewidth=2, markersize=6)
        axes[0,0].set_title('Évolution de la variance de vorticité')
        axes[0,0].set_xlabel('Temps')
        axes[0,0].set_ylabel('σ(ω)')
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(times_plot, max_grads, 'ro-', linewidth=2, markersize=6)
        axes[0,1].set_title('Gradient maximal (singularités)')
        axes[0,1].set_xlabel('Temps')
        axes[0,1].set_ylabel('max|∇ω|')
        axes[0,1].grid(True, alpha=0.3)

        axes[0,2].plot(times_plot, x_asym, 'go-', linewidth=2, markersize=6)
        axes[0,2].set_title('Développement 2D (asymétrie)')
        axes[0,2].set_xlabel('Temps')
        axes[0,2].set_ylabel('Asymétrie en x')
        axes[0,2].grid(True, alpha=0.3)

        axes[1,0].plot(times_plot, var_2d, 'mo-', linewidth=2, markersize=6)
        axes[1,0].set_title('Variance 2D totale')
        axes[1,0].set_xlabel('Temps')
        axes[1,0].set_ylabel('Var(ω) 2D')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(times_plot, ranges, 'yo-', linewidth=2, markersize=6)
        axes[1,1].set_title('Étendue des valeurs')
        axes[1,1].set_xlabel('Temps')
        axes[1,1].set_ylabel('max(ω) - min(ω)')
        axes[1,1].grid(True, alpha=0.3)

        # Évolution des profils 1D
        colors_time = plt.cm.viridis(np.linspace(0, 1, len(evolution_data)))
        for i, (data_point, color) in enumerate(zip(evolution_data, colors_time)):
            axes[2,0].plot(y, data_point['signal'], color=color, linewidth=2, 
                          label=f't = {data_point["time"]:.0f}')

        axes[2,0].set_title('Évolution des profils de vorticité')
        axes[2,0].set_xlabel('y')
        axes[2,0].set_ylabel('ω(π, y)')
        axes[2,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2,0].grid(True, alpha=0.3)

        # Champs 2D initial et final
        im1 = axes[2,1].imshow(evolution_data[0]['w_2d'], extent=[0, 2*np.pi, 0, 2*np.pi], 
                              cmap='RdBu_r', origin='lower')
        axes[2,1].set_title(f'Vorticité à t = {evolution_data[0]["time"]:.0f}')
        axes[2,1].set_xlabel('x')
        axes[2,1].set_ylabel('y')

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

    # 5. ANALYSE COMPLÈTE PAR ONDELETTES AVEC PLOTS
    print(f"\n=== 5. ANALYSE TEMPORELLE COMPLÈTE PAR ONDELETTES ===")

    all_times = []
    all_singularities = []
    all_max_coeffs = []
    all_scalograms = []
    evolution_metrics = []

    print("Chargement et analyse de tous les fichiers...")

    for t in available_times:
        # CORRECTION DES PATTERNS DE FICHIERS POUR B_0_2 (même logique que B_0_1)
        if case_name == 'B_0_2x':
            filename = f'MHD_data_kelvin_helmholtz_B_0_2_t{t}.mat'  # SANS x
        elif case_name == 'B_0_2y':
            filename = f'MHD_data_kelvin_helmholtz_B_0_2y_t{t}.mat'  # AVEC y
        else:
            filename = f'MHD_data_kelvin_helmholtz_{case_name}_t{t}.mat'

        filepath = os.path.join(data_path, filename)

        try:
            data = scipy.io.loadmat(filepath)
            w = data['w']
            signal = w[:, 256]

            # CWT
            coefficients, frequencies = pywt.cwt(signal, scales, 'cmor1.5-1.0', dt)

            # Détection de singularités
            max_coeffs_per_position = np.max(np.abs(coefficients), axis=0)
            peaks, properties = find_peaks(max_coeffs_per_position, 
                                         height=np.max(max_coeffs_per_position)*0.2)

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

            evolution_metrics.append({
                'time': t,
                'max_coeff_global': np.max(np.abs(coefficients)),
                'n_singularities': len(peaks),
                'strongest_singularity': np.max(singularities_intensities) if len(singularities_intensities) > 0 else 0,
                'wavelet_energy': np.sum(np.abs(coefficients)**2)
            })

            if t % 10 == 0 or t in [0, 5]:
                print(f"  ✅ t = {t}: {len(peaks)} singularités, coeff_max = {np.max(np.abs(coefficients)):.4f}")

        except Exception as e:
            print(f"  ❌ Erreur pour t = {t}: {e}")

    print(f"\nAnalyse terminée pour {len(all_times)} temps.")

    # PLOT 5: Analyse complète par ondelettes
    if len(evolution_metrics) > 0:
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Analyse temporelle complète par ondelettes - Cas {case_name}', fontsize=16)

        # Évolution du nombre de singularités
        ax1 = plt.subplot(3, 4, 1)
        n_sing_evolution = [m['n_singularities'] for m in evolution_metrics]
        plt.plot(all_times, n_sing_evolution, 'bo-', linewidth=2, markersize=4)
        plt.title('Évolution du nombre de singularités')
        plt.xlabel('Temps')
        plt.ylabel('Nombre de singularités')
        plt.grid(True, alpha=0.3)

        # Évolution de l'intensité maximale
        ax2 = plt.subplot(3, 4, 2)
        max_intensities = [m['strongest_singularity'] for m in evolution_metrics]
        plt.plot(all_times, max_intensities, 'ro-', linewidth=2, markersize=4)
        plt.title('Intensité de la singularité la plus forte')
        plt.xlabel('Temps')
        plt.ylabel('Intensité maximale')
        plt.grid(True, alpha=0.3)

        # Évolution de l'énergie des ondelettes
        ax3 = plt.subplot(3, 4, 3)
        wavelet_energies = [m['wavelet_energy'] for m in evolution_metrics]
        plt.plot(all_times, wavelet_energies, 'go-', linewidth=2, markersize=4)
        plt.title('Énergie totale des ondelettes')
        plt.xlabel('Temps')
        plt.ylabel('Énergie CWT')
        plt.grid(True, alpha=0.3)

        # Coefficient maximal global
        ax4 = plt.subplot(3, 4, 4)
        max_coeffs_global = [m['max_coeff_global'] for m in evolution_metrics]
        plt.plot(all_times, max_coeffs_global, 'mo-', linewidth=2, markersize=4)
        plt.title('Coefficient CWT maximal')
        plt.xlabel('Temps')
        plt.ylabel('max|CWT|')
        plt.grid(True, alpha=0.3)

        # Diagramme spatio-temporel des singularités
        ax5 = plt.subplot(3, 4, (5, 6))
        for i, (t, sing_data) in enumerate(zip(all_times, all_singularities)):
            if sing_data['n_singularities'] > 0:
                positions = sing_data['positions']
                intensities = sing_data['intensities']
                sizes = [50 * (intensity / max(max_intensities)) for intensity in intensities if max(max_intensities) > 0]
                colors = [intensity for intensity in intensities]
                if sizes and colors:
                    scatter = plt.scatter([t] * len(positions), positions, s=sizes, c=colors, 
                                        cmap='hot', alpha=0.7, edgecolors='black', linewidth=0.5)

        plt.title('Diagramme spatio-temporel des singularités')
        plt.xlabel('Temps')
        plt.ylabel('Position y')
        if len(max_intensities) > 0 and max(max_intensities) > 0:
            plt.colorbar(scatter, label='Intensité')
        plt.grid(True, alpha=0.3)

        # Scalogrammes à différents temps
        selected_times_for_scalogram = [0, max_time//5, 2*max_time//5, 3*max_time//5, 4*max_time//5, max_time]
        selected_times_for_scalogram = [t for t in selected_times_for_scalogram if t in all_times]

        for i, t_sel in enumerate(selected_times_for_scalogram[:6]):
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

        # Évolution des profils de coefficients maximaux
        ax7 = plt.subplot(3, 4, (7, 8))
        colors_prof = plt.cm.viridis(np.linspace(0, 1, len(selected_times_for_scalogram)))
        for i, t_sel in enumerate(selected_times_for_scalogram):
            if t_sel in all_times:
                idx = all_times.index(t_sel)
                plt.plot(y, all_max_coeffs[idx], color=colors_prof[i], linewidth=2, 
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
        max_time_available = max(all_times) if all_times else max_time
        early_phase = [m for m in evolution_metrics if m['time'] <= max_time_available//3]
        middle_phase = [m for m in evolution_metrics if max_time_available//3 < m['time'] <= 2*max_time_available//3]
        late_phase = [m for m in evolution_metrics if m['time'] > 2*max_time_available//3]

        print(f"\n📊 PHASES DE DÉVELOPPEMENT:")
        if early_phase:
            print(f"Phase précoce (t=0-{max_time_available//3}):")
            print(f"  - Singularités moyennes: {np.mean([m['n_singularities'] for m in early_phase]):.1f}")
            print(f"  - Intensité moyenne: {np.mean([m['strongest_singularity'] for m in early_phase]):.4f}")

        if middle_phase:
            print(f"Phase intermédiaire (t={max_time_available//3}-{2*max_time_available//3}):")
            print(f"  - Singularités moyennes: {np.mean([m['n_singularities'] for m in middle_phase]):.1f}")
            print(f"  - Intensité moyenne: {np.mean([m['strongest_singularity'] for m in middle_phase]):.4f}")

        if late_phase:
            print(f"Phase tardive (t>{2*max_time_available//3}):")
            print(f"  - Singularités moyennes: {np.mean([m['n_singularities'] for m in late_phase]):.1f}")
            print(f"  - Intensité moyenne: {np.mean([m['strongest_singularity'] for m in late_phase]):.4f}")

        # Détection des transitions
        print(f"\n🔍 DÉTECTION DES TRANSITIONS:")
        for i in range(1, len(evolution_metrics)):
            curr = evolution_metrics[i]
            prev = evolution_metrics[i-1]

            if curr['n_singularities'] > prev['n_singularities'] * 1.5:
                print(f"  📈 Augmentation singularités à t = {curr['time']}: {prev['n_singularities']} → {curr['n_singularities']}")

        # Statistiques finales
        print(f"\n📈 STATISTIQUES GLOBALES:")
        print(f"Temps total analysé: {len(all_times)} points")
        if n_sing_evolution:
            print(f"Singularités max simultanées: {max(n_sing_evolution)}")
        if max_intensities:
            print(f"Intensité maximale atteinte: {max(max_intensities):.4f}")
        if len(wavelet_energies) > 1:
            print(f"Croissance énergie ondelettes: ×{wavelet_energies[-1]/wavelet_energies[0]:.1f}")

    return {
        'case_name': case_name,
        'evolution_data': evolution_data,
        'evolution_metrics': evolution_metrics,
        'all_times': all_times,
        'all_singularities': all_singularities
    }

# ANALYSE DES DEUX SOUS-CAS AVEC TOUTES LES VISUALISATIONS
print("🚀 DÉBUT DE L'ANALYSE COMPARATIVE COMPLÈTE DES CAS B_0_2x ET B_0_2y")

# Cas B_0_2x
case_2x_path = r'C:\Users\user\Desktop\Donnees_KVI\Kelvin_Helmholtz_Instabilities_B_0_2\MHD_data_kelvin_helmholtz_B_0_2x'
results_2x = analyze_kh_case_with_plots('B_0_2x', case_2x_path, max_time=50)

# Cas B_0_2y  
case_2y_path = r'C:\Users\user\Desktop\Donnees_KVI\Kelvin_Helmholtz_Instabilities_B_0_2\MHD_data_kelvin_helmholtz_B_0_2y'
results_2y = analyze_kh_case_with_plots('B_0_2y', case_2y_path, max_time=50)

# COMPARAISON FINALE AVEC PLOTS
print(f"\n{'='*80}")
print(f"=== COMPARAISON FINALE B_0_2x vs B_0_2y ===")
print(f"{'='*80}")

if results_2x and results_2y:
    # PLOT COMPARATIF FINAL
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparaison finale B_0_2x vs B_0_2y - Champ magnétique B₀ = 0.2', fontsize=16)

    # Comparer les évolutions de singularités
    if results_2x['evolution_metrics'] and results_2y['evolution_metrics']:
        times_2x = [m['time'] for m in results_2x['evolution_metrics']]
        n_sing_2x = [m['n_singularities'] for m in results_2x['evolution_metrics']]
        intensities_2x = [m['strongest_singularity'] for m in results_2x['evolution_metrics']]

        times_2y = [m['time'] for m in results_2y['evolution_metrics']]
        n_sing_2y = [m['n_singularities'] for m in results_2y['evolution_metrics']]
        intensities_2y = [m['strongest_singularity'] for m in results_2y['evolution_metrics']]

        # Nombre de singularités
        axes[0,0].plot(times_2x, n_sing_2x, 'b-o', linewidth=2, label='B_0_2x', markersize=4)
        axes[0,0].plot(times_2y, n_sing_2y, 'r-o', linewidth=2, label='B_0_2y', markersize=4)
        axes[0,0].set_title('Évolution du nombre de singularités')
        axes[0,0].set_xlabel('Temps')
        axes[0,0].set_ylabel('Nombre de singularités')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Intensités maximales
        axes[0,1].plot(times_2x, intensities_2x, 'b-o', linewidth=2, label='B_0_2x', markersize=4)
        axes[0,1].plot(times_2y, intensities_2y, 'r-o', linewidth=2, label='B_0_2y', markersize=4)
        axes[0,1].set_title('Évolution des intensités maximales')
        axes[0,1].set_xlabel('Temps')
        axes[0,1].set_ylabel('Intensité maximale')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Ratios
        min_len = min(len(n_sing_2x), len(n_sing_2y))
        if min_len > 0:
            ratio_sing = [n_sing_2y[i] / max(n_sing_2x[i], 1) for i in range(min_len)]
            ratio_int = [intensities_2y[i] / max(intensities_2x[i], 1e-10) for i in range(min_len)]
            times_ratio = times_2y[:min_len]

            axes[0,2].plot(times_ratio, ratio_sing, 'g-o', linewidth=2, label='Ratio singularités (2y/2x)', markersize=4)
            axes[0,2].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[0,2].set_title('Ratio B_0_2y / B_0_2x')
            axes[0,2].set_xlabel('Temps')
            axes[0,2].set_ylabel('Ratio')
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)

            axes[1,0].plot(times_ratio, ratio_int, 'm-o', linewidth=2, label='Ratio intensités (2y/2x)', markersize=4)
            axes[1,0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[1,0].set_title('Ratio intensités B_0_2y / B_0_2x')
            axes[1,0].set_xlabel('Temps')
            axes[1,0].set_ylabel('Ratio')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)

        # Statistiques finales
        if results_2x['evolution_metrics'] and results_2y['evolution_metrics']:
            final_2x = results_2x['evolution_metrics'][-1]
            final_2y = results_2y['evolution_metrics'][-1]

            print(f"\n🔍 COMPARAISON DES ÉVOLUTIONS:")
            print(f"\nMétriques finales:")
            print(f"B_0_2x - Singularités: {final_2x['n_singularities']}, Intensité max: {final_2x['strongest_singularity']:.4f}")
            print(f"B_0_2y - Singularités: {final_2y['n_singularities']}, Intensité max: {final_2y['strongest_singularity']:.4f}")

            print(f"\nComparaison:")
            print(f"Ratio singularités (2y/2x): {final_2y['n_singularities']/max(final_2x['n_singularities'], 1):.2f}")
            print(f"Ratio intensités (2y/2x): {final_2y['strongest_singularity']/max(final_2x['strongest_singularity'], 1e-10):.2f}")

            # Barres de comparaison
            categories = ['Singularités finales', 'Intensité max finale']
            values_2x = [final_2x['n_singularities'], final_2x['strongest_singularity']]
            values_2y = [final_2y['n_singularities'], final_2y['strongest_singularity']]

            x = np.arange(len(categories))
            width = 0.35

            axes[1,1].bar(x - width/2, values_2x, width, label='B_0_2x', alpha=0.8, color='blue')
            axes[1,1].bar(x + width/2, values_2y, width, label='B_0_2y', alpha=0.8, color='red')
            axes[1,1].set_title('Comparaison des métriques finales')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(categories)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            # Différences relatives
            diff_sing = abs(final_2y['n_singularities'] - final_2x['n_singularities'])
            diff_int = abs(final_2y['strongest_singularity'] - final_2x['strongest_singularity'])

            axes[1,2].bar(['Diff. Singularités', 'Diff. Intensités'], [diff_sing, diff_int], 
                         color=['orange', 'purple'], alpha=0.8)
            axes[1,2].set_title('Différences absolues')
            axes[1,2].grid(True, alpha=0.3)

            print(f"\n🎯 CONCLUSION:")
            if final_2y['n_singularities'] > final_2x['n_singularities'] * 1.5:
                print(f"  📈 B_0_2y génère BEAUCOUP PLUS de singularités que B_0_2x")
            elif final_2y['n_singularities'] < final_2x['n_singularities'] * 0.7:
                print(f"  📉 B_0_2x génère plus de singularités que B_0_2y")
            else:
                print(f"  ⚖️ Nombre de singularités comparable entre 2x et 2y")

            if final_2y['strongest_singularity'] > final_2x['strongest_singularity'] * 1.5:
                print(f"  🔥 B_0_2y a des singularités BEAUCOUP PLUS intenses que B_0_2x")
            elif final_2y['strongest_singularity'] < final_2x['strongest_singularity'] * 0.7:
                print(f"  🔥 B_0_2x a des singularités plus intenses que B_0_2y")
            else:
                print(f"  ⚖️ Intensités comparables entre 2x et 2y")

    plt.tight_layout()
    plt.show()

print(f"\n✅ ANALYSE COMPLÈTE AVEC VISUALISATIONS TERMINÉE POUR LES CAS B_0_2x ET B_0_2y")
