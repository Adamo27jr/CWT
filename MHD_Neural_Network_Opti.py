
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
import pywt

# Vérifier les ondelettes disponibles
print("🌊 ONDELETTES DISPONIBLES:")
print("Continues:", pywt.wavelist(kind='continuous'))
print("Discrètes:", pywt.wavelist(kind='discrete')[:10], "...")

class MHD_WaveletNet(nn.Module):
    """
    Réseau de neurones spécialisé pour MHD avec ondelettes
    Architecture: Wavelet Features → Dense → Prédictions multiples
    """
    def __init__(self, wavelet_features=100, hidden_dims=[256, 128, 64, 32]):
        super().__init__()

        # Couches principales
        layers = []
        input_dim = wavelet_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Têtes de prédiction multiples
        self.singularity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # 4 régimes: Chaos_HD, Stable, Transition, Resonance
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        singularities = self.singularity_head(features)
        intensity = self.intensity_head(features)
        energy = self.energy_head(features)
        regime_logits = self.regime_head(features)

        return {
            'singularities': singularities,
            'intensity': intensity,
            'energy': energy,
            'regime': regime_logits
        }

class WaveletFeatureExtractor:
    """
    Extracteur de features ondelettes pour données MHD
    """
    def __init__(self, wavelet='morl', scales=np.arange(1, 21)):  # CORRECTION: 'morl' au lieu de 'morlet'
        self.wavelet = wavelet
        self.scales = scales
        self.scaler = StandardScaler()

        # Vérifier que l'ondelette existe
        if wavelet not in pywt.wavelist(kind='continuous'):
            print(f"⚠️  Ondelette '{wavelet}' non trouvée, utilisation de 'mexh' (Mexican Hat)")
            self.wavelet = 'mexh'

    def extract_features(self, data_2d):
        """
        Extrait features ondelettes d'un champ 2D
        """
        features = []

        try:
            # 1. Transformée ondelettes continue sur différentes directions
            # Horizontal cuts (réduire le nombre pour éviter trop de features)
            for i in range(0, data_2d.shape[0], max(1, data_2d.shape[0]//3)):  # 3 coupes au lieu de 5
                line = data_2d[i, :]
                if len(line) > 10:  # Vérifier longueur minimale
                    coeffs, _ = pywt.cwt(line, self.scales, self.wavelet)
                    features.extend([
                        np.mean(np.abs(coeffs)),
                        np.std(np.abs(coeffs)),
                        np.max(np.abs(coeffs)),
                        len(signal.find_peaks(np.abs(coeffs.flatten()), height=np.std(np.abs(coeffs)))[0])
                    ])

            # Vertical cuts
            for j in range(0, data_2d.shape[1], max(1, data_2d.shape[1]//3)):  # 3 coupes au lieu de 5
                line = data_2d[:, j]
                if len(line) > 10:  # Vérifier longueur minimale
                    coeffs, _ = pywt.cwt(line, self.scales, self.wavelet)
                    features.extend([
                        np.mean(np.abs(coeffs)),
                        np.std(np.abs(coeffs)),
                        np.max(np.abs(coeffs)),
                        len(signal.find_peaks(np.abs(coeffs.flatten()), height=np.std(np.abs(coeffs)))[0])
                    ])

        except Exception as e:
            print(f"⚠️  Erreur ondelettes: {e}")
            print("🔄 Utilisation de features alternatives...")

            # Features alternatives si ondelettes échouent
            features = []

        # 2. Features globales (toujours calculées)
        # Gradient magnitude
        grad_x, grad_y = np.gradient(data_2d)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.max(grad_mag),
            np.sum(grad_mag > 2*np.std(grad_mag))  # Points de fort gradient
        ])

        # 3. Features statistiques du champ
        features.extend([
            np.mean(data_2d),
            np.std(data_2d),
            np.min(data_2d),
            np.max(data_2d),
            np.sum(np.abs(data_2d) > 2*np.std(data_2d)),  # Outliers
            np.sum(data_2d > 0) / data_2d.size,  # Fraction positive
        ])

        # 4. Features FFT (alternative aux ondelettes)
        fft_2d = np.fft.fft2(data_2d)
        fft_mag = np.abs(fft_2d)
        features.extend([
            np.mean(fft_mag),
            np.std(fft_mag),
            np.max(fft_mag),
            np.sum(fft_mag > np.mean(fft_mag) + 2*np.std(fft_mag))
        ])

        # S'assurer qu'on a assez de features
        while len(features) < 50:  # Minimum 50 features
            features.extend([0.0] * (50 - len(features)))

        return np.array(features[:100])  # Limiter à 100 features max

    def fit_scaler(self, features_list):
        """Fit le scaler sur une liste de features"""
        all_features = np.vstack(features_list)
        self.scaler.fit(all_features)

    def transform(self, features):
        """Normalise les features"""
        return self.scaler.transform(features.reshape(1, -1)).flatten()

def create_synthetic_mhd_data(n_samples=500):  # Réduire le nombre d'échantillons
    """
    Crée des données MHD synthétiques pour entraînement
    Basé sur tes découvertes réelles
    """
    np.random.seed(42)

    data = []

    print(f"📊 Génération de {n_samples} échantillons...")

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"   Échantillon {i}/{n_samples}")

        # Paramètres aléatoires
        B0 = np.random.uniform(0.05, 0.25)
        nx, ny = 32, 32  # Réduire la taille pour plus de rapidité

        # Génération du champ de vorticité basé sur Kelvin-Helmholtz
        x = np.linspace(0, 2*np.pi, nx)
        y = np.linspace(0, 2*np.pi, ny)
        X, Y = np.meshgrid(x, y)

        # Instabilité KH avec influence magnétique
        vorticity = np.tanh((Y - np.pi)/0.1) + 0.1*np.sin(4*X)*np.exp(-(Y-np.pi)**2/0.5)

        # Effet du champ magnétique (basé sur tes découvertes)
        if B0 < 0.08:
            # Chaos HD
            noise_level = 0.5
            n_sing_base = 50
        elif B0 < 0.13:
            # Stable
            noise_level = 0.1
            n_sing_base = 15
        elif B0 < 0.17:
            # Transition (paradoxe B0=0.15)
            noise_level = 0.05
            n_sing_base = 12
            vorticity *= 2  # Intensité plus forte
        else:
            # Résonance
            noise_level = 0.3
            n_sing_base = 60

        # Ajouter du bruit
        vorticity += noise_level * np.random.randn(nx, ny)

        # Calculer les targets basés sur tes résultats
        n_singularities = n_sing_base + np.random.randint(-5, 6)
        max_intensity = np.max(np.abs(vorticity))
        energy_growth = max_intensity * 50 * (1 + np.random.normal(0, 0.2))

        # Régime
        if B0 < 0.08:
            regime = 0  # Chaos_HD
        elif B0 < 0.13:
            regime = 1  # Stable
        elif B0 < 0.17:
            regime = 2  # Transition
        else:
            regime = 3  # Resonance

        data.append({
            'B0': B0,
            'vorticity': vorticity,
            'n_singularities': n_singularities,
            'max_intensity': max_intensity,
            'energy_growth': energy_growth,
            'regime': regime
        })

    return data

def train_mhd_network():
    """
    Entraîne le réseau de neurones MHD
    """
    print("🚀 CRÉATION ET ENTRAÎNEMENT DU RÉSEAU MHD + ONDELETTES")
    print("="*60)

    # 1. Générer données synthétiques
    print("📊 Génération des données synthétiques...")
    synthetic_data = create_synthetic_mhd_data(500)  # Réduire pour test

    # 2. Extraire features ondelettes
    print("🌊 Extraction des features ondelettes...")
    feature_extractor = WaveletFeatureExtractor()

    features_list = []
    targets = {
        'singularities': [],
        'intensity': [],
        'energy': [],
        'regime': []
    }

    for i, sample in enumerate(synthetic_data):
        if i % 100 == 0:
            print(f"   Features {i}/{len(synthetic_data)}")

        try:
            # Extraire features
            features = feature_extractor.extract_features(sample['vorticity'])
            features_list.append(features)

            # Targets
            targets['singularities'].append(sample['n_singularities'])
            targets['intensity'].append(sample['max_intensity'])
            targets['energy'].append(sample['energy_growth'])
            targets['regime'].append(sample['regime'])

        except Exception as e:
            print(f"⚠️  Erreur échantillon {i}: {e}")
            continue

    print(f"✅ {len(features_list)} échantillons traités avec succès")

    # 3. Normalisation
    feature_extractor.fit_scaler(features_list)
    X = np.array([feature_extractor.transform(f) for f in features_list])

    print(f"📊 Shape des features: {X.shape}")

    # 4. Préparer targets
    y_sing = np.array(targets['singularities']).reshape(-1, 1)
    y_intensity = np.array(targets['intensity']).reshape(-1, 1)
    y_energy = np.array(targets['energy']).reshape(-1, 1)
    y_regime = np.array(targets['regime'])

    # 5. Split train/test
    X_train, X_test, y_sing_train, y_sing_test = train_test_split(
        X, y_sing, test_size=0.2, random_state=42
    )
    _, _, y_int_train, y_int_test = train_test_split(
        X, y_intensity, test_size=0.2, random_state=42
    )
    _, _, y_eng_train, y_eng_test = train_test_split(
        X, y_energy, test_size=0.2, random_state=42
    )
    _, _, y_reg_train, y_reg_test = train_test_split(
        X, y_regime, test_size=0.2, random_state=42
    )

    print(f"📊 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 6. Créer le modèle
    model = MHD_WaveletNet(wavelet_features=X.shape[1], hidden_dims=[256, 128, 64, 32])

    # 7. Loss functions et optimizer
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 8. Entraînement
    print("🧠 Entraînement du réseau...")
    model.train()

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_sing_train_t = torch.FloatTensor(y_sing_train)
    y_int_train_t = torch.FloatTensor(y_int_train)
    y_eng_train_t = torch.FloatTensor(y_eng_train)
    y_reg_train_t = torch.LongTensor(y_reg_train)

    losses = []

    for epoch in range(300):  # Réduire le nombre d'epochs pour test
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_t)

        # Calcul des losses
        loss_sing = mse_loss(outputs['singularities'], y_sing_train_t)
        loss_int = mse_loss(outputs['intensity'], y_int_train_t)
        loss_eng = mse_loss(outputs['energy'], y_eng_train_t)
        loss_reg = ce_loss(outputs['regime'], y_reg_train_t)

        # Loss totale
        total_loss = loss_sing + loss_int + 0.1*loss_eng + 0.5*loss_reg

        # Backward pass
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

    # 9. Évaluation
    print("\n📊 ÉVALUATION DU MODÈLE")
    print("="*30)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        test_outputs = model(X_test_t)

        # Métriques
        sing_pred = test_outputs['singularities'].numpy()
        sing_true = y_sing_test

        mse_sing = np.mean((sing_pred - sing_true)**2)
        mae_sing = np.mean(np.abs(sing_pred - sing_true))

        print(f"Singularités - MSE: {mse_sing:.2f}, MAE: {mae_sing:.2f}")

        # Régimes
        regime_pred = torch.argmax(test_outputs['regime'], dim=1).numpy()
        regime_acc = np.mean(regime_pred == y_reg_test)
        print(f"Classification régimes - Accuracy: {regime_acc:.3f}")

    # 10. Visualisation
    plt.figure(figsize=(15, 5))

    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Courbe de Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Prédictions vs réalité (singularités)
    plt.subplot(1, 3, 2)
    plt.scatter(sing_true, sing_pred, alpha=0.6)
    plt.plot([sing_true.min(), sing_true.max()], [sing_true.min(), sing_true.max()], 'r--')
    plt.xlabel('Singularités réelles')
    plt.ylabel('Singularités prédites')
    plt.title('Prédiction Singularités')
    plt.grid(True)

    # Matrice de confusion régimes
    plt.subplot(1, 3, 3)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_reg_test, regime_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice Confusion Régimes')
    plt.colorbar()

    regime_names = ['Chaos_HD', 'Stable', 'Transition', 'Resonance']
    tick_marks = np.arange(len(regime_names))
    plt.xticks(tick_marks, regime_names, rotation=45)
    plt.yticks(tick_marks, regime_names)

    plt.tight_layout()
    plt.show()

    print("\n✅ ENTRAÎNEMENT TERMINÉ!")
    print("🎯 Le réseau peut maintenant prédire:")
    print("   • Nombre de singularités")
    print("   • Intensité maximale")
    print("   • Croissance énergétique")
    print("   • Classification du régime MHD")

    return model, feature_extractor

# Exemple d'utilisation
if __name__ == "__main__":
    model, extractor = train_mhd_network()
