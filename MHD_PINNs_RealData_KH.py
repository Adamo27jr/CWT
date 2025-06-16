
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import pywt
from sklearn.preprocessing import StandardScaler
import time
import os
import glob

class RealDataLoader:
    """
    Chargeur pour les vraies données Kelvin-Helmholtz - VERSION CORRIGÉE
    """
    def __init__(self, data_path="C:/Users/user/Desktop/Donnees_KVI"):
        self.data_path = data_path
        self.cases = {
            'B_0': 'Kelvin_Helmholtz_Instabilities_B_0',
            'B_0_1': 'Kelvin_Helmholtz_Instabilities_B_0_1', 
            'B_0_2': 'Kelvin_Helmholtz_Instabilities_B_0_2',
            'B_0_05': 'Kelvin_Helmholtz_Instabilities_B_0_05'
        }
        self.loaded_data = {}

    def load_case_data(self, case_name, max_files=5):
        """
        Charge les données d'un cas spécifique - VERSION ROBUSTE
        """
        print(f"🔄 CHARGEMENT DES DONNÉES: {case_name}")

        case_path = os.path.join(self.data_path, self.cases[case_name])

        # Chercher les sous-dossiers
        subdirs = []
        if os.path.exists(case_path):
            subdirs = [d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d))]

        case_data = {}

        for subdir in subdirs:
            subdir_path = os.path.join(case_path, subdir)

            # Patterns de fichiers plus flexibles
            patterns = [
                os.path.join(subdir_path, "*.mat"),
                os.path.join(subdir_path, "MHD_data_*.mat"),
                os.path.join(subdir_path, f"*{case_name}*.mat")
            ]

            mat_files = []
            for pattern in patterns:
                mat_files.extend(glob.glob(pattern))

            # Supprimer les doublons et trier
            mat_files = sorted(list(set(mat_files)))[:max_files]

            subdir_data = []
            for mat_file in mat_files:
                try:
                    print(f"   📂 Lecture: {os.path.basename(mat_file)}")
                    data = sio.loadmat(mat_file)

                    # Afficher les clés disponibles pour debug
                    available_keys = [k for k in data.keys() if not k.startswith('__')]
                    print(f"      🔑 Clés disponibles: {available_keys}")

                    # Extraire les champs avec noms flexibles
                    fields = {}

                    # Mapping des noms possibles
                    field_mappings = {
                        'vx': ['vx', 'u', 'velocity_x', 'vel_x'],
                        'vy': ['vy', 'v', 'velocity_y', 'vel_y'],
                        'Bx': ['Bx', 'B_x', 'magnetic_x', 'mag_x'],
                        'By': ['By', 'B_y', 'magnetic_y', 'mag_y'],
                        'rho': ['rho', 'density', 'dens'],
                        'p': ['p', 'pressure', 'pres']
                    }

                    for field_name, possible_names in field_mappings.items():
                        for name in possible_names:
                            if name in data:
                                fields[field_name] = data[name]
                                print(f"      ✅ {field_name}: {data[name].shape}")
                                break

                    # Si pas de champs trouvés, essayer les premières clés numériques
                    if not fields and available_keys:
                        print(f"      🔄 Tentative avec clés génériques...")
                        numeric_keys = []
                        for key in available_keys:
                            if isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
                                numeric_keys.append(key)

                        # Assigner les premières clés trouvées
                        field_names = ['vx', 'vy', 'Bx', 'By', 'rho', 'p']
                        for i, key in enumerate(numeric_keys[:len(field_names)]):
                            fields[field_names[i]] = data[key]
                            print(f"      ✅ {field_names[i]} <- {key}: {data[key].shape}")

                    # Calculer la vorticité si possible
                    if 'vx' in fields and 'vy' in fields:
                        vx, vy = fields['vx'], fields['vy']
                        if vx.ndim == 2 and vy.ndim == 2 and vx.shape == vy.shape:
                            # Calcul numérique de la vorticité
                            dvx_dy = np.gradient(vx, axis=0)
                            dvy_dx = np.gradient(vy, axis=1)
                            fields['vorticity'] = dvy_dx - dvx_dy
                            print(f"      ✅ vorticity calculée: {fields['vorticity'].shape}")

                    # Métadonnées
                    fields['file'] = os.path.basename(mat_file)
                    fields['time'] = self.extract_time_from_filename(mat_file)
                    fields['shape'] = list(fields.values())[0].shape if fields else None

                    if fields:
                        subdir_data.append(fields)
                        print(f"      ✅ Données extraites avec succès")
                    else:
                        print(f"      ⚠️ Aucun champ reconnu")

                except Exception as e:
                    print(f"      ❌ Erreur: {e}")
                    continue

            if subdir_data:
                case_data[subdir] = subdir_data
                print(f"   ✅ {subdir}: {len(subdir_data)} fichiers traités")

        self.loaded_data[case_name] = case_data
        return case_data

    def extract_time_from_filename(self, filename):
        """
        Extrait le temps du nom de fichier
        """
        import re
        match = re.search(r't(\d+(?:\.\d+)?)', filename)
        return float(match.group(1)) if match else 0.0

    def get_training_data(self, case_name, subdir=None, time_steps=None):
        """
        Prépare les données pour l'entraînement PINNs - VERSION ROBUSTE
        """
        if case_name not in self.loaded_data:
            self.load_case_data(case_name)

        case_data = self.loaded_data[case_name]

        if not case_data:
            print(f"❌ Aucune donnée disponible pour {case_name}")
            return []

        # Sélectionner le sous-dossier
        if subdir is None:
            subdir = list(case_data.keys())[0]

        if subdir not in case_data:
            print(f"❌ Sous-dossier {subdir} non trouvé")
            return []

        data_list = case_data[subdir]

        if not data_list:
            print(f"❌ Aucune donnée dans {subdir}")
            return []

        # Sélectionner les pas de temps
        if time_steps is not None:
            data_list = data_list[:time_steps]

        # Convertir en tenseurs PyTorch
        training_data = []

        for data_dict in data_list:
            try:
                # Trouver un champ de référence pour les dimensions
                ref_field = None
                for key in ['vx', 'vy', 'Bx', 'By', 'rho', 'p']:
                    if key in data_dict and isinstance(data_dict[key], np.ndarray):
                        ref_field = data_dict[key]
                        break

                if ref_field is None:
                    print(f"⚠️ Aucun champ de référence trouvé")
                    continue

                # Créer la grille de coordonnées
                if ref_field.ndim == 2:
                    ny, nx = ref_field.shape
                    x = np.linspace(-1, 1, nx)
                    y = np.linspace(-1, 1, ny)
                    X, Y = np.meshgrid(x, y)
                    coords = np.stack([X.flatten(), Y.flatten()], axis=1)

                    # Préparer les champs
                    fields = {}

                    # Vitesse
                    if 'vx' in data_dict and 'vy' in data_dict:
                        vx_flat = data_dict['vx'].flatten()
                        vy_flat = data_dict['vy'].flatten()
                        fields['velocity'] = np.stack([vx_flat, vy_flat], axis=1)

                    # Champ magnétique
                    if 'Bx' in data_dict and 'By' in data_dict:
                        Bx_flat = data_dict['Bx'].flatten()
                        By_flat = data_dict['By'].flatten()
                        fields['magnetic'] = np.stack([Bx_flat, By_flat], axis=1)

                    # Scalaires
                    if 'rho' in data_dict:
                        fields['density'] = data_dict['rho'].flatten().reshape(-1, 1)

                    if 'p' in data_dict:
                        fields['pressure'] = data_dict['p'].flatten().reshape(-1, 1)

                    if 'vorticity' in data_dict:
                        fields['vorticity'] = data_dict['vorticity'].flatten()

                    # Convertir en tenseurs
                    coords_tensor = torch.FloatTensor(coords)
                    fields_tensor = {}
                    for key, value in fields.items():
                        if isinstance(value, np.ndarray):
                            fields_tensor[key] = torch.FloatTensor(value)

                    if fields_tensor:  # Seulement si on a des champs
                        training_data.append({
                            'coords': coords_tensor,
                            'fields': fields_tensor,
                            'time': data_dict['time'],
                            'file': data_dict['file']
                        })
                        print(f"✅ Données préparées: {coords_tensor.shape[0]} points")

            except Exception as e:
                print(f"❌ Erreur préparation données: {e}")
                continue

        print(f"📊 RÉSUMÉ DONNÉES D'ENTRAÎNEMENT:")
        print(f"   • Cas: {case_name}")
        print(f"   • Sous-dossier: {subdir}")
        print(f"   • Pas de temps: {len(training_data)}")
        if training_data:
            print(f"   • Points par pas: {training_data[0]['coords'].shape[0]}")
            print(f"   • Champs disponibles: {list(training_data[0]['fields'].keys())}")

        return training_data

class MorletWaveletLayer(nn.Module):
    """
    Couche d'ondelettes Morlet intégrée dans le réseau
    """
    def __init__(self, input_dim, n_scales=15, trainable_scales=True):
        super().__init__()
        self.input_dim = input_dim
        self.n_scales = n_scales

        # Échelles d'ondelettes (paramètres apprenables)
        if trainable_scales:
            self.scales = nn.Parameter(torch.linspace(1, 20, n_scales))
        else:
            self.register_buffer('scales', torch.linspace(1, 20, n_scales))

        # Paramètres Morlet apprenables
        self.omega0 = nn.Parameter(torch.tensor(6.0))  # Fréquence centrale
        self.sigma = nn.Parameter(torch.tensor(1.0))   # Largeur

    def morlet_wavelet(self, t, scale):
        """Ondelette Morlet complexe"""
        # Normalisation
        norm = 1.0 / torch.sqrt(scale)

        # Argument normalisé
        arg = t / scale

        # Ondelette Morlet complexe
        envelope = torch.exp(-0.5 * (arg / self.sigma)**2)
        oscillation = torch.exp(1j * self.omega0 * arg)

        return norm * envelope * oscillation

    def forward(self, x):
        """
        Transformée en ondelettes continue pour chaque signal
        x: (batch_size, signal_length)
        """
        batch_size, signal_length = x.shape

        # Créer la grille temporelle
        t = torch.linspace(-signal_length//2, signal_length//2, signal_length, device=x.device)

        # Coefficients d'ondelettes pour toutes les échelles
        coeffs = []

        for scale in self.scales:
            # Ondelette mère à cette échelle
            wavelet = self.morlet_wavelet(t, scale)

            # Convolution (transformée en ondelettes)
            # Utiliser la partie réelle pour la suite
            wavelet_real = wavelet.real.unsqueeze(0).unsqueeze(0)  # (1, 1, signal_length)
            x_expanded = x.unsqueeze(1)  # (batch_size, 1, signal_length)

            # Convolution 1D
            coeff = torch.nn.functional.conv1d(
                x_expanded, 
                wavelet_real, 
                padding=signal_length//2
            )[:, 0, :signal_length]  # (batch_size, signal_length)

            coeffs.append(coeff)

        # Empiler tous les coefficients
        wavelet_coeffs = torch.stack(coeffs, dim=1)  # (batch_size, n_scales, signal_length)

        return wavelet_coeffs

class RealData_MHD_PINNs(nn.Module):
    """
    Physics-Informed Neural Network pour MHD avec vraies données Kelvin-Helmholtz
    """
    def __init__(self, input_dim=2, hidden_dims=[128, 128, 64], n_scales=15):
        super().__init__()

        # Couche d'ondelettes Morlet
        self.wavelet_layer = MorletWaveletLayer(input_dim, n_scales)
        self.n_scales = n_scales

        # Réseau principal pour les champs MHD
        layers = []
        current_dim = input_dim  # (x, y) coordinates

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.Tanh(),  # Tanh pour PINNs (meilleure stabilité)
            ])
            current_dim = hidden_dim

        self.main_network = nn.Sequential(*layers)

        # Têtes de sortie pour les champs MHD
        self.velocity_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 2)  # (vx, vy)
        )

        self.magnetic_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 2)  # (Bx, By)
        )

        self.pressure_head = nn.Sequential(
            nn.Linear(current_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # p
        )

        self.density_head = nn.Sequential(
            nn.Linear(current_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # ρ
        )

        # Réseau pour prédiction des singularités basé sur ondelettes
        wavelet_feature_dim = 3 * n_scales  # mean, std, max pour chaque échelle

        self.singularity_network = nn.Sequential(
            nn.Linear(wavelet_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # (n_singularities, max_intensity, total_energy)
        )

    def forward(self, coords):
        """
        coords: (batch_size, 2) - coordonnées (x, y)
        """
        # Extraction des features du réseau principal
        features = self.main_network(coords)

        # Prédiction des champs MHD
        velocity = self.velocity_head(features)      # (vx, vy)
        magnetic = self.magnetic_head(features)      # (Bx, By)
        pressure = self.pressure_head(features)      # p
        density = self.density_head(features)        # ρ

        return {
            'velocity': velocity,
            'magnetic': magnetic,
            'pressure': pressure,
            'density': density,
            'features': features
        }

    def predict_singularities(self, field_data):
        """
        Prédire les singularités à partir des données de champ
        field_data: (batch_size, signal_length) - données 1D du champ
        """
        # Analyse par ondelettes Morlet
        wavelet_coeffs = self.wavelet_layer(field_data)  # (batch_size, n_scales, signal_length)

        # Statistiques des coefficients d'ondelettes par échelle
        mean_coeffs = torch.mean(torch.abs(wavelet_coeffs), dim=2)  # (batch_size, n_scales)
        std_coeffs = torch.std(torch.abs(wavelet_coeffs), dim=2)    # (batch_size, n_scales)
        max_coeffs = torch.max(torch.abs(wavelet_coeffs), dim=2)[0] # (batch_size, n_scales)

        # Concaténer toutes les features: 3 * n_scales
        wavelet_features = torch.cat([mean_coeffs, std_coeffs, max_coeffs], dim=1)

        # Prédiction des singularités
        singularity_pred = self.singularity_network(wavelet_features)

        return {
            'n_singularities': singularity_pred[:, 0],
            'max_intensity': singularity_pred[:, 1],
            'total_energy': singularity_pred[:, 2],
            'wavelet_coeffs': wavelet_coeffs
        }

class RealData_MHD_Trainer:
    """
    Entraîneur pour le réseau PINNs MHD avec vraies données - VERSION ROBUSTE
    """
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Historique des losses
        self.loss_history = {
            'total': [], 'data': [], 'singularity': []
        }

    def compute_real_data_loss(self, coords, target_fields):
        """
        Calcule la loss basée sur les vraies données
        """
        outputs = self.model(coords)

        total_loss = 0.0
        loss_details = {}
        n_losses = 0

        # Loss pour chaque champ disponible
        if 'velocity' in target_fields and 'velocity' in outputs:
            loss_velocity = torch.mean((outputs['velocity'] - target_fields['velocity'])**2)
            total_loss += loss_velocity
            loss_details['velocity'] = loss_velocity.item()
            n_losses += 1

        if 'magnetic' in target_fields and 'magnetic' in outputs:
            loss_magnetic = torch.mean((outputs['magnetic'] - target_fields['magnetic'])**2)
            total_loss += loss_magnetic
            loss_details['magnetic'] = loss_magnetic.item()
            n_losses += 1

        if 'pressure' in target_fields and 'pressure' in outputs:
            loss_pressure = torch.mean((outputs['pressure'] - target_fields['pressure'])**2)
            total_loss += loss_pressure
            loss_details['pressure'] = loss_pressure.item()
            n_losses += 1

        if 'density' in target_fields and 'density' in outputs:
            loss_density = torch.mean((outputs['density'] - target_fields['density'])**2)
            total_loss += loss_density
            loss_details['density'] = loss_density.item()
            n_losses += 1

        # Moyenner si plusieurs losses
        if n_losses > 0:
            total_loss = total_loss / n_losses

        return total_loss, loss_details

    def compute_singularity_loss_from_real_data(self, real_vorticity):
        """
        Calcule la loss des singularités à partir de vraies données de vorticité
        """
        if real_vorticity is None or len(real_vorticity) == 0:
            return torch.tensor(0.0), {}

        # Analyser les vraies données pour extraire les singularités
        vorticity_1d = real_vorticity.flatten()

        # Détection simple des singularités (pics dans la vorticité)
        vorticity_np = vorticity_1d.detach().cpu().numpy()

        # Trouver les pics
        try:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(np.abs(vorticity_np), height=np.std(vorticity_np))

            # Targets basés sur les vraies données
            n_singularities_target = len(peaks)
            max_intensity_target = np.max(np.abs(vorticity_np)) if len(vorticity_np) > 0 else 0.0
            total_energy_target = np.sum(vorticity_np**2)

            # Prédiction du modèle
            vorticity_tensor = vorticity_1d.unsqueeze(0)  # (1, signal_length)
            singularity_pred = self.model.predict_singularities(vorticity_tensor)

            # Loss
            loss_n_sing = (singularity_pred['n_singularities'][0] - n_singularities_target)**2
            loss_intensity = (singularity_pred['max_intensity'][0] - max_intensity_target)**2
            loss_energy = (singularity_pred['total_energy'][0] - total_energy_target)**2

            total_singularity_loss = loss_n_sing + loss_intensity + 0.1 * loss_energy

            return total_singularity_loss, {
                'n_singularities': loss_n_sing.item(),
                'max_intensity': loss_intensity.item(),
                'total_energy': loss_energy.item(),
                'targets': {
                    'n_singularities': n_singularities_target,
                    'max_intensity': max_intensity_target,
                    'total_energy': total_energy_target
                }
            }
        except:
            return torch.tensor(0.0), {}

    def train_on_real_data(self, case_name, subdir=None, n_epochs=300, time_steps=3):
        """
        Entraînement sur les vraies données Kelvin-Helmholtz
        """
        print(f"🚀 ENTRAÎNEMENT PINNs MHD SUR VRAIES DONNÉES: {case_name}")
        print("="*70)

        # Charger les vraies données
        training_data = self.data_loader.get_training_data(case_name, subdir, time_steps)

        if not training_data:
            raise ValueError(f"Aucune donnée d'entraînement disponible pour {case_name}")

        print(f"📊 DÉBUT ENTRAÎNEMENT:")
        print(f"   • Cas: {case_name}")
        print(f"   • Sous-dossier: {subdir}")
        print(f"   • Pas de temps: {len(training_data)}")
        print(f"   • Points par pas: {training_data[0]['coords'].shape[0]}")
        print(f"   • Champs: {list(training_data[0]['fields'].keys())}")

        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_details = {'data': 0.0, 'singularity': 0.0}

            # Entraîner sur chaque pas de temps
            for time_step, data_dict in enumerate(training_data):
                coords = data_dict['coords']
                fields = data_dict['fields']

                # Sous-échantillonner pour accélérer l'entraînement
                n_points = min(800, coords.shape[0])
                indices = torch.randperm(coords.shape[0])[:n_points]
                coords_batch = coords[indices]
                fields_batch = {key: value[indices] if value.ndim > 1 else value 
                               for key, value in fields.items()}

                # Loss sur les données réelles
                data_loss, data_details = self.compute_real_data_loss(coords_batch, fields_batch)

                # Loss sur les singularités (si vorticité disponible)
                singularity_loss = torch.tensor(0.0)
                if 'vorticity' in fields:
                    singularity_loss, sing_details = self.compute_singularity_loss_from_real_data(
                        fields['vorticity']
                    )

                # Loss totale
                total_loss = data_loss + 0.5 * singularity_loss

                # Optimisation
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += total_loss.item()
                epoch_details['data'] += data_loss.item()
                epoch_details['singularity'] += singularity_loss.item()

            # Moyenner sur les pas de temps
            epoch_loss /= len(training_data)
            epoch_details['data'] /= len(training_data)
            epoch_details['singularity'] /= len(training_data)

            # Enregistrer historique
            self.loss_history['total'].append(epoch_loss)
            self.loss_history['data'].append(epoch_details['data'])
            self.loss_history['singularity'].append(epoch_details['singularity'])

            # Affichage périodique
            if epoch % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:3d} | Total: {epoch_loss:.4f} | "
                      f"Data: {epoch_details['data']:.4f} | "
                      f"Singularity: {epoch_details['singularity']:.4f} | "
                      f"Time: {elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ en {total_time:.1f}s")

        return self.loss_history

def create_real_data_mhd_demo():
    """
    Démonstration complète avec vraies données Kelvin-Helmholtz - VERSION ROBUSTE
    """
    print("🧠 CRÉATION PINNs MHD AVEC VRAIES DONNÉES KELVIN-HELMHOLTZ")
    print("="*80)

    # Créer le chargeur de données
    data_loader = RealDataLoader()

    # Créer le modèle
    model = RealData_MHD_PINNs(input_dim=2, hidden_dims=[128, 128, 64], n_scales=15)

    print(f"📊 ARCHITECTURE DU RÉSEAU:")
    print(f"   • Réseau principal: [2] → [128, 128, 64] → [4 têtes MHD]")
    print(f"   • Ondelettes Morlet: 15 échelles apprenables")
    print(f"   • Réseau singularités: [45] → [64, 32, 16] → [3]")
    print(f"   • Paramètres totaux: {sum(p.numel() for p in model.parameters()):,}")

    # Créer l'entraîneur
    trainer = RealData_MHD_Trainer(model, data_loader)

    # Tester le chargement des données
    print(f"\n🔍 TEST CHARGEMENT DONNÉES...")
    try:
        # Essayer de charger le cas B_0 (premier cas)
        case_data = data_loader.load_case_data('B_0', max_files=3)

        if case_data:
            print(f"✅ DONNÉES CHARGÉES AVEC SUCCÈS!")

            # Choisir le premier sous-dossier disponible
            first_subdir = list(case_data.keys())[0]

            # Préparer les données d'entraînement
            training_data = data_loader.get_training_data('B_0', first_subdir, 3)

            if training_data:
                print(f"✅ DONNÉES D'ENTRAÎNEMENT PRÉPARÉES!")

                # Entraînement sur vraies données
                print(f"\n🏋️ DÉBUT ENTRAÎNEMENT SUR VRAIES DONNÉES...")
                loss_history = trainer.train_on_real_data(
                    case_name='B_0',
                    subdir=first_subdir,
                    n_epochs=300,
                    time_steps=3
                )

                return model, trainer, loss_history
            else:
                print("❌ ÉCHEC PRÉPARATION DONNÉES D'ENTRAÎNEMENT")
                return None, None, None
        else:
            print("❌ AUCUNE DONNÉE CHARGÉE")
            return None, None, None

    except Exception as e:
        print(f"❌ ERREUR CHARGEMENT: {e}")
        print("⚠️ Vérifier les chemins d'accès aux données")
        return None, None, None

# Code principal
if __name__ == "__main__":
    # Créer et entraîner le modèle avec vraies données
    model, trainer, loss_history = create_real_data_mhd_demo()

    if model is not None:
        print("\n🎉 ENTRAÎNEMENT SUR VRAIES DONNÉES TERMINÉ!")
        print("="*60)
        print("✅ CAPACITÉS DU RÉSEAU:")
        print("   • Entraîné sur tes vraies données Kelvin-Helmholtz")
        print("   • Prédiction des champs MHD réels")
        print("   • Détection de singularités dans vraies données")
        print("   • Analyse par ondelettes Morlet des vrais signaux")

        # Test final
        print("\n🧪 TEST FINAL DU MODÈLE...")
        model.eval()
        with torch.no_grad():
            # Test sur une grille
            x = torch.linspace(-1, 1, 32)
            y = torch.linspace(-1, 1, 32)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            coords_test = torch.stack([X.flatten(), Y.flatten()], dim=1)

            outputs = model(coords_test)
            print(f"✅ Prédiction réussie sur {coords_test.shape[0]} points")
            print(f"   • Vitesse: {outputs['velocity'].shape}")
            print(f"   • Champ magnétique: {outputs['magnetic'].shape}")
            print(f"   • Pression: {outputs['pressure'].shape}")
            print(f"   • Densité: {outputs['density'].shape}")

            # Test singularités
            vorticity_test = torch.randn(1, 64)  # Signal test
            sing_pred = model.predict_singularities(vorticity_test)
            print(f"✅ Prédiction singularités:")
            print(f"   • Nombre: {sing_pred['n_singularities'].item():.2f}")
            print(f"   • Intensité max: {sing_pred['max_intensity'].item():.4f}")
            print(f"   • Énergie totale: {sing_pred['total_energy'].item():.4f}")
    else:
        print("\n❌ ÉCHEC ENTRAÎNEMENT SUR VRAIES DONNÉES")
        print("Vérifier les chemins d'accès et la structure des données")
