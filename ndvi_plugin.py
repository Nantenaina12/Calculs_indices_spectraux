from qgis.PyQt.QtWidgets import (QAction, QDialog, QVBoxLayout, QPushButton, 
                                 QFileDialog, QLabel, QComboBox, QMessageBox, 
                                 QProgressBar, QGroupBox, QRadioButton, QCheckBox,
                                 QScrollArea, QWidget, QHBoxLayout, QFrame,
                                 QGridLayout, QApplication)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QIcon, QFont
from qgis.core import QgsRasterLayer, QgsProject, QgsMessageLog
import processing
from osgeo import gdal
import numpy as np
import os
import re
from datetime import datetime

class NDVICalculatorPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = u'&Indices Spectraux'
        
    def initGui(self):
        self.action = QAction(
            QIcon(), 
            u'Calculer Indices Spectraux (Landsat MTL)', 
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.menu, self.action)
        self.actions.append(self.action)
        
    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(u'&Indices Spectraux', action)
            
    def run(self):
        dialog = NDVIDialog(self.iface)
        dialog.resize(700, 600)
        dialog.show()
        dialog.exec_()

class CalculationThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, indices_to_calc, band_data, geotransform, projection, 
                 raster_dir, mtl_data, correction_type, red_band, nir_band, 
                 blue_band, green_band, swir1_band, swir2_band, output_path):
        super().__init__()
        self.indices_to_calc = indices_to_calc
        self.band_data = band_data
        self.geotransform = geotransform
        self.projection = projection
        self.raster_dir = raster_dir
        self.mtl_data = mtl_data
        self.correction_type = correction_type
        self.red_band = red_band
        self.nir_band = nir_band
        self.blue_band = blue_band
        self.green_band = green_band
        self.swir1_band = swir1_band
        self.swir2_band = swir2_band
        self.output_path = output_path
        
    def run(self):
        results = {}
        total_indices = len(self.indices_to_calc)
        
        for i, index_name in enumerate(self.indices_to_calc):
            self.progress.emit(int((i / total_indices) * 100))
            self.status.emit(f"Calcul de {index_name}...")
            
            try:
                if index_name == 'NDVI':
                    result = self.calculate_ndvi()
                elif index_name == 'Healthy Vegetation (IPVI)':
                    result = self.calculate_ipvi()
                elif index_name == 'MNDWI (Détection objets flottants)':
                    result = self.calculate_mndwi()
                elif index_name == 'NDWI':
                    result = self.calculate_ndwi()
                elif index_name == 'SAVI':
                    result = self.calculate_savi()
                elif index_name == 'NDBI':
                    result = self.calculate_ndbi()
                elif index_name == 'EVI':
                    result = self.calculate_evi()
                    
                if result is not None:
                    results[index_name] = result
                    
            except Exception as e:
                QgsMessageLog.logMessage(f"Erreur {index_name}: {str(e)}", "Indices Plugin")
                
            self.progress.emit(int(((i + 1) / total_indices) * 100))
            
        self.finished_signal.emit(results)
        
    def calculate_ndvi(self):
        """(NIR - RED) / (NIR + RED)"""
        denominator = self.band_data['nir'] + self.band_data['red']
        denominator[denominator == 0] = 1e-10
        ndvi = (self.band_data['nir'] - self.band_data['red']) / denominator
        return np.clip(ndvi, -1, 1)
    
    def calculate_ipvi(self):
        """Indice de végétation saine (Infrared Percentage Vegetation Index) = NIR / (NIR + RED)"""
        denominator = self.band_data['nir'] + self.band_data['red']
        denominator[denominator == 0] = 1e-10
        ipvi = self.band_data['nir'] / denominator
        return np.clip(ipvi, 0, 1)
    
    def calculate_mndwi(self):
        """Modified NDWI pour détection objets flottants/zone humide = (GREEN - SWIR1) / (GREEN + SWIR1)"""
        if 'green' not in self.band_data or 'swir1' not in self.band_data:
            raise Exception("Bandes nécessaires manquantes (GREEN, SWIR1)")
        denominator = self.band_data['green'] + self.band_data['swir1']
        denominator[denominator == 0] = 1e-10
        mndwi = (self.band_data['green'] - self.band_data['swir1']) / denominator
        return np.clip(mndwi, -1, 1)
    
    def calculate_ndwi(self):
        """NDWI pour détection eau = (GREEN - NIR) / (GREEN + NIR)"""
        denominator = self.band_data['green'] + self.band_data['nir']
        denominator[denominator == 0] = 1e-10
        ndwi = (self.band_data['green'] - self.band_data['nir']) / denominator
        return np.clip(ndwi, -1, 1)
    
    def calculate_savi(self, L=0.5):
        """Soil Adjusted Vegetation Index = ((NIR - RED) / (NIR + RED + L)) * (1 + L)"""
        denominator = self.band_data['nir'] + self.band_data['red'] + L
        denominator[denominator == 0] = 1e-10
        savi = ((self.band_data['nir'] - self.band_data['red']) / denominator) * (1 + L)
        return np.clip(savi, -1, 1)
    
    def calculate_ndbi(self):
        """NDBI pour détection bâti = (SWIR1 - NIR) / (SWIR1 + NIR)"""
        if 'swir1' not in self.band_data:
            raise Exception("Bande SWIR1 nécessaire")
        denominator = self.band_data['swir1'] + self.band_data['nir']
        denominator[denominator == 0] = 1e-10
        ndbi = (self.band_data['swir1'] - self.band_data['nir']) / denominator
        return np.clip(ndbi, -1, 1)
    
    def calculate_evi(self):
        """Enhanced Vegetation Index = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))"""
        if 'blue' not in self.band_data:
            raise Exception("Bande BLUE nécessaire pour EVI")
        denominator = self.band_data['nir'] + 6*self.band_data['red'] - 7.5*self.band_data['blue'] + 1
        denominator[denominator == 0] = 1e-10
        evi = 2.5 * ((self.band_data['nir'] - self.band_data['red']) / denominator)
        return np.clip(evi, -1, 1)

class NDVIDialog(QDialog):
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.mtl_path = None
        self.raster_dir = None
        self.output_path = None
        self.band_files = {}
        self.mtl_data = {}
        self.calculation_thread = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Calculateur d\'Indices Spectraux - Traitement Landsat avec MTL')
        self.setMinimumWidth(800)
        self.setMinimumHeight(700)
        
        # Widget principal et layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        
        # Création d'une zone défilable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Widget conteneur pour le scroll
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        # Groupe 1: Sélection du répertoire
        group1 = QGroupBox("1. 📁 Sélection du répertoire des images Landsat")
        group1_layout = QVBoxLayout()
        
        self.btn_select_directory = QPushButton('📂 Parcourir le répertoire contenant les bandes et le fichier MTL')
        self.btn_select_directory.clicked.connect(self.select_directory)
        self.btn_select_directory.setMinimumHeight(35)
        group1_layout.addWidget(self.btn_select_directory)
        
        self.lbl_directory = QLabel('Aucun répertoire sélectionné')
        self.lbl_directory.setWordWrap(True)
        self.lbl_directory.setStyleSheet("color: gray; padding: 5px;")
        group1_layout.addWidget(self.lbl_directory)
        
        self.lbl_mtl_status = QLabel('')
        self.lbl_mtl_status.setWordWrap(True)
        group1_layout.addWidget(self.lbl_mtl_status)
        
        group1.setLayout(group1_layout)
        scroll_layout.addWidget(group1)
        
        # Groupe 2: Information bandes
        group2 = QGroupBox("2. 📊 Bandes détectées")
        group2_layout = QVBoxLayout()
        
        self.lbl_bands_info = QLabel('En attente de sélection...')
        self.lbl_bands_info.setWordWrap(True)
        self.lbl_bands_info.setStyleSheet("font-family: monospace; background-color: #f5f5f5; padding: 5px;")
        group2_layout.addWidget(self.lbl_bands_info)
        
        group2.setLayout(group2_layout)
        scroll_layout.addWidget(group2)
        
        # Groupe 3: Sélection des bandes
        group3 = QGroupBox("3. 🎨 Configuration des bandes")
        group3_layout = QGridLayout()
        
        # Labels et combobox pour différentes bandes
        row = 0
        group3_layout.addWidget(QLabel('Bande Rouge (RED):'), row, 0)
        self.combo_red = QComboBox()
        self.combo_red.setMinimumWidth(150)
        group3_layout.addWidget(self.combo_red, row, 1)
        group3_layout.addWidget(QLabel('(B4 pour Landsat 8-9)'), row, 2)
        
        row += 1
        group3_layout.addWidget(QLabel('Bande NIR (Proche IR):'), row, 0)
        self.combo_nir = QComboBox()
        group3_layout.addWidget(self.combo_nir, row, 1)
        group3_layout.addWidget(QLabel('(B5 pour Landsat 8-9)'), row, 2)
        
        row += 1
        group3_layout.addWidget(QLabel('Bande GREEN (Vert):'), row, 0)
        self.combo_green = QComboBox()
        group3_layout.addWidget(self.combo_green, row, 1)
        group3_layout.addWidget(QLabel('(B3 pour Landsat 8-9)'), row, 2)
        
        row += 1
        group3_layout.addWidget(QLabel('Bande BLUE (Bleu):'), row, 0)
        self.combo_blue = QComboBox()
        group3_layout.addWidget(self.combo_blue, row, 1)
        group3_layout.addWidget(QLabel('(B2 pour Landsat 8-9)'), row, 2)
        
        row += 1
        group3_layout.addWidget(QLabel('Bande SWIR1 (IR moyen):'), row, 0)
        self.combo_swir1 = QComboBox()
        group3_layout.addWidget(self.combo_swir1, row, 1)
        group3_layout.addWidget(QLabel('(B6 pour Landsat 8-9)'), row, 2)
        
        row += 1
        group3_layout.addWidget(QLabel('Bande SWIR2 (IR moyen 2):'), row, 0)
        self.combo_swir2 = QComboBox()
        group3_layout.addWidget(self.combo_swir2, row, 1)
        group3_layout.addWidget(QLabel('(B7 pour Landsat 8-9)'), row, 2)
        
        group3.setLayout(group3_layout)
        scroll_layout.addWidget(group3)
        
        # Groupe 4: Correction atmosphérique
        group4 = QGroupBox("4. 🌤️ Correction atmosphérique")
        group4_layout = QVBoxLayout()
        
        self.radio_reflectance = QRadioButton("✅ Conversion en réflectance (recommandé - avec coefficients MTL)")
        self.radio_reflectance.setChecked(True)
        group4_layout.addWidget(self.radio_reflectance)
        
        self.radio_dos1 = QRadioButton("🌑 DOS1 (Dark Object Subtraction)")
        group4_layout.addWidget(self.radio_dos1)
        
        self.radio_raw = QRadioButton("📊 Aucune correction (valeurs brutes DN)")
        group4_layout.addWidget(self.radio_raw)
        
        self.lbl_correction_note = QLabel("💡 Info: La conversion en réflectance utilise les coefficients GAIN, BIAS et SUN_ELEVATION du fichier MTL")
        self.lbl_correction_note.setWordWrap(True)
        self.lbl_correction_note.setStyleSheet("color: #0066cc; font-size: 10px; padding: 5px;")
        group4_layout.addWidget(self.lbl_correction_note)
        
        group4.setLayout(group4_layout)
        scroll_layout.addWidget(group4)
        
        # Groupe 5: Sélection des indices
        group5 = QGroupBox("5. 📈 Indices spectraux à calculer")
        group5_layout = QVBoxLayout()
        
        # Checkbox pour chaque indice
        self.check_ndvi = QCheckBox("🌿 NDVI - Normalized Difference Vegetation Index")
        self.check_ndvi.setChecked(True)
        group5_layout.addWidget(self.check_ndvi)
        
        self.check_ipvi = QCheckBox("🌱 IPVI - Infrared Percentage Vegetation Index (Végétation saine)")
        group5_layout.addWidget(self.check_ipvi)
        
        self.check_mndwi = QCheckBox("💧 MNDWI - Modified NDWI (Détection objets flottants/surfaces en eau)")
        self.check_mndwi.setChecked(True)
        group5_layout.addWidget(self.check_mndwi)
        
        self.check_ndwi = QCheckBox("💦 NDWI - Normalized Difference Water Index")
        group5_layout.addWidget(self.check_ndwi)
        
        self.check_savi = QCheckBox("🌾 SAVI - Soil Adjusted Vegetation Index (avec correction du sol)")
        group5_layout.addWidget(self.check_savi)
        
        self.check_ndbi = QCheckBox("🏙️ NDBI - Normalized Difference Built-up Index (Détection bâti)")
        group5_layout.addWidget(self.check_ndbi)
        
        self.check_evi = QCheckBox("🍃 EVI - Enhanced Vegetation Index")
        group5_layout.addWidget(self.check_evi)
        
        group5.setLayout(group5_layout)
        scroll_layout.addWidget(group5)
        
        # Groupe 6: Sortie
        group6 = QGroupBox("6. 💾 Répertoire de sortie")
        group6_layout = QVBoxLayout()
        
        self.btn_select_output = QPushButton('📂 Sélectionner le répertoire de sortie')
        self.btn_select_output.clicked.connect(self.select_output)
        self.btn_select_output.setEnabled(False)
        self.btn_select_output.setMinimumHeight(35)
        group6_layout.addWidget(self.btn_select_output)
        
        self.lbl_output = QLabel('Aucun répertoire sélectionné')
        self.lbl_output.setStyleSheet("color: gray; padding: 5px;")
        group6_layout.addWidget(self.lbl_output)
        
        group6.setLayout(group6_layout)
        scroll_layout.addWidget(group6)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        scroll_layout.addWidget(self.progress_bar)
        
        # Label de statut
        self.status_label = QLabel('')
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet("color: #0066cc; padding: 5px;")
        scroll_layout.addWidget(self.status_label)
        
        # Bouton RUN
        self.btn_calculate = QPushButton('🚀 LANCER LE CALCUL DES INDICES')
        self.btn_calculate.clicked.connect(self.calculate_indices)
        self.btn_calculate.setEnabled(False)
        self.btn_calculate.setMinimumHeight(50)
        self.btn_calculate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        scroll_layout.addWidget(self.btn_calculate)
        
        # Ajouter un stretch à la fin
        scroll_layout.addStretch()
        
        # Configurer le scroll
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        self.setLayout(main_layout)
        
    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Sélectionner le répertoire contenant les bandes Landsat et le fichier MTL"
        )
        
        if directory:
            self.raster_dir = directory
            self.lbl_directory.setText(f'📁 {directory}')
            self.find_mtl_and_bands()
            
    def find_mtl_and_bands(self):
        if not self.raster_dir:
            return
            
        # Chercher le fichier MTL
        mtl_files = []
        pattern_mtl = re.compile(r'.*MTL\.txt$', re.IGNORECASE)
        
        for file in os.listdir(self.raster_dir):
            if pattern_mtl.match(file):
                mtl_files.append(file)
        
        if not mtl_files:
            QMessageBox.warning(self, "Erreur", 
                              "Aucun fichier MTL (MTL.txt) trouvé!\n\n"
                              "Le répertoire doit contenir:\n"
                              "- Le fichier *_MTL.txt\n"
                              "- Les bandes Landsat (ex: *B1.TIF, *B2.TIF, etc.)")
            self.lbl_mtl_status.setText('❌ Fichier MTL non trouvé!')
            return
            
        self.mtl_path = os.path.join(self.raster_dir, mtl_files[0])
        self.lbl_mtl_status.setText(f'✅ Fichier MTL trouvé: {mtl_files[0]}')
        
        self.extract_bands()
        self.read_mtl_file()
        self.update_band_comboboxes()
        
    def extract_bands(self):
        self.band_files = {}
        
        patterns = {
            1: re.compile(r'.*[_-]B1\.TIF$', re.IGNORECASE),
            2: re.compile(r'.*[_-]B2\.TIF$', re.IGNORECASE),
            3: re.compile(r'.*[_-]B3\.TIF$', re.IGNORECASE),
            4: re.compile(r'.*[_-]B4\.TIF$', re.IGNORECASE),
            5: re.compile(r'.*[_-]B5\.TIF$', re.IGNORECASE),
            6: re.compile(r'.*[_-]B6\.TIF$', re.IGNORECASE),
            7: re.compile(r'.*[_-]B7\.TIF$', re.IGNORECASE),
            8: re.compile(r'.*[_-]B8\.TIF$', re.IGNORECASE),
            9: re.compile(r'.*[_-]B9\.TIF$', re.IGNORECASE),
            10: re.compile(r'.*[_-]B10\.TIF$', re.IGNORECASE),
            11: re.compile(r'.*[_-]B11\.TIF$', re.IGNORECASE),
        }
        
        for file in os.listdir(self.raster_dir):
            for band_num, pattern in patterns.items():
                if pattern.match(file):
                    self.band_files[band_num] = os.path.join(self.raster_dir, file)
                    break
        
        if self.band_files:
            info_text = f"✅ {len(self.band_files)} bandes trouvées:\n"
            for band_num in sorted(self.band_files.keys()):
                band_name = os.path.basename(self.band_files[band_num])
                info_text += f"   • Bande {band_num}: {band_name}\n"
            self.lbl_bands_info.setText(info_text)
            self.btn_select_output.setEnabled(True)
        else:
            self.lbl_bands_info.setText("❌ Aucune bande Landsat trouvée!")
            self.btn_select_output.setEnabled(False)
            
    def read_mtl_file(self):
        self.mtl_data = {}
        
        try:
            with open(self.mtl_path, 'r') as f:
                content = f.read()
                
            for band in range(1, 12):
                pattern_reflectance_mult = rf'REFLECTANCE_MULT_BAND_{band}\s*=\s*([\d\.\-]+)'
                pattern_reflectance_add = rf'REFLECTANCE_ADD_BAND_{band}\s*=\s*([\d\.\-]+)'
                
                ref_mult_match = re.search(pattern_reflectance_mult, content, re.IGNORECASE)
                ref_add_match = re.search(pattern_reflectance_add, content, re.IGNORECASE)
                
                if ref_mult_match:
                    self.mtl_data[f'REFLECTANCE_MULT_{band}'] = float(ref_mult_match.group(1))
                if ref_add_match:
                    self.mtl_data[f'REFLECTANCE_ADD_{band}'] = float(ref_add_match.group(1))
                    
            sun_elev_match = re.search(r'SUN_ELEVATION\s*=\s*([\d\.\-]+)', content, re.IGNORECASE)
            if sun_elev_match:
                self.mtl_data['SUN_ELEVATION'] = float(sun_elev_match.group(1))
                
            date_match = re.search(r'DATE_ACQUIRED\s*=\s*"([^"]+)"', content, re.IGNORECASE)
            if date_match:
                self.mtl_data['DATE_ACQUIRED'] = date_match.group(1)
                
            summary = f"📊 Résumé MTL:\n"
            if 'SUN_ELEVATION' in self.mtl_data:
                summary += f"   • Élévation solaire: {self.mtl_data['SUN_ELEVATION']}°\n"
            if 'DATE_ACQUIRED' in self.mtl_data:
                summary += f"   • Date: {self.mtl_data['DATE_ACQUIRED']}\n"
            if self.mtl_data:
                self.lbl_mtl_status.setText(f"✅ {summary}")
                
        except Exception as e:
            self.lbl_mtl_status.setText(f"⚠️ Erreur lecture MTL: {str(e)}")
            
    def update_band_comboboxes(self):
        combos = [self.combo_red, self.combo_nir, self.combo_green, 
                  self.combo_blue, self.combo_swir1, self.combo_swir2]
        
        for combo in combos:
            combo.clear()
        
        available_bands = sorted(self.band_files.keys())
        
        for band_num in available_bands:
            band_info = f"Bande {band_num}"
            for combo in combos:
                combo.addItem(band_info, band_num)
        
        # Sélections par défaut
        if 4 in available_bands:
            index = self.combo_red.findData(4)
            if index >= 0: self.combo_red.setCurrentIndex(index)
        if 5 in available_bands:
            index = self.combo_nir.findData(5)
            if index >= 0: self.combo_nir.setCurrentIndex(index)
        if 3 in available_bands:
            index = self.combo_green.findData(3)
            if index >= 0: self.combo_green.setCurrentIndex(index)
        if 2 in available_bands:
            index = self.combo_blue.findData(2)
            if index >= 0: self.combo_blue.setCurrentIndex(index)
        if 6 in available_bands:
            index = self.combo_swir1.findData(6)
            if index >= 0: self.combo_swir1.setCurrentIndex(index)
        if 7 in available_bands:
            index = self.combo_swir2.findData(7)
            if index >= 0: self.combo_swir2.setCurrentIndex(index)
                
    def select_output(self):
        output_dir = QFileDialog.getExistingDirectory(self, "Sélectionner le répertoire de sortie")
        if output_dir:
            self.output_path = output_dir
            self.lbl_output.setText(f'📂 {output_dir}')
            self.btn_calculate.setEnabled(True)
            
    def apply_reflectance_correction(self, band_data, band_num):
        ref_mult_key = f'REFLECTANCE_MULT_{band_num}'
        ref_add_key = f'REFLECTANCE_ADD_{band_num}'
        
        if ref_mult_key in self.mtl_data:
            reflectance_mult = self.mtl_data[ref_mult_key]
            reflectance_add = self.mtl_data.get(ref_add_key, 0)
            reflectance = band_data * reflectance_mult + reflectance_add
            
            if 'SUN_ELEVATION' in self.mtl_data:
                sun_elevation_rad = np.radians(self.mtl_data['SUN_ELEVATION'])
                reflectance = reflectance / np.sin(sun_elevation_rad)
            
            reflectance = np.clip(reflectance, 0, 1) * 255
        else:
            reflectance = band_data
            
        return reflectance.astype(np.float32)
    
    def apply_dos1_correction(self, band_data):
        dark_pixel = np.percentile(band_data[band_data > 0], 1)
        corrected = band_data - dark_pixel
        corrected[corrected < 0] = 0
        return corrected.astype(np.float32)
    
    def load_band_data(self, band_num):
        if band_num not in self.band_files:
            return None, None, None
            
        dataset = gdal.Open(self.band_files[band_num])
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float32)
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dataset = None
        
        return data, geotransform, projection
        
    def calculate_indices(self):
        try:
            # Récupérer les indices sélectionnés
            indices_to_calc = []
            if self.check_ndvi.isChecked(): indices_to_calc.append('NDVI')
            if self.check_ipvi.isChecked(): indices_to_calc.append('Healthy Vegetation (IPVI)')
            if self.check_mndwi.isChecked(): indices_to_calc.append('MNDWI (Détection objets flottants)')
            if self.check_ndwi.isChecked(): indices_to_calc.append('NDWI')
            if self.check_savi.isChecked(): indices_to_calc.append('SAVI')
            if self.check_ndbi.isChecked(): indices_to_calc.append('NDBI')
            if self.check_evi.isChecked(): indices_to_calc.append('EVI')
            
            if not indices_to_calc:
                QMessageBox.warning(self, "Erreur", "Veuillez sélectionner au moins un indice à calculer!")
                return
                
            if not self.raster_dir or not self.output_path:
                QMessageBox.warning(self, "Erreur", "Veuillez sélectionner le répertoire d'entrée et de sortie!")
                return
                
            # Charger les bandes
            self.progress_bar.setVisible(True)
            self.status_label.setVisible(True)
            self.btn_calculate.setEnabled(False)
            
            red_band_num = self.combo_red.currentData()
            nir_band_num = self.combo_nir.currentData()
            green_band_num = self.combo_green.currentData()
            blue_band_num = self.combo_blue.currentData()
            swir1_band_num = self.combo_swir1.currentData()
            swir2_band_num = self.combo_swir2.currentData()
            
            # Chargement des données
            self.status_label.setText("Chargement des bandes...")
            QApplication.processEvents()
            
            band_data = {}
            geotransform = None
            projection = None
            
            if red_band_num:
                red_data, gt, proj = self.load_band_data(red_band_num)
                if red_data is not None:
                    band_data['red'] = red_data
                    geotransform = gt
                    projection = proj
                    
            if nir_band_num:
                nir_data, _, _ = self.load_band_data(nir_band_num)
                if nir_data is not None:
                    band_data['nir'] = nir_data
                    
            if green_band_num:
                green_data, _, _ = self.load_band_data(green_band_num)
                if green_data is not None:
                    band_data['green'] = green_data
                    
            if blue_band_num:
                blue_data, _, _ = self.load_band_data(blue_band_num)
                if blue_data is not None:
                    band_data['blue'] = blue_data
                    
            if swir1_band_num:
                swir1_data, _, _ = self.load_band_data(swir1_band_num)
                if swir1_data is not None:
                    band_data['swir1'] = swir1_data
                    
            if swir2_band_num:
                swir2_data, _, _ = self.load_band_data(swir2_band_num)
                if swir2_data is not None:
                    band_data['swir2'] = swir2_data
            
            # Appliquer correction atmosphérique
            correction_type = ""
            if self.radio_reflectance.isChecked():
                correction_type = "Reflectance_MTL"
                self.status_label.setText("Application correction réflectance...")
                QApplication.processEvents()
                for key in band_data:
                    band_num = None
                    if key == 'red': band_num = red_band_num
                    elif key == 'nir': band_num = nir_band_num
                    elif key == 'green': band_num = green_band_num
                    elif key == 'blue': band_num = blue_band_num
                    elif key == 'swir1': band_num = swir1_band_num
                    elif key == 'swir2': band_num = swir2_band_num
                    
                    if band_num:
                        band_data[key] = self.apply_reflectance_correction(band_data[key], band_num)
                        
            elif self.radio_dos1.isChecked():
                correction_type = "DOS1"
                self.status_label.setText("Application correction DOS1...")
                QApplication.processEvents()
                for key in band_data:
                    band_data[key] = self.apply_dos1_correction(band_data[key])
            else:
                correction_type = "Raw_DN"
            
            # Lancer le calcul dans un thread séparé
            self.calculation_thread = CalculationThread(
                indices_to_calc, band_data, geotransform, projection,
                self.raster_dir, self.mtl_data, correction_type,
                red_band_num, nir_band_num, green_band_num, blue_band_num,
                swir1_band_num, swir2_band_num, self.output_path
            )
            
            self.calculation_thread.progress.connect(self.update_progress)
            self.calculation_thread.status.connect(self.update_status)
            self.calculation_thread.finished_signal.connect(self.on_calculation_finished)
            self.calculation_thread.error.connect(self.on_calculation_error)
            
            self.calculation_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"❌ Erreur: {str(e)}")
            self.progress_bar.setVisible(False)
            self.status_label.setVisible(False)
            self.btn_calculate.setEnabled(True)
            
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        self.status_label.setText(message)
        QApplication.processEvents()
        
    def on_calculation_finished(self, results):
        dir_name = os.path.basename(self.raster_dir.rstrip('/\\'))
        date_str = self.mtl_data.get('DATE_ACQUIRED', datetime.now().strftime('%Y%m%d')).replace('-', '')
        
        success_count = 0
        for index_name, result_data in results.items():
            index_short = index_name.split('(')[0].strip().replace(' ', '_')
            output_filename = f"{index_short}_{dir_name}_{date_str}_{self.calculation_thread.correction_type}.tif"
            output_filepath = os.path.join(self.output_path, output_filename)
            
            # Sauvegarde
            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(
                output_filepath,
                result_data.shape[1],
                result_data.shape[0],
                1,
                gdal.GDT_Float32
            )
            
            out_dataset.SetGeoTransform(self.calculation_thread.geotransform)
            out_dataset.SetProjection(self.calculation_thread.projection)
            
            out_band = out_dataset.GetRasterBand(1)
            out_band.WriteArray(result_data)
            out_band.SetNoDataValue(-999)
            out_band.SetDescription(f'{index_name} - {self.calculation_thread.correction_type}')
            
            out_dataset.FlushCache()
            out_dataset = None
            
            # Chargement dans QGIS
            layer_name = f"{index_short}_{dir_name}"
            raster_layer = QgsRasterLayer(output_filepath, layer_name)
            if raster_layer.isValid():
                QgsProject.instance().addMapLayer(raster_layer)
                self.apply_index_style(raster_layer, index_name)
                
            success_count += 1
            
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.btn_calculate.setEnabled(True)
        
        QMessageBox.information(self, "Succès", 
                               f"✅ {success_count} indice(s) calculé(s) avec succès!\n"
                               f"📁 Résultats sauvegardés dans: {self.output_path}\n"
                               f"🎯 Correction: {self.calculation_thread.correction_type}")
        
    def on_calculation_error(self, error_msg):
        QMessageBox.critical(self, "Erreur", f"❌ Erreur lors du calcul: {error_msg}")
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)
        self.btn_calculate.setEnabled(True)
        
    def apply_index_style(self, layer, index_name):
        from qgis.core import QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer
        from qgis.PyQt.QtGui import QColor
        
        fcn = QgsColorRampShader()
        fcn.setColorRampType(QgsColorRampShader.Interpolated)
        
        if 'NDVI' in index_name or 'Vegetation' in index_name or 'EVI' in index_name or 'SAVI' in index_name:
            lst = [
                QgsColorRampShader.ColorRampItem(-1.0, QColor(0, 0, 255), 'Eau'),
                QgsColorRampShader.ColorRampItem(-0.5, QColor(0, 100, 200), 'Eau trouble'),
                QgsColorRampShader.ColorRampItem(-0.1, QColor(100, 100, 100), 'Sol nu'),
                QgsColorRampShader.ColorRampItem(0.1, QColor(150, 100, 50), 'Végétation faible'),
                QgsColorRampShader.ColorRampItem(0.3, QColor(150, 150, 0), 'Végétation modérée'),
                QgsColorRampShader.ColorRampItem(0.5, QColor(0, 150, 0), 'Végétation dense'),
                QgsColorRampShader.ColorRampItem(0.7, QColor(0, 200, 0), 'Végétation très dense'),
                QgsColorRampShader.ColorRampItem(1.0, QColor(0, 255, 0), 'Végétation maximale')
            ]
        elif 'MNDWI' in index_name or 'NDWI' in index_name:
            lst = [
                QgsColorRampShader.ColorRampItem(-1.0, QColor(139, 69, 19), 'Sol/Bâti'),
                QgsColorRampShader.ColorRampItem(-0.5, QColor(160, 82, 45), 'Sols'),
                QgsColorRampShader.ColorRampItem(-0.2, QColor(210, 180, 140), 'Zones humides'),
                QgsColorRampShader.ColorRampItem(0.0, QColor(135, 206, 235), 'Eau peu profonde'),
                QgsColorRampShader.ColorRampItem(0.3, QColor(70, 130, 180), 'Eau modérée'),
                QgsColorRampShader.ColorRampItem(0.7, QColor(0, 0, 139), 'Eau profonde'),
                QgsColorRampShader.ColorRampItem(1.0, QColor(0, 0, 255), 'Eau pure')
            ]
        elif 'NDBI' in index_name:
            lst = [
                QgsColorRampShader.ColorRampItem(-1.0, QColor(34, 139, 34), 'Végétation'),
                QgsColorRampShader.ColorRampItem(-0.5, QColor(107, 142, 35), 'Végétation éparse'),
                QgsColorRampShader.ColorRampItem(-0.1, QColor(210, 180, 140), 'Sol nu'),
                QgsColorRampShader.ColorRampItem(0.1, QColor(169, 169, 169), 'Zones urbaines légères'),
                QgsColorRampShader.ColorRampItem(0.3, QColor(128, 128, 128), 'Zones urbaines modérées'),
                QgsColorRampShader.ColorRampItem(0.7, QColor(105, 105, 105), 'Zones urbaines denses'),
                QgsColorRampShader.ColorRampItem(1.0, QColor(0, 0, 0), 'Bâti dense')
            ]
        else:
            # Palette par défaut
            lst = [
                QgsColorRampShader.ColorRampItem(-1.0, QColor(255, 0, 0), 'Min'),
                QgsColorRampShader.ColorRampItem(0.0, QColor(255, 255, 255), 'Moyen'),
                QgsColorRampShader.ColorRampItem(1.0, QColor(0, 255, 0), 'Max')
            ]
        
        fcn.setColorRampItemList(lst)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(fcn)
        
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()