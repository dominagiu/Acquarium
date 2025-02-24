import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dropout, LSTM, Reshape, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Imposta GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Definizione dei parametri
n_timesteps = 120  # Correzione dell'errore di definizione
infra_features = ['Crest Factor', 'Zero Crossing Rate', 'Peak-to-Peak Amplitude', 'Spectral Bandwidth',
                  'Kurtosis', 'Variance', 'Spectral Flatness', 'Rise Time', 'RMS Amplitude', 'Spectral Centroid']
n_features_infra = len(infra_features)

seismic_features = ['statistics_variance', 'statistics_std_dev', 'energy', 'motion_direction', 'rolling_snr',
                    'peak_count', 'statistics_skewness', 'horizontal_to_vertical_ratio', 'rms', 'statistics_kurtosis']
n_features_seismic = len(seismic_features)

# Caricamento delle etichette effettive
labels_path = 'data/csv_eruzioni/etichette_2021.csv'
labels_df = pd.read_csv(labels_path)
labels_df['start_time'] = pd.to_datetime(labels_df['start_time'])
labels_df['end_time'] = pd.to_datetime(labels_df['end_time'])
actual_intervals = [(row['start_time'], row['end_time']) for _, row in labels_df.iterrows()]

def load_and_process_data(base_path, category, sequence_length=120, actual_intervals=None):
    """
    Carica i dati da file .pkl, assegna i label basati sulle eruzioni e normalizza i dati.
    """
    print(f"\n🔍 Avvio caricamento dati per: {category}")

    # Definisce il tipo di file e le feature da caricare
    if category.upper() == 'INFRA':
        selected_features = ['Crest Factor', 'Zero Crossing Rate', 'Peak-to-Peak Amplitude', 'Spectral Bandwidth',
                             'Kurtosis', 'Variance', 'Spectral Flatness', 'Rise Time', 'RMS Amplitude', 'Spectral Centroid']
        file_suffix = "_cleaned.pkl"  # INFRA usa file con _cleaned.pkl
    elif category.upper() == 'SEISMIC':
        selected_features = ['statistics_variance', 'statistics_std_dev', 'energy', 'motion_direction', 'rolling_snr',
                             'peak_count', 'statistics_skewness', 'horizontal_to_vertical_ratio', 'rms', 'statistics_kurtosis']
        file_suffix = ".pkl"  # SEISMIC usa file normali .pkl
    else:
        raise ValueError(f"Categoria '{category}' non riconosciuta. Usa 'INFRA' o 'SEISMIC'.")

    scaler = MinMaxScaler()
    all_sequences, all_labels = [], []
    category_path = os.path.join(base_path, category)

    if not os.path.exists(category_path) or not os.listdir(category_path):
        print(f"❌ ERRORE: La cartella {category_path} è vuota o non esiste!")
        return np.array([]), np.array([])

    print(f"📂 Stazioni disponibili in {category}: {os.listdir(category_path)}")

    for station in os.listdir(category_path):
        station_path = os.path.join(category_path, station)
        if not os.path.isdir(station_path) or station.startswith('.'):
            continue

        print(f"\n📍 Stazione trovata: {station}")
        print(f"📄 File disponibili: {os.listdir(station_path)}")

        dfs = []
        for feature_file in os.listdir(station_path):
            # Filtra i file in base alla categoria (INFRA usa _cleaned.pkl, SEISMIC no)
            if category.upper() == "INFRA" and not feature_file.endswith("_cleaned.pkl"):
                continue
            if category.upper() == "SEISMIC" and (not feature_file.endswith(".pkl") or feature_file.endswith("_cleaned.pkl")):
                continue

            feature_name = feature_file.replace("_cleaned.pkl", "").replace(".pkl", "").strip()  # Nome pulito della feature
            file_path = os.path.join(station_path, feature_file)

            try:
                data = pd.read_pickle(file_path)
                print(f"✅ Caricato: {feature_file} | Shape: {data.shape}")

                # Se il file è una Series, converti in DataFrame
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=feature_name)

                # Verifica che ci sia una colonna timestamp
                if 'timestamp' not in data.columns:
                    if isinstance(data.index, pd.DatetimeIndex):
                        data = data.reset_index()
                        data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True)
                    else:
                        print(f"⚠️ File {feature_file} non ha timestamp valido. Skipping...")
                        continue

                # Rinominare la colonna "amplitude" con il nome corretto della feature
                if "amplitude" in data.columns:
                    data.rename(columns={"amplitude": feature_name}, inplace=True)

                dfs.append(data)

            except Exception as e:
                print(f"❌ Errore nel caricamento di {feature_file}: {e}")
                continue

        if not dfs:
            print(f"⚠️ Nessun file valido per {station}. Skipping...")
            continue

        # Merge di tutti i DataFrame sulla colonna 'timestamp'
        station_data = dfs[0]
        for df in dfs[1:]:
            station_data = station_data.merge(df, on='timestamp', how='inner')

        station_data = station_data.sort_values(by='timestamp')

        # Debug: Verifica le colonne disponibili dopo il merge
        print(f"📊 Colonne dopo il merge: {station_data.columns.tolist()}")

        # Normalizza i nomi delle colonne per evitare problemi di confronto
        station_data.columns = station_data.columns.str.lower().str.strip()
        selected_features = [feat.lower().strip() for feat in selected_features]

        # Seleziona solo le feature desiderate che esistono nel DataFrame
        available_features = set(station_data.columns)
        selected_features_filtered = [feat for feat in selected_features if feat in available_features]

        print(f"🔢 Feature selezionate per {station}: {selected_features_filtered}")
        if not selected_features_filtered:
            print(f"⚠️ Nessuna feature valida trovata per {station}. Skipping...")
            continue

        # Mantieni solo timestamp + feature desiderate
        station_data = station_data[['timestamp'] + selected_features_filtered].fillna(0)

        # Seleziona solo le colonne numeriche
        numeric_columns = station_data.select_dtypes(include=[np.number]).columns.tolist()

        # Debug: Numero di feature dopo il filtro
        print(f"📊 Numero di feature numeriche: {len(numeric_columns)} (atteso: {len(selected_features)})")

        if not numeric_columns:
            print(f"⚠️ Nessuna colonna numerica valida in {station}. Skipping...")
            continue

        # Normalizzazione dei dati
        station_data[numeric_columns] = scaler.fit_transform(station_data[numeric_columns])
        station_data['timestamp'] = pd.to_datetime(station_data['timestamp'], errors='coerce')

        # Creazione delle sequenze temporali e assegnazione dei label
        sequences, labels = [], []
        for i in range(len(station_data) - sequence_length + 1):
            seq = station_data.iloc[i:i + sequence_length]
            seq_start = seq['timestamp'].iloc[0]
            seq_end = seq['timestamp'].iloc[-1]
            sequences.append(seq[numeric_columns].values)

            # Assegna il label in base agli intervalli delle eruzioni
            label = 0
            if actual_intervals:
                for act_start, act_end in actual_intervals:
                    if (seq_start <= act_end) and (seq_end >= act_start):
                        label = 1
                        break
            labels.append(label)

        all_sequences.extend(sequences)
        all_labels.extend(labels)

    # Debug finale: verifica che i dati abbiano la dimensione corretta
    if all_sequences:
        print(f"✅ Numero di sequenze generate: {len(all_sequences)}")
        print(f"✅ Forma della prima sequenza: {np.array(all_sequences[0]).shape} (atteso: ({sequence_length}, {len(selected_features_filtered)}))")

    return np.array(all_sequences, dtype=np.float32), np.array(all_labels, dtype=np.int32)



# Modifica il formato dei dati per adattarlo a una rete CNN 2D
def reshape_for_cnn(X):
    return X.reshape(X.shape[0], n_timesteps, X.shape[2], 1)  # (samples, time, features, 1 channel)

def build_cnn_lstm_model():
 # Input per dati infrasuono
    input_infra = Input(shape=(n_timesteps, n_features_infra, 1))
    x_infra = Conv2D(32, (3, 3), activation='relu', padding='same')(input_infra)
    x_infra = MaxPooling2D((2, 2))(x_infra)
    x_infra = Conv2D(64, (3, 3), activation='relu', padding='same')(x_infra)
    x_infra = MaxPooling2D((2, 2))(x_infra)
    x_infra = Flatten()(x_infra)
    
    # Reshape per LSTM
    x_infra = Reshape((n_timesteps, -1))(x_infra)  
    x_infra = LSTM(64, return_sequences=True)(x_infra)  # Primo LSTM con return_sequences=True
    x_infra = LSTM(32, return_sequences=False)(x_infra)  # Secondo LSTM senza return_sequences

    # Input per dati sismici
    input_seismic = Input(shape=(n_timesteps, n_features_seismic, 1))
    x_seismic = Conv2D(32, (3, 3), activation='relu', padding='same')(input_seismic)
    x_seismic = MaxPooling2D((2, 2))(x_seismic)
    x_seismic = Conv2D(64, (3, 3), activation='relu', padding='same')(x_seismic)
    x_seismic = MaxPooling2D((2, 2))(x_seismic)
    x_seismic = Flatten()(x_seismic)
    
    # Reshape per LSTM
    x_seismic = Reshape((n_timesteps, -1))(x_seismic)  
    x_seismic = LSTM(64, return_sequences=True)(x_seismic)  # Primo LSTM con return_sequences=True
    x_seismic = LSTM(32, return_sequences=False)(x_seismic)  # Secondo LSTM senza return_sequences

    # Concatenazione delle due reti
    merged = Concatenate()([x_infra, x_seismic])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_infra, input_seismic], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Calcolo delle metriche
def calculate_metrics(actual, predicted):
    if not all(isinstance(p, tuple) and len(p) == 2 for p in predicted):
        raise ValueError("Il parametro 'predicted' deve essere una lista di tuple (pred_start, pred_end).")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    lead_times = []
    correct_alert_time = 0
    total_alert_time = 0

    actual_intervals = [(pd.to_datetime(act_start), pd.to_datetime(act_end)) for act_start, act_end in actual]
    predicted_intervals = [(pd.to_datetime(pred_start), pd.to_datetime(pred_end)) for pred_start, pred_end in predicted]

    for pred_start, pred_end in predicted_intervals:
        total_alert_time += (pred_end - pred_start).total_seconds()
        overlapping = False

        for act_start, act_end in actual_intervals:
            if (pred_start <= act_end) and (pred_end >= act_start):
                overlapping = True
                true_positives += 1
                lead_time = (pred_start - act_start).total_seconds() / 3600
                if lead_time <= 6:
                    lead_times.append(lead_time)
                correct_alert_time += (min(pred_end, act_end) - max(pred_start, act_start)).total_seconds()
                break
        
        if not overlapping:
            false_positives += 1

    for act_start, act_end in actual_intervals:
        if not any((pred_start <= act_end) and (pred_end >= act_start) for pred_start, pred_end in predicted_intervals):
            false_negatives += 1

    TPR = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    FDR = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    FTA = correct_alert_time / total_alert_time if total_alert_time > 0 else 0
    avg_lead_time = np.mean(lead_times) if lead_times else 0
    print(f'TPR: {TPR}, FDR: {FDR}, FTA: {FTA}, Average Lead Time: {avg_lead_time}')
    
    metrics_file = 'CNN_metrics_final.csv'
    metrics_df = pd.DataFrame([{'TPR': TPR, 'FDR': FDR, 'FTA': FTA, 'Average Lead Time': avg_lead_time}])
    metrics_df.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)

    return TPR, FDR, FTA, avg_lead_time

def save_predicted_events(y_pred_proba, timestamps, threshold=0.8, output_csv="cnn_lstm_predicted_events.csv"):
    """
    Salva gli eventi predetti in un file CSV, basandosi sulla soglia data.

    Args:
        y_pred_proba (array): Array con le probabilità predette dal modello.
        timestamps (array): Timestamp associati alle predizioni.
        threshold (float): Soglia per considerare un evento predetto (default=0.5).
        output_csv (str): Nome del file CSV in cui salvare i risultati.
    """
    # Convertiamo le probabilità in valori binari (0 o 1)
    y_pred_binary = (y_pred_proba > threshold).astype(int)

    # Creiamo un DataFrame con timestamp e predizioni
    df = pd.DataFrame({"timestamp": timestamps, "predicted_label": y_pred_binary})

    # Convertiamo i timestamp in datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Troviamo gli intervalli degli eventi predetti
    predicted_intervals = []
    start_time = None

    for i in range(len(df)):
        if df["predicted_label"].iloc[i] == 1 and start_time is None:
            start_time = df["timestamp"].iloc[i]  # Inizio evento
        elif df["predicted_label"].iloc[i] == 0 and start_time is not None:
            end_time = df["timestamp"].iloc[i - 1]  # Fine evento
            predicted_intervals.append((start_time, end_time))
            start_time = None

    # Se l'ultimo evento è ancora aperto, chiudilo
    if start_time is not None:
        predicted_intervals.append((start_time, df["timestamp"].iloc[-1]))

    # Creiamo un DataFrame con gli intervalli predetti
    events_df = pd.DataFrame(predicted_intervals, columns=["start_time", "end_time"])

    # Salviamo in un file CSV
    events_df.to_csv(output_csv, index=False)

    print(f"✅ Eventi predetti salvati in {output_csv}")

def binary_labels_to_intervals(labels, timestamps):
    """
    Converte una lista di etichette binarie in una lista di intervalli temporali (start_time, end_time).
    """
    intervals = []
    start_time = None

    for i in range(len(labels)):
        if labels[i] == 1 and start_time is None:  # Inizio di un evento
            start_time = timestamps[i]
        elif labels[i] == 0 and start_time is not None:  # Fine di un evento
            end_time = timestamps[i - 1]
            intervals.append((start_time, end_time))
            start_time = None

    # Se l'ultimo evento è rimasto aperto
    if start_time is not None:
        intervals.append((start_time, timestamps[-1]))

    return intervals
def binary_labels_to_intervals(labels, timestamps, min_event_duration=5):
    """
    Converte una sequenza di etichette binarie in intervalli temporali, 
    ignorando eventi troppo brevi.
    
    Args:
        labels: Lista di etichette binarie (0 o 1).
        timestamps: Lista di timestamp associati alle etichette.
        min_event_duration: Numero minimo di minuti affinché un intervallo sia considerato un evento reale.

    Returns:
        Lista di tuple [(start_time, end_time), ...].
    """
    intervals = []
    start_time = None

    for i in range(len(labels)):
        if labels[i] == 1 and start_time is None:  # Inizio di un evento
            start_time = timestamps[i]
        elif labels[i] == 0 and start_time is not None:  # Fine dell'evento
            end_time = timestamps[i - 1]
            
            # **Scarta eventi troppo brevi per ridurre FDR**
            if (end_time - start_time).total_seconds() / 60 >= min_event_duration:
                intervals.append((start_time, end_time))
            
            start_time = None

    # Se l'ultimo evento è rimasto aperto
    if start_time is not None:
        end_time = timestamps[-1]
        if (end_time - start_time).total_seconds() / 60 >= min_event_duration:
            intervals.append((start_time, end_time))

    return intervals


def tune_threshold(y_pred_proba, timestamps, actual_intervals, thresholds=np.arange(0.1, 0.85, 0.05), output_csv="cnn-lstm_tuning_results.csv"):
    """
    Testa diversi valori di threshold per valutare l'impatto sulle metriche.

    Args:
        y_pred_proba: probabilità predette dal modello.
        timestamps: timestamp corrispondenti alle predizioni.
        actual_intervals: lista di tuple con gli intervalli reali [(start_time, end_time), ...].
        thresholds: valori di soglia da testare.
        output_csv: nome del file CSV in cui salvare i risultati.
    """
    results = []

    for threshold in thresholds:
        print(f"\n🔍 Testando threshold: {threshold:.2f}")

        # **Generiamo nuove etichette binarie con la soglia corrente**
        y_pred_binary = (y_pred_proba > threshold).astype(int)

        # **Creiamo intervalli predetti considerando una durata minima per ridurre FDR**
        predicted_intervals = binary_labels_to_intervals(y_pred_binary, timestamps, min_event_duration=10)

        # **Calcoliamo le metriche**
        TPR, FDR, FTA, avg_lead_time = calculate_metrics(actual_intervals, predicted_intervals)

        # **Salviamo i risultati**
        results.append({"Threshold": threshold, "TPR": TPR, "FDR": FDR, "FTA": FTA, "Avg Lead Time": avg_lead_time})

    # **Creiamo un DataFrame e salviamo in CSV**
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Risultati del tuning salvati in {output_csv}")




# Caricamento dei dati
base_path = 'data/final/'
X_infra, y_infra = load_and_process_data(base_path, 'INFRA', sequence_length=120, actual_intervals=actual_intervals)
X_seismic, y_seismic = load_and_process_data(base_path, 'SEISMIC', sequence_length=120, actual_intervals=actual_intervals)

# ✅ Aggiungi debug per controllare i dati caricati
print(f"🔍 Debug: Verifica dati caricati")
print(f"X_infra shape: {X_infra.shape}")  # Deve essere (num_samples, 120, 10)
print(f"X_seismic shape: {X_seismic.shape}")  # Non deve essere (0, ...)
print(f"y_infra shape: {y_infra.shape}")
print(f"y_seismic shape: {y_seismic.shape}")

# Assicuriamoci che X_infra e X_seismic abbiano la stessa dimensione
min_samples = min(len(X_infra), len(X_seismic))

X_infra = X_infra[:min_samples]
X_seismic = X_seismic[:min_samples]
y_infra = y_infra[:min_samples]  # Anche le etichette devono essere ridimensionate

# Divisione in training, validation e test
train_size = int(0.6 * min_samples)
val_size = int(0.15 * min_samples)

X_train = [X_infra[:train_size], X_seismic[:train_size]]
y_train = y_infra[:train_size]

X_val = [X_infra[train_size:train_size + val_size], X_seismic[train_size:train_size + val_size]]
y_val = y_infra[train_size:train_size + val_size]

X_test = [X_infra[train_size + val_size:], X_seismic[train_size + val_size:]]
y_test = y_infra[train_size + val_size:]

# Debug: Controllo distribuzione classi in y_train
print(f"🔍 Debug: Distribuzione delle classi in y_train")
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"📊 Class Distribution: {class_distribution}")

# Verifica che ci siano almeno due classi in y_train
if len(np.unique(y_train)) > 1:
    try:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {int(c): class_weights[i] for i, c in enumerate(np.unique(y_train))}
        print(f"✅ Class Weights calcolati correttamente: {class_weight_dict}")
    except Exception as e:
        print(f"❌ Errore nel calcolo dei pesi delle classi: {e}")
        class_weight_dict = None
else:
    print("⚠️ ATTENZIONE: Il dataset contiene solo una classe, quindi i pesi non verranno applicati.")
    class_weight_dict = None



# Addestramento del modello
model_save_path = 'cnn-lstm_polling_model.h5'
if os.path.exists(model_save_path):
    print("Caricamento del modello pre-addestrato...")
    model = tf.keras.models.load_model(model_save_path)
else:
    print("Creazione e addestramento del modello...")
    model = build_cnn_model()
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Generazione delle probabilità predette su X_test
y_pred_proba = model.predict(X_test).flatten()

# Definizione della soglia fissa
threshold = 0.1

# Inizializzazione degli intervalli predetti e del tempo iniziale
predicted_intervals = []
start_time = pd.Timestamp("2021-01-01")

# Loop attraverso le predizioni minuto per minuto per generare gli intervalli
for i, pred in enumerate(y_pred_proba):
    if pred > threshold:  # Evento rilevato
        if not predicted_intervals or start_time > predicted_intervals[-1][1]:
            # Se non ci sono intervalli o l'attuale evento non è consecutivo, creane uno nuovo
            predicted_intervals.append((start_time, start_time + pd.Timedelta(minutes=120)))
        else:
            # Se l'evento è consecutivo, estendi l'ultimo intervallo
            predicted_intervals[-1] = (predicted_intervals[-1][0], start_time + pd.Timedelta(minutes=120))
    
    # Incrementa il tempo di 1 minuto
    start_time += pd.Timedelta(minutes=1)

# Calcola le metriche usando gli intervalli reali e quelli predetti
TPR, FDR, FTA, avg_lead_time = calculate_metrics(actual_intervals, predicted_intervals)
timestamps = pd.date_range(start="2021-01-01", periods=len(y_pred_proba), freq="1min")
tune_threshold(y_pred_proba, timestamps, actual_intervals)
save_predicted_events(y_pred_proba, timestamps, threshold=0.8, output_csv="CNN-LSTM_predicted_events.csv")
# Output delle metriche
print(f"TPR: {TPR}, FDR: {FDR}, FTA: {FTA}, Average Lead Time: {avg_lead_time}")

# Salvataggio del modello
model.save(model_save_path)
print(f"Modello salvato in {model_save_path}")
