import os
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras.backend as K
import gc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

labels_path = 'data/csv_eruzioni/etichette_2021.csv'
labels_df = pd.read_csv(labels_path)
labels_df['start_time'] = pd.to_datetime(labels_df['start_time'])
labels_df['end_time'] = pd.to_datetime(labels_df['end_time'])
actual_intervals = [(row['start_time'], row['end_time']) for _, row in labels_df.iterrows()]

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(base_path, category, sequence_length=120, scaler=None):
    """
    Carica e processa i dati da file .pkl per una specifica categoria (INFRA o SEISMIC).
    """
    print(f"\n🔍 Avvio caricamento dati per: {category}")
    
    # Percorso della cartella della categoria
    category_path = os.path.join(base_path, category)
    
    # Controllo se la cartella esiste e non è vuota
    if not os.path.exists(category_path) or not os.listdir(category_path):
        print(f"❌ ERRORE: La cartella {category_path} è vuota o non esiste!")
        return np.array([]), np.array([]), scaler

    # Selezione delle feature in base alla categoria
    if category.upper() == 'INFRA':
        selected_features = infra_features
    elif category.upper() == 'SEISMIC':
        selected_features = seismic_features
    else:
        raise ValueError(f"Categoria '{category}' non riconosciuta. Usa 'INFRA' o 'SEISMIC'.")

    # Inizializza MinMaxScaler se non è stato passato
    if scaler is None:
        scaler = MinMaxScaler()
    
    all_sequences, all_labels = [], []
    data_list = []

    print(f"📂 Stazioni disponibili in {category}: {os.listdir(category_path)}")

    # Itera sulle stazioni disponibili
    for station in os.listdir(category_path):
        station_path = os.path.join(category_path, station)

        if not os.path.isdir(station_path):
            continue  # Salta i file che non sono directory
        
        print(f"\n📍 Stazione trovata: {station}")
        print(f"📄 File disponibili: {os.listdir(station_path)}")
        
        dfs = []
        
        for feature_file in os.listdir(station_path):
            if feature_file.endswith('_cleaned.pkl'):  
                feature_name = feature_file.replace('_cleaned.pkl', '')  # Estrai il nome corretto della feature
                file_path = os.path.join(station_path, feature_file)
                
                # Carica il file .pkl
                try:
                    data = pd.read_pickle(file_path)
                    print(f"✅ Caricato: {feature_file} | Shape: {data.shape}")

                    # Se il file è una Series, converti in DataFrame
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name=feature_name)

                    # Controlla se ha un timestamp, se non lo ha, prova a estrarlo
                    if 'timestamp' not in data.columns:
                        if isinstance(data.index, pd.DatetimeIndex):
                            data = data.reset_index()
                            data.rename(columns={data.columns[0]: 'timestamp'}, inplace=True)
                        else:
                            print(f"⚠️ File {feature_file} non ha timestamp valido. Skipping...")
                            continue

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

        # Seleziona solo le feature desiderate
        available_features = set(station_data.columns)
        selected_features_filtered = [feat for feat in selected_features if feat in available_features]

        if not selected_features_filtered:
            print(f"⚠️ Nessuna feature valida trovata per {station}. Skipping...")
            continue

        print(f"🔢 Feature disponibili: {selected_features_filtered}")

        # Seleziona solo le feature numeriche + timestamp
        station_data = station_data[['timestamp'] + selected_features_filtered]
        numeric_columns = station_data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            print(f"⚠️ Nessuna colonna numerica valida in {station}. Skipping...")
            continue

        data_list.append(station_data[numeric_columns])

    # Se nessun dato è stato caricato
    if not data_list:
        print(f"❌ Nessun dato valido trovato per {category}! Verifica la directory e le feature.")
        return np.array([]), np.array([]), scaler

    # Normalizzazione
    full_data = pd.concat(data_list)
    scaler.fit(full_data)  # Fit scaler su tutti i dati prima della trasformazione
    
    print(f"📊 Normalizzazione effettuata. Shape totale dati: {full_data.shape}")

    # Trasforma i dati con lo scaler
    for i in range(len(data_list)):
        data_list[i] = data_list[i].copy()
        data_list[i].loc[:, numeric_columns] = scaler.transform(data_list[i][numeric_columns])
    
    # Creazione delle sequenze temporali
    for station_data in data_list:
        for i in range(len(station_data) - sequence_length + 1):
            seq = station_data.iloc[i:i + sequence_length]
            all_sequences.append(seq.values)
            all_labels.append(1 if 'event' in seq.columns and seq['event'].iloc[-1] == 1 else 0)

    print(f"✅ Caricate {len(all_sequences)} sequenze da {category}.")
    
    return np.array(all_sequences, dtype=np.float16), np.array(all_labels, dtype=np.float16), scaler

def build_lstm_model(input_shape=(120, 15)):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
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

    return TPR, FDR, FTA, avg_lead_time
  
performance_results = []
model_save_path = "saved_models"

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

categories = ['INFRA']
X, y, feature_names = [], [], []
for category in categories:
    X_cat, y_cat, features_cat = load_and_process_data('data/', category, sequence_length=120)
    X.append(X_cat)
    y.append(y_cat)
    feature_names.extend(features_cat)
    
X = np.concatenate(X)
y = np.concatenate(y)

for feature in feature_names:
    model_save_file = os.path.join(model_save_path, f'model_feature_removed_{feature}.h5')
    feature_index = feature_names.index(feature)
    X_reduced = np.delete(X, feature_index, axis=2)  # Assicura che X_reduced sia sempre definito
    
    if os.path.exists(model_save_file):  
        print(f"Modello per {feature} già esistente. Caricamento...")
        model = tf.keras.models.load_model(model_save_file)
    else:
        print(f"Allenando il modello rimuovendo la feature: {feature} ({feature_names.index(feature) + 1}/{len(feature_names)})")
        
        model = build_lstm_model(input_shape=(120, X_reduced.shape[2]))
        model.fit(X_reduced, y, epochs=5, batch_size=32, verbose=1)
        model.save(model_save_file)  
        print(f"Modello completato e salvato: {model_save_file}\n")
    
    # Pulizia della sessione prima di passare alle predizioni
    K.clear_session()
    gc.collect()

    # Converte X_reduced in tensor prima di passarlo a model.predict
    X_reduced = tf.convert_to_tensor(X_reduced, dtype=tf.float32)
    y_pred = model.predict(X_reduced, batch_size=16)

    predicted_intervals = []
    start_time = pd.Timestamp("2021-01-01")

    for i, pred in enumerate(y_pred):
        if pred > 0.1:
            if not predicted_intervals or start_time > predicted_intervals[-1][1]:
                predicted_intervals.append((start_time, start_time + pd.Timedelta(minutes=120)))
            else:
                predicted_intervals[-1] = (predicted_intervals[-1][0], start_time + pd.Timedelta(minutes=120))
        start_time += pd.Timedelta(minutes=1)

    TPR, FDR, FTA, avg_lead_time = calculate_metrics(actual_intervals, predicted_intervals)
    
    result = pd.DataFrame([{
        'Feature Rimossa': feature,
        'True Positive Rate': TPR,
        'False Discovery Rate': FDR,
        'FTA': FTA,
        'avg_lead_time': avg_lead_time
    }])

    result.to_csv("feature_selection_results.csv", mode='a', header=not os.path.exists("feature_selection_results.csv"), index=False)
