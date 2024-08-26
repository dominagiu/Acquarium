import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K


# Funzione per calcolare la media escludendo valori 0 e lead time > 4 ore
def mean_excluding_zeros_and_large_lead_times(values):
    filtered_values = [v for v in values if v != 0 and v <= 4]
    if len(filtered_values) > 0:
        return np.mean(filtered_values)
    else:
        return 0

# Funzione per calcolare le metriche
def calculate_metrics(actual, predicted):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    lead_times = []
    correct_alert_time = 0  
    total_alert_time = 0  

    for pred_start, pred_end in predicted:
        pred_start, pred_end = pd.to_datetime(pred_start), pd.to_datetime(pred_end)
        total_alert_time += (pred_end - pred_start).total_seconds()
        
        overlapping_events = actual[
            (actual[:, 0] <= pred_end) & (actual[:, 1] >= pred_start)
        ]

        if len(overlapping_events) > 0:
            true_positives += 1
            lead_time = (pred_start - pd.to_datetime(overlapping_events[0, 0])).total_seconds() / 3600
            if lead_time <= 4:  
                lead_times.append(lead_time)
            
            # Calcola il tempo di allerta corretto
            for act_start, act_end in overlapping_events:
                overlap_start = max(pred_start, act_start)
                overlap_end = min(pred_end, act_end)
                overlap_duration = (overlap_end - overlap_start) / np.timedelta64(1, 's')
                correct_alert_time += overlap_duration
        else:
            false_positives += 1

    for act_start, act_end in actual:
        act_start, act_end = pd.to_datetime(act_start), pd.to_datetime(act_end)
        if not any((pd.to_datetime(pred_start) <= act_end) & (pd.to_datetime(pred_end) >= act_start) for pred_start, pred_end in predicted):
            false_negatives += 1

    TPR = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    FDR = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    FTA = (correct_alert_time / total_alert_time) if total_alert_time > 0 else 0
    average_lead_time = np.mean(lead_times) if lead_times else 0

    return TPR, FDR, FTA, average_lead_time

# Caricamento dei dati sismici e degli eventi vulcanici
seismic_data_path = 'data4giulio/final_seismic_signals.pkl'
volcanic_events_path = 'csv_eruzioni/volcanic_events_numeric_controlled.csv'

time_series_df = pd.read_pickle(seismic_data_path)
volcanic_events_df = pd.read_csv(volcanic_events_path)

# Conversione delle colonne 'Starting Time' e 'Ending Time' in formato datetime
volcanic_events_df['Starting Time'] = pd.to_datetime(volcanic_events_df['Starting Time'])
volcanic_events_df['Ending Time'] = pd.to_datetime(volcanic_events_df['Ending Time'])

# Filtraggio degli eventi per l'anno 2021
volcanic_events_2021_df = volcanic_events_df[
    (volcanic_events_df['Starting Time'].dt.year == 2021) &
    (volcanic_events_df['Ending Time'].dt.year == 2021)
]

# Estrazione dei dati sismici per il 2021
time_series_2021_df = time_series_df[time_series_df.index.year == 2021]

# Inizializzazione delle etichette (0 = nessuna eruzione) per tutti i timestamp del 2021
labels_2021 = pd.Series(0, index=time_series_2021_df.index)

# Assegnazione delle etichette di eruzione basate sugli eventi vulcanici
for _, row in volcanic_events_2021_df.iterrows():
    labels_2021[(labels_2021.index >= row['Starting Time']) & (labels_2021.index <= row['Ending Time'])] = 1

# Normalizzazione dei dati
scaler = MinMaxScaler()
normalized_data_2021 = scaler.fit_transform(time_series_2021_df)
normalized_df_2021 = pd.DataFrame(normalized_data_2021, index=time_series_2021_df.index, columns=time_series_2021_df.columns)

# Parametri configurabili
timesteps_range = [10, 20, 30, 40, 50]  # Numero di timesteps precedenti da considerare
future_steps_range = [1, 3, 6, 9, 12]  # Numero di future steps per fare la previsione

best_metrics = {'TPR': -1, 'FDR': float('inf'), 'FTA': -1, 'Lead Time': float('inf')}
best_params = {'timesteps': None, 'future_steps': None, 'threshold': None}

# Nome del file CSV per i progressi e del file di log
metrics_file = 'model_metrics_progress.csv'
log_file = 'processed_models.log'
checkpoint_dir = 'checkpoints'

# Creazione della directory per i checkpoint
os.makedirs(checkpoint_dir, exist_ok=True)

# Caricamento dei progressi esistenti e del file di log
if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    print("Progressi caricati dal file CSV esistente.")
else:
    metrics_df = pd.DataFrame(columns=['Timesteps', 'Future Steps', 'Threshold', 'TPR', 'FDR', 'FTA', 'Lead Time'])

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed_combinations = set(tuple(line.strip().split(',')) for line in f)
else:
    processed_combinations = set()

# Griglia di ricerca per tuning
for timesteps in timesteps_range:
    for future_steps in future_steps_range:
        # Verifica se la combinazione di timesteps e future_steps è già stata processata
        if any((str(timesteps), str(future_steps), str(threshold)) in processed_combinations for threshold in np.arange(0.1, 0.9, 0.1)):
            print(f"Combinazione Timesteps={timesteps}, Future Steps={future_steps} già processata, saltando...")
            continue

        # Preparazione dei dati per LSTM [samples, timesteps, features]
        X, y = [], []
        for i in range(timesteps, len(normalized_df_2021) - future_steps):
            X.append(normalized_df_2021.iloc[i-timesteps:i].values)
            y.append(labels_2021.iloc[i + future_steps])
        X, y = np.array(X), np.array(y)

        # Suddivisione dei dati in training e validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Resetta la sessione
        K.clear_session()

        # Definizione del modello LSTM
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Definizione del percorso per salvare i checkpoint
        checkpoint_filepath = os.path.join(checkpoint_dir, 'model_timesteps_{}_futuresteps_{}.h5'.format(timesteps, future_steps))

        # Definizione del callback per salvare il modello
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        # Addestramento del modello
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])
        print(f"allenamento Timesteps={timesteps}, Future Steps={future_steps}")

        # Caricamento del miglior modello salvato
        model = load_model(checkpoint_filepath)

        # Salvare le predizioni per l'anno di test (2021)
        predictions_2021 = model.predict(X)

        # Testare il modello su tutti i threshold previsti
        for threshold in np.arange(0.1, 1, 0.1):
            combination_key = (str(timesteps), str(future_steps), str(threshold))
            if combination_key in processed_combinations:
                print(f"Combinazione Timesteps={timesteps}, Future Steps={future_steps}, Threshold={threshold} già processata, saltando...")
                continue
            # Crea predizioni binarie basate sulla soglia
            binary_predictions = (predictions_2021 >= threshold).astype(int)

            # Identifica i tempi di inizio e fine delle eruzioni previste
            start_times = []
            end_times = []
            eruption_ongoing = False

            for i in range(len(predictions_2021)):
                if binary_predictions[i] == 1 and not eruption_ongoing:
                    start_times.append(labels_2021.index[i])
                    eruption_ongoing = True
                elif binary_predictions[i] == 0 and eruption_ongoing:
                    end_times.append(labels_2021.index[i-1])
                    eruption_ongoing = False

            if eruption_ongoing:
                end_times.append(labels_2021.index[-1])

            predicted_eruption_times = list(zip(start_times, end_times))

            # Calcola le metriche per ogni anno e soglia corrente
            TPRs, FDRs, FTAs, Lead_Times = [], [], [], []
            for year in range(2011, 2023):
                if year == 2021:
                    continue

                # Estrai i dati per l'anno corrente
                time_series_year_df = time_series_df[time_series_df.index.year == year]

                # Inizializza le etichette con 0 (nessuna eruzione) per tutti i timestamp dell'anno corrente
                labels_year = pd.Series(0, index=time_series_year_df.index)

                # Filtra gli eventi eruttivi per l'anno corrente
                volcanic_events_year_df = volcanic_events_df[
                    (volcanic_events_df['Starting Time'].dt.year == year) &
                    (volcanic_events_df['Ending Time'].dt.year == year)
                ]

                # Assegna le etichette per i periodi di eruzione
                for _, row in volcanic_events_year_df.iterrows():
                    labels_year[(labels_year.index >= row['Starting Time']) & (labels_year.index <= row['Ending Time'])] = 1

                # Normalizza i dati usando lo stesso scaler addestrato sui dati del 2021
                normalized_data_year = scaler.transform(time_series_year_df)
                normalized_df_year = pd.DataFrame(normalized_data_year, index=time_series_year_df.index, columns=time_series_year_df.columns)

                # Reshape the data to 3D format for LSTM [samples, timesteps, features]
                X_year = []
                for i in range(timesteps, len(normalized_df_year) - future_steps):
                    X_year.append(normalized_df_year.iloc[i-timesteps:i].values)
                X_year = np.array(X_year)

                # Predici sulle serie temporali dell'anno corrente
                predictions = model.predict(X_year)

                # Assicurati che l'indice temporale corrisponda alla lunghezza delle predizioni
                correct_index = time_series_year_df.index[timesteps:timesteps + len(predictions)]

                # Converti le predizioni in un DataFrame con l'indice corretto
                predictions_df = pd.DataFrame(predictions, columns=['Predicted Eruptions'], index=correct_index)

                # Crea predizioni binarie basate sulla soglia
                binary_predictions = (predictions_df['Predicted Eruptions'] >= threshold).astype(int)

                # Identifica i tempi di inizio e fine delle eruzioni previste
                start_times = []
                end_times = []
                eruption_ongoing = False

                for i in range(len(predictions_df)):
                    if binary_predictions.iloc[i] == 1 and not eruption_ongoing:
                        start_times.append(predictions_df.index[i])
                        eruption_ongoing = True
                    elif binary_predictions.iloc[i] == 0 and eruption_ongoing:
                        end_times.append(predictions_df.index[i-1])
                        eruption_ongoing = False

                if eruption_ongoing:
                    end_times.append(predictions_df.index[-1])

                predicted_eruption_times = list(zip(start_times, end_times))

                # Calcola le metriche per l'anno corrente e per la soglia corrente
                actual_eruption_times = volcanic_events_year_df[['Starting Time', 'Ending Time']].values
                TPR, FDR, FTA, avg_lead_time = calculate_metrics(actual_eruption_times, predicted_eruption_times)

                # Accumula le metriche per questo anno
                TPRs.append(TPR)
                FDRs.append(FDR)
                FTAs.append(FTA)
                Lead_Times.append(avg_lead_time)

            # Calcola la media per questo threshold escludendo 0 e lead time > 4 ore
            mean_TPR = mean_excluding_zeros_and_large_lead_times(TPRs)
            mean_FDR = mean_excluding_zeros_and_large_lead_times(FDRs)
            mean_FTA = mean_excluding_zeros_and_large_lead_times(FTAs)
            mean_Lead_Time = mean_excluding_zeros_and_large_lead_times(Lead_Times)

            # Aggiungi le metriche al DataFrame
            new_row = pd.DataFrame({
                'Timesteps': [timesteps],
                'Future Steps': [future_steps],
                'Threshold': [threshold],
                'TPR': [mean_TPR],
                'FDR': [mean_FDR],
                'FTA': [mean_FTA],
                'Lead Time': [mean_Lead_Time]
            })
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

            # Salva i progressi attuali
            metrics_df.to_csv(metrics_file, index=False)

            # Aggiorna il file di log
            with open(log_file, 'a') as log_f:
                log_f.write(','.join(combination_key) + '\n')

            # Verifica se questa combinazione è migliore
            if (mean_TPR > best_metrics['TPR'] and mean_FDR < best_metrics['FDR'] and
                mean_Lead_Time < best_metrics['Lead Time'] and mean_FTA > best_metrics['FTA']):
                best_metrics['TPR'] = mean_TPR
                best_metrics['FDR'] = mean_FDR
                best_metrics['FTA'] = mean_FTA
                best_metrics['Lead Time'] = mean_Lead_Time
                best_params['timesteps'] = timesteps
                best_params['future_steps'] = future_steps
                best_params['threshold'] = threshold

# Salva tutte le metriche
metrics_df.to_csv('model_metrics_final.csv', index=False)

print("Migliori parametri trovati:")
print(f"Timesteps: {best_params['timesteps']}")
print(f"Future Steps: {best_params['future_steps']}")
print(f"Threshold: {best_params['threshold']}")
print(f"TPR: {best_metrics['TPR']}")
print(f"FDR: {best_metrics['FDR']}")
print(f"FTA: {best_metrics['FTA']}")
print(f"Lead Time: {best_metrics['Lead Time']}")
