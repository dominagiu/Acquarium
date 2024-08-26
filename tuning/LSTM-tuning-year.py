import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os


output_directory = 'output_files'
os.makedirs(output_directory, exist_ok=True)

# Load seismic data and volcanic events
seismic_data_path = 'data/final_seismic_signals.pkl'
volcanic_events_path = 'csv_eruzioni/volcanic_events_numeric_controlled.csv'

time_series_df = pd.read_pickle(seismic_data_path)
volcanic_events_df = pd.read_csv(volcanic_events_path)

# Convert the 'Starting Time' and 'Ending Time' columns to datetime
volcanic_events_df['Starting Time'] = pd.to_datetime(volcanic_events_df['Starting Time'])
volcanic_events_df['Ending Time'] = pd.to_datetime(volcanic_events_df['Ending Time'])

# Define the list of years of interest
years_of_interest = range(2011, 2023)

# Set the fixed threshold
threshold = 0.1

# Function to calculate metrics
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
            lead_time = (pred_start - pd.to_datetime(overlapping_events[0, 0])).total_seconds() / 3600  # lead time in hours
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

# Loop through each year for individual model training
for year_of_interest in years_of_interest:
    print(f"\nProcessing Year: {year_of_interest}")

    # Filter the dataset to include only the events in the specified year
    volcanic_events_year_df = volcanic_events_df[
        (volcanic_events_df['Starting Time'].dt.year == year_of_interest) &
        (volcanic_events_df['Ending Time'].dt.year == year_of_interest)
    ]

    # Extract data for the specified year
    time_series_year_df = time_series_df[time_series_df.index.year == year_of_interest]

    # Create labels for the training year
    labels_train = pd.Series(0, index=time_series_year_df.index)
    for _, row in volcanic_events_year_df.iterrows():
        labels_train[(labels_train.index >= row['Starting Time']) & (labels_train.index <= row['Ending Time'])] = 1

    # Normalize the training data using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_train_data = scaler.fit_transform(time_series_year_df)

    # Convert back to DataFrame
    normalized_train_df = pd.DataFrame(normalized_train_data, index=time_series_year_df.index, columns=time_series_year_df.columns)

    # Reshape the data to be 3D [samples, timesteps, features] for LSTM
    X_train = np.expand_dims(normalized_train_df, axis=2)
    y_train = labels_train

    # Define the neural network model with LSTM layers
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the model
    model_save_path = os.path.join(output_directory, f'model_for_{year_of_interest}.h5')
    model.save(model_save_path)
    print(f"Model for year {year_of_interest} saved at {model_save_path}")

    # Initialize list to store metrics for this model across each validation year
    metrics_across_years = []

    # Loop through each validation year separately
    for validation_year in years_of_interest:
        if validation_year == year_of_interest:
            continue  

        print(f"Validating on Year: {validation_year}")

        # Extract validation data for the specific year
        validation_year_df = time_series_df[time_series_df.index.year == validation_year]

        # Create labels for the validation year
        labels_val = pd.Series(0, index=validation_year_df.index)
        for _, row in volcanic_events_df.iterrows():
            if row['Starting Time'].year == validation_year:
                labels_val[(labels_val.index >= row['Starting Time']) & (labels_val.index <= row['Ending Time'])] = 1

        # Normalize the validation year data using the same scaler
        normalized_val_data = scaler.transform(validation_year_df)

        # Convert back to DataFrame
        normalized_val_df = pd.DataFrame(normalized_val_data, index=validation_year_df.index, columns=validation_year_df.columns)

        # Reshape the data to be 3D [samples, timesteps, features] for LSTM
        X_val = np.expand_dims(normalized_val_df, axis=2)

        # Make predictions on the validation data
        predictions = model.predict(X_val)

        # Convert predictions to binary format using the threshold
        binary_predictions = (predictions >= threshold).astype(int)

        # Identify start and end times of predicted eruptions
        predicted_events = []
        eruption_ongoing = False
        for i in range(len(binary_predictions)):
            if binary_predictions[i] == 1 and not eruption_ongoing:
                start_time = validation_year_df.index[i]
                eruption_ongoing = True
            elif binary_predictions[i] == 0 and eruption_ongoing:
                end_time = validation_year_df.index[i-1]
                predicted_events.append((start_time, end_time))
                eruption_ongoing = False
        if eruption_ongoing:
            end_time = validation_year_df.index[-1]
            predicted_events.append((start_time, end_time))

        # Calculate metrics for this validation year
        actual_events = volcanic_events_df[['Starting Time', 'Ending Time']].values
        TPR, FDR, FTA, avg_lead_time = calculate_metrics(actual_events, predicted_events)
        metrics_across_years.append((year_of_interest, validation_year, TPR, FDR, FTA, avg_lead_time))


    metrics_df = pd.DataFrame(metrics_across_years, columns=['Train Year', 'Validation Year', 'TPR', 'FDR', 'FTA', 'Avg_Lead_Time'])
    print(metrics_df)

    # Save the metrics
    metrics_df.to_csv(os.path.join(output_directory, f'metrics_for_{year_of_interest}.csv'), index=False)

