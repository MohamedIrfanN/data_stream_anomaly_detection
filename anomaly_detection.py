import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
window_size = 50      # Moving window size for calculating mean and std deviation
threshold_factor = 3  # Number of std deviations from mean for anomaly threshold
stream_length = 500   # Total number of points in data stream

def simulate_data_stream():
    """Simulates a data stream with a seasonal pattern, noise, and random anomalies."""
    t = np.arange(stream_length)
    # Seasonality with noise
    data = 10 + 5 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 1, stream_length)
    # Inject random anomalies
    anomaly_indices = np.random.choice(stream_length, size=10, replace=False)
    data[anomaly_indices] += np.random.normal(15, 5, size=10)  # Anomalies added as large spikes
    return data

def detect_anomalies(data_stream):
    """Detects anomalies in a continuous data stream using moving average and std deviation."""
    anomalies = []
    mean_vals = []
    std_devs = []
    
    # Initialize window for moving statistics
    window = []

    for i, point in enumerate(data_stream):
        window.append(point)
        
        # Keep window size constant
        if len(window) > window_size:
            window.pop(0)

        # Calculate mean and std deviation over the window
        mean = np.mean(window)
        std_dev = np.std(window)

        # Store mean and std deviation for visualization
        mean_vals.append(mean)
        std_devs.append(std_dev)

        # Check if point is outside the threshold
        if abs(point - mean) > threshold_factor * std_dev:
            anomalies.append(i)  # Store index of anomaly

        # Real-time simulation delay (optional)
        time.sleep(0.05)
    
    return anomalies, mean_vals, std_devs

def visualize_data(data_stream, anomalies, mean_vals, std_devs):
    """Visualizes the data stream, anomalies, and moving statistics."""
    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label="Data Stream")
    plt.plot(mean_vals, label="Moving Mean", linestyle="--")
    plt.fill_between(range(len(data_stream)),
                     np.array(mean_vals) - threshold_factor * np.array(std_devs),
                     np.array(mean_vals) + threshold_factor * np.array(std_devs),
                     color='gray', alpha=0.2, label="Anomaly Threshold")
    
    # Plot anomalies
    plt.scatter(anomalies, data_stream[anomalies], color='red', label="Anomalies", marker='x')
    
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Real-Time Anomaly Detection in Data Stream")
    plt.legend()
    plt.show()

def main():
    # Step 1: Simulate Data Stream
    data_stream = simulate_data_stream()
    
    # Step 2: Detect Anomalies in Real-Time
    anomalies, mean_vals, std_devs = detect_anomalies(data_stream)
    
    # Step 3: Visualize the Results
    visualize_data(data_stream, anomalies, mean_vals, std_devs)

if __name__ == "__main__":
    main()
