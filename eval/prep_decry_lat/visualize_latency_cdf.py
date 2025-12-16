import sys
import numpy as np
import re
import os

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("To install matplotlib, run: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False


def parse_log_file(log_file_path):
    """Parse a performance log file to extract latency values."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file '{log_file_path}' not found.")
        return None, None
    
    try:
        latencies = []
        success_rate = None
        
        with open(log_file_path, 'r', errors='replace') as f:
            lines = f.readlines()
        
        for line in lines:
            if "Decryption Success Rate:" in line:
                match = re.search(r'Decryption Success Rate: (\d+\.\d+)%', line)
                if match:
                    success_rate = float(match.group(1))
        
        for line in lines:
            if ("preparation time:" in line or "decryption time:" in line) and "ms" in line:
                match = re.search(r'time: (\d+\.\d+) ms', line)
                if match:
                    latencies.append(float(match.group(1)))
        
        if not latencies:
            num_cloves = None
            for line in lines:
                if "Number of cloves:" in line:
                    match = re.search(r'Number of cloves: (\d+)', line)
                    if match:
                        num_cloves = int(match.group(1))
                        break
            
            p50 = p90 = p99 = None
            for line in lines:
                if "p50 latency:" in line:
                    match = re.search(r'p50 latency: (\d+\.\d+) ms', line)
                    if match:
                        p50 = float(match.group(1))
                if "p90 latency:" in line:
                    match = re.search(r'p90 latency: (\d+\.\d+) ms', line)
                    if match:
                        p90 = float(match.group(1))
                if "p99 latency:" in line:
                    match = re.search(r'p99 latency: (\d+\.\d+) ms', line)
                    if match:
                        p99 = float(match.group(1))
            
            # If we have percentiles and number of cloves, generate an approximate distribution
            if p50 and p90 and p99 and num_cloves:
                print("Using percentiles to approximate distribution...")
                # Create a distribution that matches the percentiles
                latencies = np.zeros(num_cloves)
                latencies[int(num_cloves * 0.5)] = p50
                latencies[int(num_cloves * 0.9)] = p90
                latencies[int(num_cloves * 0.99)] = p99
                
                # Interpolate the rest
                for i in range(num_cloves):
                    if i < int(num_cloves * 0.5):
                        latencies[i] = p50 * (i / (num_cloves * 0.5))
                    elif i < int(num_cloves * 0.9):
                        t = (i - int(num_cloves * 0.5)) / (int(num_cloves * 0.9) - int(num_cloves * 0.5))
                        latencies[i] = p50 + t * (p90 - p50)
                    elif i < int(num_cloves * 0.99):
                        t = (i - int(num_cloves * 0.9)) / (int(num_cloves * 0.99) - int(num_cloves * 0.9))
                        latencies[i] = p90 + t * (p99 - p90)
                    else:
                        t = (i - int(num_cloves * 0.99)) / (num_cloves - int(num_cloves * 0.99))
                        latencies[i] = p99 + t * (p99 * 0.2)  # Extrapolate a bit beyond p99
        
        return np.sort(latencies) if latencies else None, success_rate
    
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None, None


def plot_cdf(latencies, output_path, title, success_rate=None):
    """Plot the CDF of latency values."""
    if latencies is None or len(latencies) == 0:
        print("No latency data to plot.")
        return False
    
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot generate CDF plot: matplotlib is not available.")
        return False
    
    # Sort latencies and calculate CDF
    sorted_latencies = np.sort(latencies)
    p = np.linspace(0, 1, len(sorted_latencies))
    
    # Create the CDF plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_latencies, p, marker='.', linestyle='-', markersize=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Cumulative Probability')
    
    # Add success rate info to title if available
    if success_rate is not None:
        title = f"{title} (Success Rate: {success_rate:.2f}%)"
    
    plt.title(title)
    
    # Add lines for important percentiles
    p50_idx = int(len(sorted_latencies) * 0.5)
    p90_idx = int(len(sorted_latencies) * 0.9)
    p99_idx = int(len(sorted_latencies) * 0.99)
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=0.99, color='b', linestyle='--', alpha=0.5)
    
    plt.axvline(x=sorted_latencies[p50_idx], color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=sorted_latencies[p90_idx], color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=sorted_latencies[p99_idx], color='b', linestyle='--', alpha=0.5)
    
    plt.text(sorted_latencies[p50_idx] * 1.05, 0.51, f'P50: {sorted_latencies[p50_idx]:.2f} ms', color='r')
    plt.text(sorted_latencies[p90_idx] * 1.05, 0.91, f'P90: {sorted_latencies[p90_idx]:.2f} ms', color='g')
    plt.text(sorted_latencies[p99_idx] * 1.05, 0.995, f'P99: {sorted_latencies[p99_idx]:.2f} ms', color='b')
    
    plt.tight_layout()
    plt.xlim(0, 1.05)
    plt.savefig(output_path, dpi=300)
    print(f"CDF plot saved to {output_path}")
    
    # Display the plot if in an interactive environment
    plt.show()
    return True


def print_latency_summary(latencies, title, success_rate=None):
    """Print a text summary of the latency data."""
    print(f"\n===== {title} =====")
    if success_rate is not None:
        print(f"Success Rate: {success_rate:.2f}%")
    
    print(f"Number of data points: {len(latencies)}")
    print(f"Min latency: {min(latencies):.3f} ms")
    print(f"Max latency: {max(latencies):.3f} ms")
    print(f"Mean latency: {np.mean(latencies):.3f} ms")
    print(f"Median (P50) latency: {np.median(latencies):.3f} ms")
    
    # Calculate percentiles
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"P90 latency: {p90:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print(f"P99 latency: {p99:.3f} ms")
    print("=" * (len(title) + 12))


def plot_combined_cdf(encrypt_latencies, decrypt_latencies, output_path, success_rate=None):
    """Plot the CDF of both encryption and decryption latencies on the same graph."""
    if (encrypt_latencies is None or len(encrypt_latencies) == 0) and \
       (decrypt_latencies is None or len(decrypt_latencies) == 0):
        print("No latency data to plot.")
        return False
    
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot generate combined CDF plot: matplotlib is not available.")
        return False
    
    plt.figure(figsize=(12, 7))
    
    # Plot encryption latencies if available
    if encrypt_latencies is not None and len(encrypt_latencies) > 0:
        sorted_encrypt = np.sort(encrypt_latencies)
        p_encrypt = np.linspace(0, 1, len(sorted_encrypt))
        plt.plot(sorted_encrypt, p_encrypt, 'b-', linewidth=2, label='Encryption Latency')
        
        # Add lines for important percentiles
        p50_idx = int(len(sorted_encrypt) * 0.5)
        p90_idx = int(len(sorted_encrypt) * 0.9)
        
        plt.axvline(x=sorted_encrypt[p50_idx], color='b', linestyle='--', alpha=0.3)
        plt.axvline(x=sorted_encrypt[p90_idx], color='b', linestyle='--', alpha=0.3)
        
        # Add text annotations for percentiles
        plt.text(sorted_encrypt[p50_idx] * 1.05, 0.45, f'P50 Enc: {sorted_encrypt[p50_idx]:.2f} ms', color='b')
        plt.text(sorted_encrypt[p90_idx] * 1.05, 0.85, f'P90 Enc: {sorted_encrypt[p90_idx]:.2f} ms', color='b')
    
    # Plot decryption latencies if available
    if decrypt_latencies is not None and len(decrypt_latencies) > 0:
        sorted_decrypt = np.sort(decrypt_latencies)
        p_decrypt = np.linspace(0, 1, len(sorted_decrypt))
        plt.plot(sorted_decrypt, p_decrypt, 'r-', linewidth=2, label='Decryption Latency')
        
        # Add lines for important percentiles
        p50_idx = int(len(sorted_decrypt) * 0.5)
        p90_idx = int(len(sorted_decrypt) * 0.9)
        
        plt.axvline(x=sorted_decrypt[p50_idx], color='r', linestyle='--', alpha=0.3)
        plt.axvline(x=sorted_decrypt[p90_idx], color='r', linestyle='--', alpha=0.3)
        
        # Add text annotations for percentiles
        plt.text(sorted_decrypt[p50_idx] * 1.05, 0.5, f'P50 Dec: {sorted_decrypt[p50_idx]:.2f} ms', color='r')
        plt.text(sorted_decrypt[p90_idx] * 1.05, 0.9, f'P90 Dec: {sorted_decrypt[p90_idx]:.2f} ms', color='r')
    
    # Add horizontal lines for percentiles
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3)
    
    # Format the plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('Latency (ms)')
    plt.xlim(0, 5)
    plt.ylabel('Cumulative Probability')
    
    title = "Comparison of Encryption vs Decryption Latency CDF"
    if success_rate is not None:
        title += f" (Decryption Success Rate: {success_rate:.2f}%)"
    
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    print(f"Combined CDF plot saved to {output_path}")
    
    # Display the plot if in an interactive environment
    plt.show()
    return True


def print_comparison_analysis(encrypt_latencies, decrypt_latencies, success_rate=None):
    """Print a comparison analysis between encryption and decryption latencies."""
    if encrypt_latencies is None or decrypt_latencies is None:
        print("Cannot compare: One or both datasets are missing.")
        return
    
    print("\n===== Encryption vs Decryption Comparison =====")
    
    if success_rate is not None:
        print(f"Decryption Success Rate: {success_rate:.2f}%")
    
    # Calculate and print basic stats
    enc_min, enc_max = min(encrypt_latencies), max(encrypt_latencies)
    dec_min, dec_max = min(decrypt_latencies), max(decrypt_latencies)
    
    enc_mean, enc_median = np.mean(encrypt_latencies), np.median(encrypt_latencies)
    dec_mean, dec_median = np.mean(decrypt_latencies), np.median(decrypt_latencies)
    
    enc_p90, enc_p99 = np.percentile(encrypt_latencies, 90), np.percentile(encrypt_latencies, 99)
    dec_p90, dec_p99 = np.percentile(decrypt_latencies, 90), np.percentile(decrypt_latencies, 99)
    
    # Calculate percentage differences correctly
    min_diff_pct = ((enc_min / dec_min) - 1) * 100 if dec_min != 0 else float('inf')
    mean_diff_pct = ((enc_mean / dec_mean) - 1) * 100 if dec_mean != 0 else float('inf')
    median_diff_pct = ((enc_median / dec_median) - 1) * 100 if dec_median != 0 else float('inf')
    p90_diff_pct = ((enc_p90 / dec_p90) - 1) * 100 if dec_p90 != 0 else float('inf')
    p99_diff_pct = ((enc_p99 / dec_p99) - 1) * 100 if dec_p99 != 0 else float('inf')
    max_diff_pct = ((enc_max / dec_max) - 1) * 100 if dec_max != 0 else float('inf')
    
    print(f"Min Latency: Encryption={enc_min:.3f}ms, Decryption={dec_min:.3f}ms, Diff={enc_min-dec_min:.3f}ms ({min_diff_pct:.1f}%)")
    print(f"Mean Latency: Encryption={enc_mean:.3f}ms, Decryption={dec_mean:.3f}ms, Diff={enc_mean-dec_mean:.3f}ms ({mean_diff_pct:.1f}%)")
    print(f"Median (P50) Latency: Encryption={enc_median:.3f}ms, Decryption={dec_median:.3f}ms, Diff={enc_median-dec_median:.3f}ms ({median_diff_pct:.1f}%)")
    print(f"P90 Latency: Encryption={enc_p90:.3f}ms, Decryption={dec_p90:.3f}ms, Diff={enc_p90-dec_p90:.3f}ms ({p90_diff_pct:.1f}%)")
    print(f"P99 Latency: Encryption={enc_p99:.3f}ms, Decryption={dec_p99:.3f}ms, Diff={enc_p99-dec_p99:.3f}ms ({p99_diff_pct:.1f}%)")
    print(f"Max Latency: Encryption={enc_max:.3f}ms, Decryption={dec_max:.3f}ms, Diff={enc_max-dec_max:.3f}ms ({max_diff_pct:.1f}%)")
    
    # Determine which operation is faster on average
    if enc_median < dec_median:
        faster_pct = ((dec_median / enc_median) - 1) * 100
        print(f"\nEncryption is generally faster than decryption by {dec_median-enc_median:.3f}ms ({faster_pct:.1f}%) at median.")
    elif dec_median < enc_median:
        faster_pct = ((enc_median / dec_median) - 1) * 100
        print(f"\nDecryption is generally faster than encryption by {enc_median-dec_median:.3f}ms ({faster_pct:.1f}%) at median.")
    else:
        print("\nEncryption and decryption have identical median latencies.")
    
    # Analyze tail latencies
    if enc_p99 > dec_p99:
        tail_pct = ((enc_p99 / dec_p99) - 1) * 100
        print(f"Encryption has worse tail latencies (P99) by {enc_p99-dec_p99:.3f}ms ({tail_pct:.1f}%).")
    elif dec_p99 > enc_p99:
        tail_pct = ((dec_p99 / enc_p99) - 1) * 100
        print(f"Decryption has worse tail latencies (P99) by {dec_p99-enc_p99:.3f}ms ({tail_pct:.1f}%).")
    else:
        print("Tail latencies (P99) are identical for both operations.")
    
    print("=" * 45)


def main():
    # Define log file paths
    encrypt_log_file_path = "../../build/model_prepare_performance.log"
    decrypt_log_file_path = "../../build/model_decrypt_performance.log"
    
    # Define output paths for plots
    encrypt_output_path = "clove_prep_latency_cdf.png"
    decrypt_output_path = "clove_decrypt_latency_cdf.png"
    combined_output_path = "combined_latency_cdf.png"
    
    # Process encryption performance data
    print(f"Parsing encryption log file: {encrypt_log_file_path}")
    encrypt_latencies, _ = parse_log_file(encrypt_log_file_path)
    
    if encrypt_latencies is not None:
        print(f"Found {len(encrypt_latencies)} encryption latency data points.")
        print_latency_summary(encrypt_latencies, "Clove Preparation Latency")
        
        if MATPLOTLIB_AVAILABLE:
            if plot_cdf(encrypt_latencies, encrypt_output_path, "Clove Preparation Latency CDF"):
                print(f"Encryption CDF plot saved to {encrypt_output_path}")
            else:
                print("Warning: Failed to generate encryption CDF plot.")
    else:
        print("Failed to extract encryption latency data from the log file.")
    
    # Process decryption performance data
    print(f"\nParsing decryption log file: {decrypt_log_file_path}")
    decrypt_latencies, success_rate = parse_log_file(decrypt_log_file_path)
    
    # If normal parsing failed, try reading in binary mode and extracting regex matches
    if decrypt_latencies is None:
        print("Trying alternative binary parsing approach for decryption log...")
        try:
            with open(decrypt_log_file_path, 'rb') as f:
                content = f.read()
            
            # Convert binary to string, replacing invalid chars
            content_str = content.decode('utf-8', errors='replace')
            
            # Extract decryption times
            decrypt_latencies = []
            success_rate = None
            
            # Try to find success rate
            success_match = re.search(r'Decryption Success Rate: (\d+\.\d+)%', content_str)
            if success_match:
                success_rate = float(success_match.group(1))
            
            # Find all decryption times
            for match in re.finditer(r'decryption time: (\d+\.\d+) ms', content_str):
                decrypt_latencies.append(float(match.group(1)))
            
            if not decrypt_latencies:
                # Try a more general pattern
                for match in re.finditer(r'time: (\d+\.\d+) ms', content_str):
                    decrypt_latencies.append(float(match.group(1)))
            
            decrypt_latencies = np.sort(decrypt_latencies) if decrypt_latencies else None
            
        except Exception as e:
            print(f"Alternative parsing also failed: {e}")
    
    if decrypt_latencies is not None:
        print(f"Found {len(decrypt_latencies)} decryption latency data points.")
        print_latency_summary(decrypt_latencies, "Clove Decryption Latency", success_rate)
        
        if MATPLOTLIB_AVAILABLE:
            if plot_cdf(decrypt_latencies, decrypt_output_path, "Clove Decryption Latency CDF", success_rate):
                print(f"Decryption CDF plot saved to {decrypt_output_path}")
            else:
                print("Warning: Failed to generate decryption CDF plot.")
    else:
        print("Failed to extract decryption latency data from the log file.")
    
    # Create a combined plot if both datasets are available
    if encrypt_latencies is not None and decrypt_latencies is not None:
        if MATPLOTLIB_AVAILABLE:
            plot_combined_cdf(encrypt_latencies, decrypt_latencies, combined_output_path, success_rate)
        
        # Print comparative analysis
        print_comparison_analysis(encrypt_latencies, decrypt_latencies, success_rate)


if __name__ == "__main__":
    main() 