import os
import pandas as pd
import matplotlib.pyplot as plt

# Enlarge fonts and lines
plt.rcParams.update({
    'font.size': 14 * 1.65,
    'axes.labelsize': 14 * 1.65,
    'xtick.labelsize': 12 * 1.65,
    'ytick.labelsize': 12 * 1.65,
    'legend.fontsize': 12 * 1.65,
    'lines.linewidth': 4,
    'figure.figsize': (8, 6.2)
})

# Define a color palette for both series
palette = {
    'PUN': 'red',                  # muted blue
    'Frequent Pattern Mining': 'blue'  # safety orange
}

# Map folder names to CSV filenames
datasets = {
    'bj':    'bj.csv',
    'porto': 'porto.csv',
}

for folder, csvfile in datasets.items():
    # Create output folder
    os.makedirs(folder, exist_ok=True)

    # Load & clean columns
    df = pd.read_csv(csvfile)
    df.columns = df.columns.str.strip().str.lower()

    # Scenario 1: top_k_paths = 10
    df1 = df[df['top_k_paths'] == 10]

    # A) Comp Time vs num_paths
    fig, ax = plt.subplots()
    ax.plot(df1['num_paths'], df1['util_comp_time'], label='PUN', color=palette['PUN'])
    ax.plot(df1['num_paths'], df1['freq_comp_time'], label='Frequent Pattern Mining', color=palette['Frequent Pattern Mining'])
    ax.set_xlabel('Total Number Of Paths (Chi)')
    ax.set_ylabel('Time (s)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'timechi.png'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # B) Total Utility vs num_paths
    fig, ax = plt.subplots()
    ax.plot(df1['num_paths'], df1['total_high'], label='PUN', color=palette['PUN'])
    ax.plot(df1['num_paths'], df1['total_freq'], label='Frequent Pattern Mining', color=palette['Frequent Pattern Mining'])
    ax.set_xlabel('Total Number Of Paths (Chi)')
    ax.set_ylabel('Total Matching Score (TMS)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'tmschi.png'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # Scenario 2: num_paths = 100
    df2 = df[df['num_paths'] == 100]

    # C) Comp Time vs top_k_paths
    fig, ax = plt.subplots()
    ax.plot(df2['top_k_paths'], df2['util_comp_time'], label='PUN', color=palette['PUN'])
    ax.plot(df2['top_k_paths'], df2['freq_comp_time'], label='Frequent Pattern Mining', color=palette['Frequent Pattern Mining'])
    ax.set_xlabel('k')
    ax.set_ylabel('Time (s)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'timek.png'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # D) Total Utility vs top_k_paths
    fig, ax = plt.subplots()
    ax.plot(df2['top_k_paths'], df2['total_high'], label='PUN', color=palette['PUN'])
    ax.plot(df2['top_k_paths'], df2['total_freq'], label='Frequent Pattern Mining', color=palette['Frequent Pattern Mining'])
    ax.set_xlabel('k')
    ax.set_ylabel('Total Matching Score (TMS)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'tmsk.png'), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    print(f"Saved plots for {csvfile} in ./{folder}/")
