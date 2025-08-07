import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_csv(filepath):
    """
    Loads the CSV and returns a DataFrame with the average value per model,
    preserving the original order in which models appear.
    """
    df = pd.read_csv(filepath)
    
    # Capture model order from first appearance in the CSV
    model_order = df['model'].drop_duplicates()
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    
    # Group by model and compute the mean
    processed_df = df.groupby("model", sort=False).mean(numeric_only=True).reset_index()
    
    return processed_df


def plot_value(processed_df, value, exclude_models=None):
    """
    Plots a bar chart of the average value (e.g., 'rag_time') per model.
    
    Parameters:
    - processed_df: DataFrame with one row per model and averaged metrics
    - value: The column name to plot (e.g., 'sem_score')
    - exclude_models: List of model names to exclude from the plot
    """
    if value not in processed_df.columns:
        raise ValueError(f"'{value}' is not a valid column. Available options: {list(processed_df.columns)}")
    
    df_to_plot = processed_df.copy()
    
    if exclude_models:
        df_to_plot = df_to_plot[~df_to_plot['model'].isin(exclude_models)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_to_plot['model'], df_to_plot[value], color='skyblue')
    plt.title(f'Average {value} per Model')
    plt.xlabel('Model')
    plt.ylabel(f'Average {value}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_quality_scores(processed_df, exclude_models=None):
    """
    Plots separate bar charts for sem_score, overlap, and lang_score,
    each with independently scaled y-axes.
    
    Parameters:
    - processed_df: DataFrame with average values per model
    - exclude_models: Optional list of model names to exclude
    """
    df_to_plot = processed_df.copy()

    if exclude_models:
        df_to_plot = df_to_plot[~df_to_plot['model'].isin(exclude_models)]

    models = df_to_plot['model']
    sem_scores = df_to_plot['sem_score']
    overlap_scores = df_to_plot['overlap']
    lang_scores = df_to_plot['lang_score']

    x = np.arange(len(models))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].bar(models, sem_scores, color='skyblue')
    axes[0].set_ylabel('Semantic Score')
    axes[0].set_title('Semantic Similarity')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    axes[1].bar(models, overlap_scores, color='lightgreen')
    axes[1].set_ylabel('Overlap')
    axes[1].set_title('Token Overlap')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    axes[2].bar(models, lang_scores, color='salmon')
    axes[2].set_ylabel('Language Score')
    axes[2].set_title('Language Fluency')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].grid(axis='y', linestyle='--', alpha=0.6)

    plt.suptitle('Quality Metrics per Model', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_quality_scores_normalized(processed_df, exclude_models=None):
    """
    Plots a grouped bar chart of normalized sem_score, overlap, and lang_score per model.
    
    Parameters:
    - processed_df: DataFrame with average values per model
    - exclude_models: Optional list of model names to exclude
    """
    df_to_plot = processed_df.copy()

    if exclude_models:
        df_to_plot = df_to_plot[~df_to_plot['model'].isin(exclude_models)]

    # Normalize scores to [0, 1] for visual comparability
    df_to_plot['sem_score_norm'] = df_to_plot['sem_score'] / df_to_plot['sem_score'].max()
    df_to_plot['overlap_norm'] = df_to_plot['overlap'] / df_to_plot['overlap'].max()
    df_to_plot['lang_score_norm'] = df_to_plot['lang_score'] / df_to_plot['lang_score'].max()

    models = df_to_plot['model']
    x = np.arange(len(models))  # x positions
    width = 0.25  # width of each bar

    # Extract normalized scores
    sem_scores = df_to_plot['sem_score_norm']
    overlap_scores = df_to_plot['overlap_norm']
    lang_scores = df_to_plot['lang_score_norm']

    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, sem_scores, width, label='Semantic Score', color='skyblue')
    ax.bar(x, overlap_scores, width, label='Overlap', color='lightgreen')
    ax.bar(x + width, lang_scores, width, label='Language Score', color='salmon')

    # Labels and formatting
    ax.set_ylabel('Normalized Score (0–1)')
    ax.set_xlabel('Model')
    ax.set_title('Normalized Quality Metrics per Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_stacked_time(processed_df, exclude_models=None):
    """
    Plots a stacked bar chart of response_time and rag_time per model.
    rag_time is stacked on top of response_time.

    Parameters:
    - processed_df: DataFrame with average values per model
    - exclude_models: Optional list of model names to exclude
    """
    df_to_plot = processed_df.copy()

    if exclude_models:
        df_to_plot = df_to_plot[~df_to_plot['model'].isin(exclude_models)]

    # Check required columns
    for col in ['total_time', 'rag_time']:
        if col not in df_to_plot.columns:
            raise ValueError(f"Missing required column: {col}")

    models = df_to_plot['model']
    response_time = df_to_plot['response_time']
    prompt_eval_time = df_to_plot['prompt_eval_time']
    rag_time = df_to_plot['rag_time']

    plt.figure(figsize=(10, 6))
    plt.bar(models, response_time, label='Response Generation Time', color='#87ceeb')
    plt.bar(models, prompt_eval_time, bottom=response_time, label='Prompt Eval Time', color='#7398db')
    cumulative_bottom = response_time + prompt_eval_time
    plt.bar(models, rag_time, bottom=cumulative_bottom, label='RAG Time', color='#3333b2')


    plt.title('Stacked Bar Chart of Response Time and RAG Time')
    plt.xlabel('Model')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_distribution(processed_df, exclude_models=None):
    """
    Plots a pie chart showing the average time spent on RAG vs response generation.

    Parameters:
    - processed_df: DataFrame with columns ['model', 'rag_time', 'total_time']
    - exclude_models: Optional list of model names to exclude
    """
    df_to_plot = processed_df.copy()

    if exclude_models:
        df_to_plot = df_to_plot[~df_to_plot['model'].isin(exclude_models)]

    # Compute total average rag_time and response_time across all models
    avg_rag_time = df_to_plot['rag_time'].mean()
    avg_prompt_time = df_to_plot['prompt_eval_time'].mean()
    avg_response_time = df_to_plot['response_time'].mean()

    # Sanity check (avoid negative slice)
    avg_response_time = max(avg_response_time, 0)

    # Data for the pie chart
    labels = ['RAG Time', 'Prompt Eval Time', 'Response Generation Time']
    sizes = [avg_rag_time, avg_prompt_time, avg_response_time]
    colors = ['#87ceeb', '#7398db', '#3333b2']

    # Plotting
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=140, shadow=False)

    ax.set_title('Average Time Distribution (RAG vs Prompt Eval vs Response Generation)', fontsize=14)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.show()

def main():
    data = load_and_process_csv("plots/data/chatbot_comparison_results_final.csv")

    """plot_value(data, "total_time", exclude_models=["david_openai_control"])
    plot_value(data, "prompt_tokens", exclude_models=["david_openai_control"])
    plot_value(data, "prompt_eval_time", exclude_models=["david_openai_control"])
    plot_value(data, "response_tokens", exclude_models=["david_openai_control"])
    plot_value(data, "response_time", exclude_models=["david_openai_control"])

    plot_value(data, "sem_score")
    plot_value(data, "overlap")
    plot_value(data, "lang_score")

    plot_quality_scores_normalized(data)"""

    #plot_time_distribution(data, exclude_models=["deepseek-r1:8b", "david-8b4", "deepseek-r1:14b", "david-14b2", "david-openai1"])

    plot_stacked_time(data, exclude_models=["david_openai_control"])

if __name__ == "__main__":
    main()