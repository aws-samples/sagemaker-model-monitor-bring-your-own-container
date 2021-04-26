import numpy as np


def plot_drift_score(drift_df, top_k=5):    
    """Plot normalized feature drifts over time"""
    
    __plot_top_k_drifts(drift_df, feature='drift_score', ylabel='Feature Drift Score (%)', top_k=top_k)


def plot_p_values(drift_df, top_k=5):
    """Plot p-values of feature drifts over time"""

    df = drift_df.copy()
    df['inv_log_p_value'] = -np.log(df['p-value'])

    __plot_top_k_drifts(df, feature='inv_log_p_value', ylabel='Inverse log p-value', top_k=top_k)


def plot_accuracy(drift_df):
    """Plot accuracy projection over time"""
    
    df = drift_df[['Time', 'accuracy']].drop_duplicates(subset=['Time']).copy()

    ax = df.plot(x='Time', y='accuracy')  
    ax.set_xlabel('Time')
    ax.set_ylabel('Inverse log p-value')
    

def __plot_top_k_drifts(drift_df, feature, ylabel, top_k):
    df = drift_df.copy()
    
    feature_score = [(feat, max(df[df.Feature == feat][feature]))
                     for feat in df.Feature.value_counts().keys()]
    
    # Sort by drift score to pick top k features
    feature_score = [x[0] for x in sorted(feature_score, key=lambda x: x[1], reverse=True)][:top_k]
    
    df = df[['Feature', feature, 'Time']]
    df = df[df.Feature.isin(feature_score)]
    df = df.pivot(index='Time', columns='Feature', values=feature)

    ax = df.plot()
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
