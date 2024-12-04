# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import HousingDataPreprocessor
from model_training import HousePriceModel
from visualization import HousingVisualizer

def main():
    # Load data
    df = pd.read_csv('housing.csv')
    
    # Initialize classes
    preprocessor = HousingDataPreprocessor()
    model = HousePriceModel()
    visualizer = HousingVisualizer()
    
    # Initial data visualization
    visualizer.plot_price_distribution(df)
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Preprocess data
    X_processed = preprocessor.preprocess_data(X)
    
    # Plot correlation matrix
    visualizer.plot_correlation_matrix(X_processed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    results = model.train_evaluate_models(X_train, X_test, y_train, y_test)
    
    predictions = model.get_predictions(X_test, y_test)

    # Print results
    print("\nModel Performance:")
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"Training RMSE: {metrics['train_rmse']:,.2f}")
        print(f"Testing RMSE: {metrics['test_rmse']:,.2f}")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"Cross-validation RMSE: {metrics['cv_rmse']:,.2f}")
    
    # Plot feature importance for random forest
    model.plot_feature_importance(X_processed)

    print("\nSample Predictions:")
    for name, pred_df in predictions.items():
        print(f"\n{name.upper()} Model:")
        print(pred_df.head())
        
        # Save predictions to CSV
        pred_df.to_csv(f'{name}_predictions.csv', index=False)
        
        # Plot actual vs predicted
        visualizer.plot_predictions(predictions, name)
        
        # Calculate prediction statistics
        mean_error = pred_df['Difference'].mean()
        mean_abs_error = pred_df['Difference'].abs().mean()
        
        print(f"\nPrediction Statistics for {name}:")
        print(f"Mean Error: {mean_error:,.2f}")
        print(f"Mean Absolute Error: {mean_abs_error:,.2f}")


if __name__ == "__main__":
    main()