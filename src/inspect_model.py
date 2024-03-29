import xgboost as xgb
import matplotlib.pyplot as plt

def load_and_inspect_model(model_path):
    # Load the model
    model = xgb.Booster()
    model.load_model(model_path)

    # Print a summary of the first tree
    print("First tree:", model.get_dump()[0])

    # Plot feature importances
    print("\nPlotting feature importances...")
    xgb.plot_importance(model)
    plt.show()

    # Visualize the first tree
    print("\nVisualizing the first tree...")
    xgb.plot_tree(model, num_trees=0)
    plt.show()

if __name__ == "__main__":
    # Adjust the path to where your model is saved
    model_path = 'C:\\Users\\ali\\diabetes_risk_prediction\\notebooks\\Updated_model_xgboost.json'
    
    load_and_inspect_model(model_path)
