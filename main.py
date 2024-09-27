from data_preprocessing import load_data, preprocess_data
from model import build_nn_model, build_mc_model
from train import train_model
from evaluation import adjust_weights, predict, mc_predict
from visualization import plot_results, plot_histogram

# Load and preprocess data
df_train, df_test = load_data("/content/sample_data/california_housing_train.csv", "/content/sample_data/california_housing_test.csv")
X_train_norm, y_train, X_test_norm, y_test = preprocess_data(df_train, df_test)

# Build and train standard NN model
model_NN = build_nn_model()
train_model(model_NN, X_train_norm, y_train)

# Predict using the standard NN model
adjust_weights(model_NN)
predictions = predict(model_NN, X_test_norm)
df_test["predictions"] = predictions

# Plot results
plot_results(df_test, predictions)

# Build and train MC model
model_MC = build_mc_model()
train_model(model_MC, X_train_norm, y_train)

# Monte Carlo predictions
adjust_weights(model_MC)
mc_predictions = mc_predict(model_MC, X_test_norm)

# Plot Monte Carlo predictions
plot_histogram(df_train, df_test, mc_predictions)
