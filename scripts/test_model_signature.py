import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

# Set your remote MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-13-60-49-130.eu-north-1.compute.amazonaws.com:8000/")

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    client = MlflowClient()

    # Get latest model version
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None
    assert latest_version is not None, f"No model found in '{stage}' stage for '{model_name}'"

    try:
        # Load model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Dummy input
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())

        # Align input_df to model signature
        if model.metadata.get_input_schema() is not None:
            signature = model.metadata.get_input_schema()
            expected_columns = [col.name for col in signature.inputs]
            input_df = input_df.reindex(columns=expected_columns, fill_value=0.0)  # <-- float
            # Ensure all columns are float
            input_df = input_df.astype('float64')

        # Predict
        prediction = model.predict(input_df)

        # Validate shapes
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version {latest_version} successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
