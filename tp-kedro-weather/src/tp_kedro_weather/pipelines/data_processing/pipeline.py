from kedro.pipeline import Node, Pipeline

from .nodes import load_weather_data, clean_weather_data, train_model, extract_model, extract_metrics, convert_metrics_to_json, create_metrics_visualization


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_weather_data,
                inputs="weather_data",
                outputs="loaded_weather_data",
                name="load_weather_data_node",
            ),
            Node(
                func=clean_weather_data,
                inputs="loaded_weather_data",
                outputs="donnees_nettoyees",
                name="clean_weather_data_node",
            ),
            Node(
                func=train_model,
                inputs="donnees_nettoyees",
                outputs="model_results",
                name="train_weather_model_node",
            ),
            Node(
                func=extract_model,
                inputs="model_results",
                outputs="Modele_entraine",
                name="save_model_node",
            ),
            Node(
                func=extract_metrics,
                inputs="model_results",
                outputs="metrics",
                name="save_metrics_node",
            ),
            Node(
                func=convert_metrics_to_json,
                inputs="metrics",
                outputs="metrics_json",
                name="convert_metrics_to_json_node",
            ),
            Node(
                func=create_metrics_visualization,
                inputs="metrics",
                outputs="metrics_visualization",
                name="create_metrics_visualization_node",
            ),
        ]
    )
