from kedro.pipeline import Pipeline, node

from .nodes import inferance_data, treat_missing_inferance,inferance_data_split,lebel_encoding_inferance, inferance_prediction,final_report

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=inferance_data,
                inputs="weather_aus_raw",
                outputs="df_inferance",
                name="inferance_data_node",
            ),
            node(
                func=treat_missing_inferance,
                inputs="df_inferance",
                outputs="df1_treat_inferance_data",
                name="treat_missing_inferance_node",
            ),
            node(
                func=inferance_data_split,
                inputs="df1_treat_inferance_data",
                outputs=["X_inferance", "y_inferance"],
                name="inferance_data_split",
            ),
            node(
                func=lebel_encoding_inferance,
                inputs="X_inferance",
                outputs="df2_inferance",
                name="lebel_encoding_inferance_node",
            ),
            node(
                func=inferance_prediction,
                inputs=["logreg","df2_inferance"],
                outputs="y_pred_inferance",
                name="inferance_prediction",
            ),
            node(
                func=final_report,
                inputs=["df_inferance","y_pred_inferance"],
                outputs="result_Dataset",
                name="final_report_node",
            ),
        ]
    )