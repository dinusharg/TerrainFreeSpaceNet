import gradio as gr

from terrainfreespacenet.infer import predict

CHECKPOINT_PATH = "checkpoints/test_model.pt"


def run_inference(file):
    if file is None:
        return "Please upload a CSV file."

    result = predict(
        csv_path=file.name,
        checkpoint_path=CHECKPOINT_PATH,
    )

    score = result["free_space_score"]
    return f"Predicted free-space score: {score:.4f}"


demo = gr.Interface(
    fn=run_inference,
    inputs=gr.File(label="Upload point cloud CSV"),
    outputs=gr.Textbox(label="Prediction"),
    title="TerrainFreeSpaceNet",
    description="Predict terrain free-space score from a 3D point-cloud CSV.",
)

if __name__ == "__main__":
    demo.launch()