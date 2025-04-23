from huggingface_hub import hf_hub_download


def download_facedet():
    hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt",
        local_dir="."
    )
