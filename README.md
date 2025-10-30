# CS494 – Generative AI Course Homeworks

This repository contains the three major homework notebooks for Northwestern's CS494 course on Generative AI. Each notebook focuses on a different modelling paradigm, progressing from classical statistical language models to modern large language models and retrieval‑augmented variational autoencoders.

- `CS494_HW1_SLM_Gautham_Satyanarayana.ipynb` – Statistical language models (unigram and N‑gram variants) with smoothing and text generation.
- `CS494_HW2_LLM_Gautham_Satyanarayana.ipynb` – From-scratch transformer pretraining, LoRA fine-tuning, and QLoRA quantisation on story datasets.
- `CS494_HW3_RAG_VAE_Gautham_Satyanarayana.ipynb` – Conditional VAEs, FAISS-backed retrieval augmentation, and stabilisation tricks for training.

The notebooks are designed to run in Google Colab but may also be executed locally with a compatible Python environment.

## Getting Started

### Option 1: Google Colab (recommended)
- Upload the notebook of interest to your Google Drive (or open it directly in Colab from Drive).
- From the Colab `Runtime` menu, select `Change runtime type` and enable a GPU (T4 or A100 works well).
- Run the initial setup cells to mount Drive (`google.colab.drive`) and install any required libraries.
- Place any external data (see the **Datasets** section) in your Drive and update paths inside the notebook as needed.

### Option 2: Local execution
1. Ensure you have Python 3.9+ and a recent `pip` available. A CUDA-capable GPU is strongly recommended for HW2 and HW3.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install core dependencies (adapt as assignments evolve):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy matplotlib tqdm transformers datasets scikit-learn nltk faiss-cpu ipykernel
   ```
4. Register the environment as a Jupyter kernel (optional but convenient):
   ```bash
   python -m ipykernel install --user --name cs494-hw
   ```
5. Launch Jupyter Lab/Notebook and open the desired `.ipynb` file. Execute cells top-to-bottom, ensuring all outputs are captured before submission.

> **GPU note:** HW2 (transformer training) and HW3 (RegA-VAE) are compute intensive. When running locally without a GPU, expect significantly longer runtimes or consider adjusting model sizes/batch sizes where permitted in the instructions.

## Datasets & External Assets

- **HW1 – Shakespeare N-gram corpus:** Download `train.txt` and `dev.txt` from the course Blackboard/Canvas resources. Place them in an accessible Drive or local directory, then update the data path variables near the top of the notebook.
- **HW2 – TinyStories & ROCStories:** The notebook includes cells that download the datasets via `datasets.load_dataset`. Ensure you have an internet connection (Colab handles this automatically; local runs may require enabling network access).
- **HW3 – RegA-VAE data:** The notebook bundles helper functions to fetch the required datasets. No manual downloads are typically necessary, but you may cache data to Drive to avoid repeated downloads.
- **Model checkpoints:** Later sections of HW2 and HW3 may instruct you to save checkpoints (e.g., `./checkpoints/pretrained_model.pt`). Create the directories beforehand, or modify the save paths to point to a Drive-mounted location.

## Working Through the Notebooks

- Run the notebooks sequentially, executing every cell so that outputs are preserved for grading.
- Complete all TODO/blank sections. These typically include function implementations, training loops, and evaluation code.
- Keep markdown responses concise but informative—many prompts expect written reflections on the experiments.
- Save intermediate results (e.g., trained models, plots) when instructed. This makes reruns reproducible and minimises redundant training.
