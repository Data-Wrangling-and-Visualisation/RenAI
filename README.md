# R.E.N.A.I - AI Art Vision

R.E.N.A.I (Research & Exploration Network for Art Interpretation) is an interactive full-stack web application designed for the visualization and analysis of data derived from the processing of artworks by neural networks. The application facilitates the exploration of embeddings, attention maps, and GradCAM outputs, providing an intuitive interface to investigate how artificial intelligence perceives and interprets art.

## ✨ Features

*   **Interactive Embedding Visualization:**
    *   2D and 3D projections using t-SNE and UMAP algorithms ([`EmbeddingProjection`](react-renai/src/components/EmbeddingProjection.jsx)).
    *   An interactive similarity graph of artworks utilizing `react-force-graph-3d` ([`ArtSimilarityGraph`](react-renai/src/components/ArtSimilarityGraph.jsx)).
    *   A similarity heatmap comparing groups of artworks (e.g., by style, epoch) implemented with `@nivo/heatmap` ([`SimilarityHeatmap`](react-renai/src/components/SimilarityHeatmap.jsx)).
*   **Model Attention Visualization:**
    *   Display of attention maps and GradCAM for analyzing the model's regions of interest ([`AttentionVisualizer`](react-renai/src/components/AttentionVisualizer.jsx)).
    *   Modes for comparing and overlaying maps onto the original artwork.
*   **Artistic Feature Analysis:**
    *   Visualization of stylistic profiles, color analysis, and compositional analysis using `chart.js` ([`StyleAnalysis`](react-renai/src/components/StyleAnalysis.jsx)).
*   **User-Friendly Interface:**
    *   A navigation sidebar ([`Sidebar`](react-renai/src/components/Sidebar.jsx)) for switching between different analysis modes.
    *   Modern design implemented with Material UI and Tailwind CSS.
*   **Data Processing Backend:**
    *   A Node.js/Express server ([`server.js`](react-renai/server.js)) to handle API requests.
    *   Invocation of Python scripts ([`convert_embeddings.py`](#convert_embeddings.py), `convert_gradcam.py`, [`convert_attention.py`](react-renai/convert_attention.py)) for loading and converting PyTorch data (`.pt`) to JSON format.
*   **Docker Configuration:**
    *   Pre-configured Dockerfiles ([`Dockerfile.frontend`](/Users/fllarpy/projects/DWV/renai/react-renai/Dockerfile.frontend), [`Dockerfile.backend`](react-renai/Dockerfile.backend)) and a `docker-compose.yml` file for streamlined deployment.

## 🛠️ Technology Stack

*   **Frontend:** React 19, Vite, Material UI, Tailwind CSS, Recharts, Nivo, Chart.js, react-force-graph-3d, Three.js, D3.js
*   **Backend:** Node.js, Express
*   **Data Processing:** Python 3, PyTorch
*   **Containerization:** Docker, Docker-compose

## 🐍 Data Preprocessing Scripts (`#processing-data`)

These scripts utilize various neural network models to extract features, embeddings, and saliency maps.

*   [`embeddings.py`](processing-data/embeddings.py): Uses the DINOv2 Vision Transformer (`dinov2_vitl14`) to generate image embeddings.
*   [`Gradcam.py`](processing-data/Gradcam.py): Generates GradCAM saliency maps using a pre-trained ResNet50 model. It also utilizes the CLIP model (`clip-vit-base-patch32`) to generate separate image embeddings (normalized).
*   [`hitmaps.py`](processing-data/hitmaps.py): Generates attention maps from the final block of a Vision Transformer model.


## Installation and Execution

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd RenAI/react-renai
    ```

2.  **Install frontend dependencies:**
    ```bash
    npm install --legacy-peer-deps
    ```
    *(The `--legacy-peer-deps` flag is used due to a version conflict between React 19 and `@nivo/heatmap`)*

3.  **Configure the backend:**
    *   Ensure **Python 3** and **pip** are installed.
    *   Install **PyTorch**: Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and select the installation command appropriate for your system (CPU or GPU). For example, for CPU:
        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
    *   Verify that your processed `.pt` files are located within the `react-renai/processed/` directory, organized into subdirectories corresponding to data categories (e.g., `processed/embeddings/`, `processed/gradcam/`). The conversion scripts ([`convert_embeddings.py`](#convert_embeddings.py), `convert_gradcam.py`, [`convert_attention.py`](react-renai/convert_attention.py)) expect files at `processed/<category>/<file_name>`.

4.  **Start the backend server:**
    ```bash
    node server.js
    ```
    The server will initiate on `http://localhost:3000`.

5.  **Start the frontend (in a separate terminal):**
    ```bash
    npm run dev
    ```
    The application will be accessible at `http://localhost:5173`.

### Docker Deployment

1.  Ensure **Docker** and **Docker Compose** are installed.
2.  **Install PyTorch (Important!):** The [`Dockerfile.backend`](react-renai/Dockerfile.backend) does **not** install PyTorch automatically. You must either add the `pip install torch ...` command to the [`Dockerfile.backend`](react-renai/Dockerfile.backend) or utilize a base Docker image that includes a pre-installed PyTorch environment.
3.  **Build and run the containers:**
    ```bash
    docker-compose up --build -d
    ```
4.  The application will be accessible at `http://localhost:5173`.
5.  **Stopping the containers:**
    ```bash
    docker-compose down
    ```

## Usage

1.  Access the application via your web browser (`http://localhost:5173` for local or Docker deployment).
2.  Utilize the sidebar ([`Sidebar`](react-renai/src/components/Sidebar.jsx)) to navigate through the various visualization sections:
    *   **Overview:** (Dashboard) - Initial landing page.
    *   **Embedding Analysis:** t-SNE/UMAP projections, clustering, nearest neighbor search.
    *   **Visual Attention:** Attention maps, GradCAM, comparison modes.
    *   **Artwork Similarity:** Heatmap, relationship graph.
    *   **Artistic Features:** Style, composition, and color analysis.
3.  Interact with the visualizations:
    *   Adjust projection parameters (t-SNE/UMAP).
    *   Scale, rotate, and pan within the 3D similarity graph.
    *   Hover over elements in the heatmap or projection plots for supplementary information.
    *   Select artworks, layers, and display modes within the attention visualizer.

## Backend API

The backend server ([`server.js`](#server.js)) exposes the following endpoints:

*   `GET /api/embeddings?category=<category>&file=<file_name>`: Loads and converts a `.pt` embedding file from `processed/<category>/<file_name>` to JSON using [`convert_embeddings.py`](#convert_embeddings.py).

## Python Scripts

*   [`convert_embeddings.py`](#convert_embeddings.py): Loads a PyTorch tensor or dictionary of tensors from a `.pt` file and outputs it in JSON format.


## License

This project is distributed under the MIT License. Refer to the `LICENSE` file for detailed information (Note: `LICENSE` file not provided, MIT assumed).
# RenAI