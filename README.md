# R.E.N.A.I - Research & Exploration Network for Art Interpretation

## Overview

R.E.N.A.I is a full-stack web application designed for the interactive exploration and analysis of artworks through the lens of artificial intelligence. The platform processes artwork images from the Metropolitan Museum of Art collection and user-uploaded images, generating embeddings, visualizations, and detailed analysis of style, color, and composition. It provides researchers and enthusiasts with an intuitive interface to investigate computational interpretations of art, facilitating the study of visual similarity, model interpretability, and stylistic features.

## ✨ Features

* **Upload and Processing**:
  * Upload your own images for analysis and comparison with museum artworks
  * Process external images via URL
  * All uploaded images are analyzed for style, color, and composition
  * Uploaded images are included in embedding space and similarity analyses

* **Interactive Embedding Visualization:**
  * Dimensionality reduction techniques (t-SNE, UMAP) for visualizing high-dimensional image embeddings in 2D/3D space ([`EmbeddingProjection`](react-renai/src/components/EmbeddingProjection.jsx))
  * Network graph visualization of artwork similarity based on embedding proximity ([`ArtSimilarityGraph`](react-renai/src/components/ArtSimilarityGraph.jsx))
  * Heatmap representation of similarities between artworks ([`SimilarityHeatmap`](react-renai/src/components/SimilarityHeatmap.jsx))

* **Model Interpretability Visualization:**
  * Gradient-weighted Class Activation Mapping (Grad-CAM) using ResNet50 to highlight influential image regions ([`AttentionVisualizer`](react-renai/src/components/AttentionVisualizer.jsx))
  * Vision Transformer (ViT) attention maps to visualize the model's focus areas ([`AttentionVisualizer`](react-renai/src/components/AttentionVisualizer.jsx))
  * Comparative and overlay modes for analyzing saliency maps against original artworks

* **Artwork Feature Analysis:**
  * Detailed stylistic analysis with quantitative measurements of various artistic parameters ([`StyleAnalysis`](react-renai/src/components/StyleAnalysis.jsx))
  * Color palette extraction and visualization showing dominant colors and their proportions
  * Compositional analysis identifying structural elements, balance, and complexity
  * Interactive visualization of analysis results using radar charts, doughnut charts, and interactive sliders

* **User Interface:**
  * Responsive navigation sidebar with virtualized list rendering for smooth browsing of large artwork collections ([`Sidebar`](react-renai/src/components/Sidebar.jsx))
  * Global image panel for easy navigation between different analysis types ([`GlobalImagePanel`](react-renai/src/components/GlobalImagePanel.jsx))
  * Session persistence with browser storage to maintain state between visits
  * Modern interface built with Material UI and React

## 🛠️ Technology Stack

* **Frontend:** React (v18+), Material UI, Recharts, Nivo, Chart.js, react-force-graph-3d, react-window (virtualization)
* **Backend:** Python 3 (3.10 or 3.11 Recommended), Flask
* **Machine Learning & Data Processing:** PyTorch, TensorFlow, Timm, Transformers, NumPy, OpenCV, Pillow
* **External APIs:** Metropolitan Museum of Art Collection API
* **Containerization (Optional):** Docker, Docker Compose

## 🏛️ Architecture

The application follows a client-server architecture:

1. **React Frontend (`react-renai`):** Renders the user interface, manages client-side state, and sends API requests to the backend.
2. **Flask Backend (`server`):** Handles API requests, processes artwork data, orchestrates the execution of machine learning models, manages data storage, and serves processed results.
3. **Processing Modules:**
   * `embeddings.py`: Generates image embeddings using EfficientNetB0/DINOv2
   * `Gradcam.py`: Produces Grad-CAM visualizations using ResNet50
   * `hitmaps.py`: Creates attention maps using Vision Transformer
   * `improved_analysis.py`: Performs enhanced analysis of style, color, and composition
4. **Data Storage (`server/data`):** The backend saves generated files (`.npy` for embeddings and maps, `.json` for metadata) to avoid redundant computations.
5. **Metropolitan Museum of Art API:** External source for artwork metadata and original images.

## 🚀 Installation and Execution

**Prerequisites:**

* Git
* Python (3.10 or 3.11 strongly recommended for ML library compatibility) & `pip`
* Node.js (v16+) & `npm`

**Steps:**

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd RenAI
   ```

1.1. **Run using bash: (Optional)**
   ```
   chmod +x start_renai.sh
   ./run_start.sh
   ```

2. **Set up Python Backend (`server`):**
   * Navigate to the root directory (`RenAI`).
   * **Create a virtual environment (Recommended):**
     * Windows (using Python 3.11 example):
       ```powershell
       # Ensure you use your Python 3.10/3.11 executable path
       C:\Path\To\Python311\python.exe -m venv venv
       ```
     * Linux/macOS:
       ```bash
       python3 -m venv venv 
       # Or specify python3.10/python3.11 if needed
       # python3.11 -m venv venv 
       ```
   * **Activate the virtual environment:**
     * Windows (PowerShell):
       ```powershell
       # May require execution policy change for the session:
       # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
       .\venv\Scripts\activate
       ```
     * Linux/macOS:
       ```bash
       source venv/bin/activate
       ```
     *(Ensure `(venv)` appears in your terminal prompt)*
   * **Install Python dependencies:**
   ```bash
     # Ensure venv is active
     pip install -r requirements.txt
     ```

3. **Set up React Frontend (`react-renai`):**
   * Navigate to the frontend directory:
     ```bash
     cd react-renai
     ```
   * **Install Node.js dependencies:**
     ```bash
     npm install
     # If encountering peer dependency issues, try:
     # npm install --legacy-peer-deps
     ```

4. **Run the Application:**
   * **Start the Backend Server:**
     * Open a terminal in the **root `RenAI` directory**.
     * **Activate the virtual environment** (e.g., `source venv/bin/activate`).
     * Run the Flask server:
       * Windows (PowerShell):
         ```powershell
         $env:FLASK_APP="server/api.py"; $env:TF_CPP_MIN_LOG_LEVEL=2; flask run --port 5000
         ```
       * Linux/macOS:
   ```bash
         export FLASK_APP=server/api.py
         export TF_CPP_MIN_LOG_LEVEL=2 # Suppress TensorFlow info messages
         flask run --port 5000
         ```
     * Keep this terminal running.
   * **Start the Frontend Server:**
     * Open **another** terminal.
     * Navigate to the `react-renai` directory (`cd react-renai`).
     * Run the development server:
   ```bash
   npm run dev
   ```
     * Keep this terminal running.

5. **Access the Application:** Open your web browser and navigate to the URL provided by the development server (typically `http://localhost:5173`).

## Backend API Endpoints (`server/api.py`)

The Flask backend provides the following primary endpoints:

* **Metadata and Initialization:**
  * `GET /api/museum_artworks`: Retrieves a paginated list of artworks with metadata and embeddings from the Met collection.
  * `GET /api/objects`: Retrieves the complete list of object IDs from the Met API.
  * `GET /api/embeddings`: Loads all processed embeddings and basic metadata from `server/data/embeddings/`.
  * `GET /api/health`: Basic health check endpoint to verify the server is running.
  * `GET /api/diagnostics`: Returns detailed information about the backend environment and loaded models.

* **Artwork Processing:**
  * `GET /api/process/<int:object_id>`: Processes a specific artwork by ID, generating embeddings and maps.
  * `POST /api/process_by_url`: Processes an artwork from an external URL.
  * `POST /api/upload_and_analyze`: Uploads and processes a user image, generating embeddings, maps, and analysis in a single request.
  * `GET /api/proxy_image`: Proxies external image requests to avoid CORS issues.

* **Visualization Maps:**
  * `GET /api/gradcam_image/<string:object_id>`: Returns the processed Grad-CAM visualization as an image.
  * `GET /api/attention_image/<string:object_id>`: Returns the processed attention map visualization as an image.

* **Analysis Endpoints:**
  * `GET /api/analyze/style/<string:object_id>`: Returns detailed style analysis data.
  * `GET /api/analyze/color/<string:object_id>`: Returns color analysis with dominant colors and proportions.
  * `GET /api/analyze/composition/<string:object_id>`: Returns compositional analysis data.

## 🐳 Docker (Optional)

Docker configuration (`Dockerfile.frontend`, `Dockerfile.backend`, `docker-compose.yml`) is provided but may require modification:

* The Python environment needs to be correctly specified within `Dockerfile.backend` or managed via a suitable base image.
* Ensure build contexts and volume mounts are correctly configured in `docker-compose.yml`.

### Pre-built Docker Images

Pre-built Docker images are available on Docker Hub:

* Frontend: `fllarp/renai:frontend`
* Backend: `fllarp/renai:backend`

#### Quick Start with Docker

```bash
# Run backend
docker run -d --name renai-backend -p 5000:5000 fllarp/renai:backend

# Run frontend (after backend is running)
docker run -d --name renai-frontend -p 5173:5173 fllarp/renai:frontend

# Application is available at: http://localhost:5173
```

To stop:
```bash
docker stop renai-frontend renai-backend
docker rm renai-frontend renai-backend
```

#### Using docker-compose

You can also run the application with docker-compose:

```bash
# Create docker-compose.yml with:
version: '3'
services:
  backend:
    image: fllarp/renai:backend
    ports:
      - "5000:5000"
  
  frontend:
    image: fllarp/renai:frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend

# Then run:
docker-compose up -d
```

To stop: `docker-compose down`

Alternatively, build and run locally using: `docker-compose up --build -d`.

## 📝 Important Considerations

* **Models Used:** The primary models are TensorFlow/EfficientNetB0 (for embeddings), PyTorch/ResNet50 (for Grad-CAM), and PyTorch/Timm-ViT (for attention maps). Custom models for style, color, and composition analysis are included in `improved_analysis.py`.

* **Data Caching:**
  * Processed embeddings and maps are saved as `.npy` files in the `server/data/` directory.
  * Image caching for frontend has been disabled by default to reduce browser storage consumption.
  * The application utilizes session storage to maintain state between refreshes.

* **User-Uploaded Images:**
  * User-uploaded images are processed immediately upon upload, generating embeddings and analysis.
  * Uploaded image embeddings are included in all visualizations (projections, similarity graphs, heatmaps).
  * Note: Currently, attention and Grad-CAM maps may not generate properly for uploaded images since the original file isn't permanently stored on the server.

* **Performance Considerations:**
  * The sidebar implements virtualization (via `react-window`) to handle large artwork collections without performance degradation.
  * Initial load fetches a limited number of artworks to improve startup time.
  * The application uses "lazy loading" for analysis data, fetching it only when an artwork is selected.

* **Development vs. Production:**
  * The current setup uses the Flask development server and processes artworks on demand.
  * For production or handling many users, consider:
    * Using a production-grade WSGI server (e.g., Gunicorn, Waitress)
    * Implementing more robust caching (e.g., Redis, Memcached)
    * Pre-processing artworks in batches
    * Optimizing model loading and inference

* **Met API Dependency:** The application relies on the Metropolitan Museum of Art Collection API for museum artwork data. Be mindful of their API usage terms and potential rate limits.
