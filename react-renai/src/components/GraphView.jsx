import React, { useEffect, useState, Suspense } from "react";

const EmbeddingsGraph = React.lazy(() => import("./EmbeddingsGraph"));

const GraphVisualization = ({ selectedImage }) => {
  const [data, setData] = useState({
    embeddings: null,
    gradcam: null,
    attention: null,
  });
  const [loading, setLoading] = useState(false);
  const [showGradcam, setShowGradcam] = useState(false);
  const [showAttention, setShowAttention] = useState(false);

  useEffect(() => {
    console.log("selectedImage:", selectedImage);
    if (!selectedImage) return;
    setLoading(true);
    Promise.all([
      fetch(
        `http://localhost:3000/api/embeddings?category=${selectedImage.category}&file=${encodeURIComponent(
          selectedImage.fileName
        )}`
      ).then((res) => res.json()),
      fetch(
        `http://localhost:3000/api/gradcam?category=${selectedImage.category}&file=${encodeURIComponent(
          selectedImage.fileName
        )}`
      ).then((res) => res.json()),
      fetch(
        `http://localhost:3000/api/attention?category=${selectedImage.category}&file=${encodeURIComponent(
          selectedImage.fileName
        )}`
      ).then((res) => res.json()),
    ])
      .then(([embeddingsData, gradcamData, attentionData]) => {
        setData({
          embeddings: embeddingsData,
          gradcam: gradcamData,
          attention: attentionData,
        });
        setLoading(false);
      })
      .catch((error) => {
        console.error("error loading:", error);
        setLoading(false);
      });
  }, [selectedImage]);

  if (!selectedImage) {
    return <p className="p-4 text-white"></p>;
  }

  if (loading || !data.embeddings) {
    return <p className="p-4 text-white">Loading...</p>;
  }

  return (
    <div
      className="min-h-screen p-4 text-white bg-cover bg-center"
      style={{
        backgroundImage:
          "url('https://i.ibb.co/PhkGp9S/starry-background.jpg')",
      }}
    >
      <div className="border border-white p-2 mb-6">
        <Suspense fallback={<div>Loading...</div>}>
          <EmbeddingsGraph
            data={data.embeddings}
            selectedImage={selectedImage}
            onNodeClick={(node) => {
              console.log("Клик по узлу:", node);
            }}
          />
        </Suspense>
      </div>

      <div className="mb-6">
        <button
          onClick={() => setShowGradcam(!showGradcam)}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 focus:outline-none"
        >
          {showGradcam ? "Скрыть GradCAM" : "See GradCAM"}
        </button>
        {showGradcam && (
          <div className="mt-4 bg-gray-800 p-4 rounded">
            <h3 className="text-xl font-semibold mb-2">GradCAM</h3>
            <pre className="overflow-x-auto">
              {JSON.stringify(data.gradcam, null, 2)}
            </pre>
          </div>
        )}
      </div>

      <div>
        <button
          onClick={() => setShowAttention(!showAttention)}
          className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 focus:outline-none"
        >
          {showAttention ? "Скрыть Attention" : "See Attention"}
        </button>
        {showAttention && (
          <div className="mt-4 bg-gray-800 p-4 rounded">
            <h3 className="text-xl font-semibold mb-2">Attention</h3>
            <pre className="overflow-x-auto">
              {JSON.stringify(data.attention, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default GraphVisualization;