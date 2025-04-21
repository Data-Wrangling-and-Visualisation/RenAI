import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const ArtworkAnalyzer = () => {
  const { objectID } = useParams();
  const [analysisData, setAnalysisData] = useState(null);

  useEffect(() => {
    const processArtwork = async () => {

      await fetch(`/api/process/${objectID}`);
      
      const embedding = await fetch(`/processed/embeddings/${objectID}.npy`).then(res => res.arrayBuffer());
      const gradcam = await fetch(`/processed/gradcam/${objectID}.npy`).then(res => res.arrayBuffer());
      const attention = await fetch(`/processed/attention/${objectID}.npy`).then(res => res.arrayBuffer());
      
      setAnalysisData({
        embedding: new Float32Array(embedding),
        gradcam: new Uint8Array(gradcam),
        attention: new Uint8Array(attention)
      });
    };

    processArtwork();
  }, [objectID]);

  return (
    <div>
      {analysisData && (
        <ScatterChart width={600} height={400}>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" />
          <YAxis type="number" dataKey="y" />
          <Tooltip />
          <Scatter data={analysisData.embedding} fill="#8884d8" />
        </ScatterChart>
      )}
    </div>
  );
};
