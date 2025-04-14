import React, { useState } from "react";
import Sidebar from "../components/Sidebar";
import GraphVisualization from "../components/GraphView";
import ThreeScene from "../components/StarBackground";

const GraphPage = () => {
  const [selectedImage, setSelectedImage] = useState(null);

  return (
    <div className="relative min-h-screen">
      {/* Задний фон — сцена three.js */}
      <ThreeScene />
      
      {/* Контейнер основного контента */}
      <div className="relative z-40 flex">
        {/* Sidebar остаётся поверх сцены.
            Его можно свернуть внутри самого компонента Sidebar */}
        <div className="w-1/4 border-r border-gray-700 bg-gray-800 bg-opacity-90">
          <Sidebar onSelectImage={(img) => setSelectedImage(img)} />
        </div>
        <div className="flex-1 overflow-y-auto relative z-30">
          <GraphVisualization selectedImage={selectedImage} />
        </div>
      </div>
    </div>
  );
};

export default GraphPage;