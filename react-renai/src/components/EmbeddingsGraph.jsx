import React, { useRef, useEffect } from "react";
import * as d3 from "d3";

const EmbeddingsGraph = ({ data, selectedImage, onNodeClick }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data) return;

    // Если данные уже имеют структуру nodes/links, используем их.
    // Иначе предполагаем, что data — это массив векторов (например, [[0.13, -1.17, ...], ...])
    let graphData;
    if (data.nodes && data.links) {
      graphData = data;
    } else if (Array.isArray(data)) {
      // Преобразуем каждый вектор в узел: используем первые две координаты как x и y.
      const nodes = data.map((embedding, i) => {
        const x = embedding[0] !== undefined ? embedding[0] * 50 + 400 : Math.random() * 800;
        const y = embedding[1] !== undefined ? embedding[1] * 50 + 300 : Math.random() * 600;
        return { id: i, x, y, embedding };
      });
      // Пока не рассчитываем связи; можно добавить логику для вычисления связей при необходимости.
      graphData = { nodes, links: [] };
    } else {
      return;
    }

    const width = 800, height = 600;
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .style("border", "1px solid white");

    // Очистим содержимое от предыдущего рендеринга.
    svg.selectAll("*").remove();

    const simulation = d3.forceSimulation(graphData.nodes)
      .force("charge", d3.forceManyBody().strength(-50))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(5))
      .on("tick", ticked);

    function ticked() {
      nodeElements
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    }

    const nodeElements = svg.selectAll("circle")
      .data(graphData.nodes)
      .enter()
      .append("circle")
      .attr("r", 5)
      .attr("fill", "lightblue")
      .on("click", (event, d) => {
        if (onNodeClick) onNodeClick(d);
      });

    return () => simulation.stop();
  }, [data, selectedImage, onNodeClick]);

  return <svg ref={svgRef} />;
};

export default EmbeddingsGraph;