import React, { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import { Paper, Typography, Box } from '@mui/material';

// --- Tooltip Component (can be defined outside or inside, here inside for simplicity) ---
const GraphTooltip = ({ tooltipData }) => {
  if (!tooltipData || !tooltipData.visible) return null;

  const { x, y, node } = tooltipData;
  const imageUrl = node.cachedImageUrl || node.primaryImageSmall; // Assuming metadata is merged into node
  const title = node.title || 'Unknown Title';
  const id = node.id || node.objectID || 'Unknown ID';
  const artist = node.artistDisplayName || 'Unknown Artist';
  const date = node.objectDate || 'Unknown Date';
  const medium = node.medium || 'Unknown Medium';

  return (
    <Paper 
      sx={{
        position: 'absolute',
        left: `${x + 15}px`, // Offset from cursor
        top: `${y + 15}px`,
        padding: '10px',
        maxWidth: '250px',
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: '4px',
        boxShadow: 3,
        pointerEvents: 'none', // Important: prevent tooltip from blocking mouse events on graph
        zIndex: 1000, // Ensure tooltip is on top
        transition: 'opacity 0.1s ease-in-out', // Optional: smooth fade
        opacity: tooltipData.visible ? 1 : 0, 
      }}
    >
      {imageUrl ? (
        <img 
          src={imageUrl} 
          alt={`Artwork ${id}`}
          style={{ 
            width: '100%', height: 'auto', maxWidth: '230px', 
            display: 'block', marginBottom: '8px', borderRadius: '4px' 
          }} 
        />
      ) : (
        <Typography variant="caption" display="block" sx={{ mb: 1 }}>
           Image not available
        </Typography>
      )}
      <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 0.5, lineHeight: 1.3 }}>
        {title}
      </Typography>
      <Typography variant="caption" display="block">Artist: {artist}</Typography>
      <Typography variant="caption" display="block">Date: {date}</Typography>
      <Typography variant="caption" display="block" sx={{ mb: 0.5 }}>Medium: {medium}</Typography>
      <Typography variant="caption" display="block" sx={{ color: 'text.secondary' }}>ID: {id}</Typography>
    </Paper>
  );
};
// --- End Tooltip Component ---

const EmbeddingsGraph = ({ data, selectedImage, onNodeClick, artworkMetadata = [] }) => {
  const svgRef = useRef(null);
  const [tooltipData, setTooltipData] = useState({ visible: false, x: 0, y: 0, node: null });
  const nodeSize = 24; // Define a size for the image nodes
  const fallbackRadius = 6; // Radius for fallback circles

  useEffect(() => {
    if (!data || !artworkMetadata) return;

    // Если данные уже имеют структуру nodes/links, используем их.
    // Иначе предполагаем, что data — это массив векторов (например, [[0.13, -1.17, ...], ...])
    let graphData;
    if (data.nodes && data.links) {
      graphData = {
        nodes: data.nodes.map((node, i) => ({
          ...node,
          ...(artworkMetadata.find(meta => meta.id === node.id || meta.objectID === node.id) || {}),
          // Add image URL directly to node for easier access
          nodeImageUrl: (artworkMetadata.find(meta => meta.id === node.id || meta.objectID === node.id) || {}).cachedImageUrl || (artworkMetadata.find(meta => meta.id === node.id || meta.objectID === node.id) || {}).primaryImageSmall
        })),
        links: data.links
      };
    } else if (Array.isArray(data)) {
      // Ensure metadata aligns with embeddings
      if (data.length !== artworkMetadata.length) {
        console.warn(`Graph Embeddings (${data.length}) and metadata (${artworkMetadata.length}) lengths differ.`);
        return;
      }
      const nodes = data.map((embedding, i) => {
        const meta = artworkMetadata[i] || {};
        const x = embedding[0] !== undefined ? embedding[0] * 50 + 400 : Math.random() * 800;
        const y = embedding[1] !== undefined ? embedding[1] * 50 + 300 : Math.random() * 600;
        return { 
          id: meta.id || meta.objectID || i,
          x, y, 
          embedding, 
          ...meta,
          nodeImageUrl: meta.cachedImageUrl || meta.primaryImageSmall // Add image URL
        };
      });
      graphData = { nodes, links: [] };
    } else {
      console.error("Invalid data format for EmbeddingsGraph");
      return;
    }

    const width = 800, height = 600;
    const svg = d3.select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .style("border", "1px solid lightgrey");

    svg.selectAll("*").remove();

    // Define clipPath for circular images
    svg.append("defs").append("clipPath")
        .attr("id", "node-clip")
      .append("circle")
        .attr("r", nodeSize / 2);

    const simulation = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(50)) // Increased distance for images
      .force("charge", d3.forceManyBody().strength(-80))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(nodeSize / 2 + 4)) // Collision based on image size
      .on("tick", ticked);
      
    const linkElements = svg.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(graphData.links)
      .join("line")
        .attr("stroke-width", d => Math.sqrt(d.value || 1));

    // --- Create node groups --- 
    const nodeGroups = svg.append("g")
      .selectAll("g.node-group")
      .data(graphData.nodes)
      .join("g")
      .attr("class", "node-group")
      .style("cursor", "pointer")
      .call(drag(simulation))
      .on("click", (event, d) => {
          event.stopPropagation();
          if (onNodeClick) onNodeClick(d); 
      })
      .on("mouseover", (event, d) => {
          setTooltipData({ visible: true, x: event.pageX, y: event.pageY, node: d });
          // Highlight effect (e.g., border on the group or image/circle)
          d3.select(event.currentTarget).select('.node-shape').attr("stroke", "orange").attr("stroke-width", 2);
      })
      .on("mouseout", (event, d) => {
          setTooltipData({ visible: false, x: 0, y: 0, node: null });
          // Remove highlight
          d3.select(event.currentTarget).select('.node-shape').attr("stroke", "none");
      });

    // --- Append image or fallback circle to each group --- 
    nodeGroups.each(function(d) {
      const group = d3.select(this);
      if (d.nodeImageUrl) {
        group.append("image")
          .attr("class", "node-shape") // Add class for selection
          .attr("xlink:href", d.nodeImageUrl)
          .attr("clip-path", "url(#node-clip)") // Apply circular clip
          .attr("width", nodeSize)
          .attr("height", nodeSize)
          .attr("x", -nodeSize / 2) // Offset for centering
          .attr("y", -nodeSize / 2)
          .on("error", function() { // Handle image loading error -> replace with fallback
              d3.select(this).remove(); // Remove broken image
              group.append("circle") // Append fallback circle instead
                 .attr("class", "node-shape")
                 .attr("r", fallbackRadius)
                 .attr("fill", "grey"); // Indicate fallback
          });
      } else {
        group.append("circle")
          .attr("class", "node-shape") // Add class for selection
          .attr("r", fallbackRadius)
          .attr("fill", "steelblue");
      }
    });

    function ticked() {
      linkElements
          .attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);
          
      // Update group position instead of individual shapes
      nodeGroups
        .attr("transform", d => `translate(${d.x},${d.y})`);
    }

    function drag(simulation) {
      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }
      
      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }
      
      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
      
      return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
    }

    return () => simulation.stop();
  }, [data, artworkMetadata, selectedImage, onNodeClick]);

  return (
    <Box sx={{ position: 'relative', width: '800px', height: '600px' }}>
      <svg ref={svgRef}></svg>
      <GraphTooltip tooltipData={tooltipData} />
    </Box>
  );
};

export default EmbeddingsGraph;