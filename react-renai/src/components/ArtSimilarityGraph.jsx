import React, { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import { 
  Box, Paper, Typography, Grid, Slider, FormControl,
  InputLabel, Select, MenuItem, CircularProgress, Tooltip as MuiTooltip,
  Link // Для ссылки на Met
} from '@mui/material';
import ForceGraph3D from 'react-force-graph-3d';
import { useTheme } from '@mui/material/styles';
// import SpriteText from 'three-spritetext'; // SpriteText может быть медленным, попробуем без него
import * as THREE from 'three';
import * as d3 from 'd3';

const calculateSimilarity = (vec1, vec2) => {
  if (!vec1 || !vec2 || vec1.length !== vec2.length) return 0;
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }
  
  norm1 = Math.sqrt(norm1);
  norm2 = Math.sqrt(norm2);
  
  // Avoid division by zero
  if (norm1 === 0 || norm2 === 0) return 0;
  
  const similarity = dotProduct / (norm1 * norm2);
  // Clamp similarity to [0, 1] range as negative cosine similarity is not intuitive here
  return Math.max(0, similarity);
};

const getColorAttributeValue = (point, attribute) => {
    switch (attribute) {
        case 'era':
            if (!point.objectDate) return 'Unknown Era';
            const yearMatch = String(point.objectDate).match(/\b(\d{4})\b/);
            const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
            if (!year) return 'Unknown Date Format';
            if (year < 1500) return 'Before 1500';
            if (year < 1700) return '1500-1699';
            if (year < 1800) return '1700-1799';
            if (year < 1900) return '1800-1899';
            return '1900 onwards';
        case 'style':
            return point.classification || 'Unknown Classification';
        case 'region':
            return point.culture || 'Unknown Culture';
        case 'artist':
             return point.artistDisplayName || 'Unknown Artist';
        case 'department':
             return point.department || 'Unknown Department';
        default:
            return point[attribute] || `Unknown`;
    }
};


const ArtSimilarityGraph = ({ embeddings, artworkMetadata, onArtworkSelect, selectedArtworkId, isLoading }) => {
  // Log received props on component render/update
  console.log('ArtSimilarityGraph received props:', {
      embeddingsLength: embeddings?.length,
      metadataLength: artworkMetadata?.length,
      // First few embedding dimensions (example)
      firstEmbeddingSample: embeddings?.[0]?.slice(0, 5),
      // First metadata item ID (example)
      firstMetadataId: artworkMetadata?.[0]?.id,
      isLoading,
      selectedArtworkId,
  });

  const theme = useTheme();
  const graphRef = useRef();
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [similarityThreshold, setSimilarityThreshold] = useState(0.65);
  const [maxConnections, setMaxConnections] = useState(5);
  const [colorBy, setColorBy] = useState('era');
  const [internalLoading, setInternalLoading] = useState(true);
  const [hoveredNode, setHoveredNode] = useState(null);

  const colorScales = useMemo(() => ({
    era: d3.scaleOrdinal(d3.schemeCategory10),
    style: d3.scaleOrdinal(d3.schemeSet2),
    region: d3.scaleOrdinal(d3.schemePaired),
    artist: d3.scaleOrdinal(d3.schemeTableau10),
    department: d3.scaleOrdinal(d3.schemeAccent),
  }), []);

  useEffect(() => {
    setInternalLoading(true);
    console.log("ArtSimilarityGraph: Recalculating graph data...");
    const timer = setTimeout(() => {
        if (!embeddings || !artworkMetadata) {
            console.warn("ArtSimilarityGraph: Missing embeddings or metadata.");
            setGraphData({ nodes: [], links: [] });
            setInternalLoading(false);
            return;
        }

        try {
            console.log(`ArtSimilarityGraph: Processing data - embeddings length: ${embeddings.length}, metadata length: ${artworkMetadata.length}`);
            
            const metadataMap = new Map();
            artworkMetadata.forEach(meta => {
                if (meta && meta.id) {
                    metadataMap.set(meta.id, meta);
                }
            });
            
            const embeddingMap = new Map();
            embeddings.forEach((emb, idx) => {
                const meta = idx < artworkMetadata.length ? artworkMetadata[idx] : null;
                if (meta && meta.id && emb) {
                    embeddingMap.set(meta.id, emb);
                }
            });
            
            console.log(`ArtSimilarityGraph: Created maps - metadata: ${metadataMap.size}, embeddings: ${embeddingMap.size}`);
            
            const validIds = [...metadataMap.keys()].filter(id => embeddingMap.has(id));
            console.log(`ArtSimilarityGraph: Found ${validIds.length} valid items with both metadata and embeddings`);
            
            if (validIds.length === 0) {
                console.warn("ArtSimilarityGraph: No valid items with both metadata and embeddings.");
                setGraphData({ nodes: [], links: [] });
                setInternalLoading(false);
                return;
            }
            
            // Ограничим количество узлов для производительности
            const MAX_NODES = 300;
            let nodesToUse = validIds;
            
            if (validIds.length > MAX_NODES) {
                console.log(`ArtSimilarityGraph: Limiting to ${MAX_NODES} nodes for performance (from ${validIds.length} valid IDs)`);
                // Если есть выбранный узел, убедимся, что он включен
                if (selectedArtworkId && validIds.includes(selectedArtworkId)) {
                    // Выбираем случайные узлы, но включаем выбранный
                    const shuffledIds = validIds.filter(id => id !== selectedArtworkId)
                        .sort(() => Math.random() - 0.5)
                        .slice(0, MAX_NODES - 1);
                    
                    nodesToUse = [selectedArtworkId, ...shuffledIds];
                } else {
                    // Просто выбираем случайные узлы
                    nodesToUse = validIds.sort(() => Math.random() - 0.5).slice(0, MAX_NODES);
                }
            }
            
            // Create nodes only for items that have both metadata and embedding
            const nodes = nodesToUse.map(id => ({
                id: id,
                ...metadataMap.get(id),
                val: 15
            }));

            const links = [];
            
            // Более эффективный алгоритм для больших наборов данных
            // Для каждого узла вычисляем similarity со всеми остальными узлами
            // и сохраняем только top maxConnections с similarity >= similarityThreshold
            console.time('ArtSimilarityGraph: Links computation');
            
            for (let i = 0; i < nodes.length; i++) {
                const sourceNode = nodes[i];
                const sourceEmbedding = embeddingMap.get(sourceNode.id);
                if (!sourceEmbedding) continue; 

                const similarities = [];
                for (let j = 0; j < nodes.length; j++) { 
                    if (i === j) continue; // Пропускаем связь с самим собой
                    
                    const targetNode = nodes[j];
                    const targetEmbedding = embeddingMap.get(targetNode.id);
                    if (!targetEmbedding) continue;

                    const similarity = calculateSimilarity(sourceEmbedding, targetEmbedding);
                    if (similarity >= similarityThreshold) {
                        similarities.push({ target: targetNode.id, similarity });
                    }
                }

                // Сортируем по убыванию similarity и берем top maxConnections
                similarities.sort((a, b) => b.similarity - a.similarity);
                similarities.slice(0, maxConnections).forEach(sim => {
                    // Проверяем, не добавили ли мы уже эту связь в обратном направлении
                    const linkExists = links.some(
                        link => (link.source === sim.target && link.target === sourceNode.id) ||
                              (link.source === sourceNode.id && link.target === sim.target)
                    );
                    
                    if (!linkExists) {
                        links.push({
                            source: sourceNode.id,
                            target: sim.target,
                            value: sim.similarity
                        });
                    }
                });
            }
            
            console.timeEnd('ArtSimilarityGraph: Links computation');
            console.log(`ArtSimilarityGraph: Generated ${nodes.length} nodes and ${links.length} links.`);
            setGraphData({ nodes, links });
        } catch (error) {
            console.error('ArtSimilarityGraph: Error processing graph data:', error);
            setGraphData({ nodes: [], links: [] }); // Clear data on error
        } finally {
             setInternalLoading(false);
             console.log("ArtSimilarityGraph: Graph data calculation finished.");
        }
     }, 50);

     return () => clearTimeout(timer); // Очистка таймера

  }, [embeddings, artworkMetadata, similarityThreshold, maxConnections, selectedArtworkId]); // Пересчет при изменении этих зависимостей

  const getNodeColor = useCallback((node) => {
    if (!node) return theme.palette.grey[500]; // Default grey if node is null
    
    if (selectedArtworkId && node.id === selectedArtworkId) {
        return theme.palette.secondary.main; // Яркий цвет для выделения
    }

    const attributeValue = getColorAttributeValue(node, colorBy);
    const scale = colorScales[colorBy] || d3.scaleOrdinal(d3.schemeCategory10); // Fallback scale

    return scale(attributeValue);
  }, [colorBy, colorScales, theme, selectedArtworkId]);

  // Обработчик клика на узел
  const handleNodeClick = useCallback((node) => {
    if (node && node.id && onArtworkSelect) {
        console.log('ArtSimilarityGraph: Node clicked, selecting ID:', node.id);
        onArtworkSelect(node.id);

        const distance = 60;
        const distRatio = 1 + distance / Math.hypot(node.x || 0, node.y || 0, node.z || 0);
        if (graphRef.current) {
             graphRef.current.cameraPosition(
                { 
                    x: (node.x || 0) * distRatio,
                    y: (node.y || 0) * distRatio,
                    z: (node.z || 0) * distRatio 
                },
                node,
                1000
             );
        }
    }
  }, [onArtworkSelect]);

  const memoizedGraphData = useMemo(() => graphData, [graphData]);

  const nodeThreeObject = useCallback((node) => {
        const scale = (selectedArtworkId && node.id === selectedArtworkId) ? 1.5 : 1;
        const geometry = new THREE.SphereGeometry(2.5 * scale); 
        const material = new THREE.MeshLambertMaterial({ 
            color: getNodeColor(node), 
            transparent: true, 
            opacity: 0.85 
        });
        return new THREE.Mesh(geometry, material);
  }, [getNodeColor, selectedArtworkId]);

  const handleNodeHover = useCallback((node) => {
      setHoveredNode(node || null);
      document.body.style.cursor = node ? 'pointer' : 'default';
  }, []);

  return (
    <Paper sx={{ p: 2, height: 'calc(100vh - 64px - 32px)', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        Artwork similarity graph (3D)
      </Typography>
      
      {/* Панель управления */}
      <Grid container spacing={2} sx={{ mb: 2, flexShrink: 0 }}>
        <Grid item xs={12} md={4}>
          <Typography variant="caption" display="block" gutterBottom>
            Similarity threshold: {similarityThreshold.toFixed(2)}
          </Typography>
          <Slider
            value={similarityThreshold}
            onChange={(_, newValue) => setSimilarityThreshold(newValue)}
            min={0.5}
            max={0.95}
            step={0.01}
            valueLabelDisplay="auto"
            size="small"
          />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Typography variant="caption" display="block" gutterBottom>
            Maximum number of connections: {maxConnections}
          </Typography>
          <Slider
            value={maxConnections}
            onChange={(_, newValue) => setMaxConnections(newValue)}
            min={1}
            max={20}
            step={1}
            valueLabelDisplay="auto"
            size="small"
          />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Node color</InputLabel>
            <Select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value)}
              label="Node color"
            >
              <MenuItem value="era">Era</MenuItem>
              <MenuItem value="style">Style (Classification)</MenuItem>
              <MenuItem value="region">Region (Culture)</MenuItem>
              <MenuItem value="artist">Artist</MenuItem>
              <MenuItem value="department">Department</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      
      {/* Область графа и тултипа */}
      <Box sx={{ flex: 1, minHeight: 400, position: 'relative', border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
        {(isLoading || internalLoading) && (
           <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(255, 255, 255, 0.7)', zIndex: 10 }}>
             <CircularProgress />
             <Typography sx={{ ml: 2 }}>Loading graph data...</Typography>
           </Box>
        )}
        {!(isLoading || internalLoading) && memoizedGraphData.nodes.length === 0 && (
            <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', p: 3, textAlign: 'center' }}>
                <Typography variant="body1">No data to display the graph.</Typography>
                <Typography variant="caption">Ensure that the embeddings are loaded.</Typography>
                <Box sx={{ mt: 2, maxWidth: 500 }}>
                    <Typography variant="caption" color="text.secondary">
                        Detected mismatch between metadata and embeddings ({artworkMetadata?.length || 0} metadata, {embeddings?.length || 0} embeddings).
                        Possibly, some items do not have embeddings or there was a synchronization error.
                    </Typography>
                </Box>
            </Box>
        )}
        {/* Tooltip Overlay */} 
        {hoveredNode && (
            <Paper 
                elevation={4} 
                sx={{
                    position: 'absolute',
                    top: 10, // Позиционируем вверху
                    left: 10, // Позиционируем слева
                    p: 1.5,
                    maxWidth: 250,
                    zIndex: 10,
                    pointerEvents: 'none', // Чтобы не мешал взаимодействию с графом
                    backgroundColor: 'rgba(255, 255, 255, 0.9)'
                }}
            >
                { (hoveredNode.cachedImageUrl || hoveredNode.primaryImageSmall) && (
                     <Box sx={{ mb: 1, width: '100%', height: 100, display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
                         <img 
                            src={hoveredNode.cachedImageUrl || `/api/proxy_image?url=${encodeURIComponent(hoveredNode.primaryImageSmall)}`}
                            alt={hoveredNode.title || 'Preview'}
                            style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                            onError={(e) => { e.target.style.display = 'none'; }}
                         />
                     </Box>
                )}
                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{hoveredNode.title || 'Untitled'}</Typography>
                <Typography variant="caption" display="block">{hoveredNode.artistDisplayName || 'Unknown Artist'}</Typography>
                <Typography variant="caption" display="block">{hoveredNode.objectDate || 'Unknown Date'}</Typography>
                {hoveredNode.objectURL && (
                     <Link href={hoveredNode.objectURL} target="_blank" rel="noopener" variant="caption">View on Met</Link>
                 )}
            </Paper>
        )}

        {!(isLoading || internalLoading) && memoizedGraphData.nodes.length > 0 && (
             <ForceGraph3D
                ref={graphRef}
                graphData={memoizedGraphData}
                nodeColor={getNodeColor}
                nodeThreeObject={nodeThreeObject}
                nodeThreeObjectExtend={false}
                linkWidth={link => link.value * 2}
                linkDirectionalParticles={1}
                linkDirectionalParticleWidth={1.5}
                linkDirectionalParticleResolution={4}
                linkDirectionalParticleSpeed={d => d.value * 0.005}
                linkOpacity={0.4}
                backgroundColor={theme.palette.background.paper}
                showNavInfo={false}
                enableNavigationControls={true}
                onNodeClick={handleNodeClick}
                onNodeHover={handleNodeHover}
                cooldownTicks={100}
                d3ForceLink={d3.forceLink().distance(20).strength(0.5)}
                d3ForceManyBody={d3.forceManyBody().strength(-15)}
                d3ForceCenter={d3.forceCenter()}
             />
         )}
      </Box>
    </Paper>
  );
};

export default ArtSimilarityGraph;
