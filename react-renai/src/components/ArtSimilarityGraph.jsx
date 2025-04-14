import React, { useEffect, useState, useRef } from 'react';
import { 
  Box, Paper, Typography, Grid, Slider, FormControl,
  InputLabel, Select, MenuItem, CircularProgress
} from '@mui/material';
import ForceGraph3D from 'react-force-graph-3d';
import { useTheme } from '@mui/material/styles';
import SpriteText from 'three-spritetext';
import * as THREE from 'three';

// Утилита для расчета сходства
const calculateSimilarity = (vec1, vec2) => {
  // Косинусное сходство
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
  
  return dotProduct / (norm1 * norm2);
};

const ArtSimilarityGraph = ({ embeddings, artworkMetadata }) => {
  const theme = useTheme();
  const graphRef = useRef();
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [maxConnections, setMaxConnections] = useState(5);
  const [colorBy, setColorBy] = useState('era');
  
  useEffect(() => {
    if (!embeddings || !artworkMetadata) return;
    
    setLoading(true);
    
    // Построение графа на основе эмбеддингов
    const buildGraph = () => {
      const nodes = artworkMetadata.map((meta, index) => ({
        id: index.toString(),
        name: meta.title || `Artwork ${index}`,
        era: meta.era || 'Unknown',
        style: meta.style || 'Unknown',
        artist: meta.artist || 'Unknown',
        imageUrl: meta.imageUrl || '',
        val: 1 // Размер узла
      }));
      
      const links = [];
      
      // Создание связей между похожими произведениями
      for (let i = 0; i < embeddings.length; i++) {
        const similarities = [];
        
        for (let j = 0; j < embeddings.length; j++) {
          if (i === j) continue;
          
          const similarity = calculateSimilarity(embeddings[i], embeddings[j]);
          if (similarity >= similarityThreshold) {
            similarities.push({ target: j, similarity });
          }
        }
        
        // Сортировка и ограничение количества связей
        similarities.sort((a, b) => b.similarity - a.similarity);
        similarities.slice(0, maxConnections).forEach(sim => {
          links.push({
            source: i.toString(),
            target: sim.target.toString(),
            value: sim.similarity
          });
        });
      }
      
      return { nodes, links };
    };
    
    const graph = buildGraph();
    setGraphData(graph);
    setLoading(false);
  }, [embeddings, artworkMetadata, similarityThreshold, maxConnections]);
  
  // Получение цвета узла на основе выбранного атрибута
  const getNodeColor = (node) => {
    if (!node) return theme.palette.primary.main;
    
    const attribute = node[colorBy];
    if (!attribute) return theme.palette.grey[500];
    
    // Хеш-функция для генерации стабильного цвета
    let hash = 0;
    for (let i = 0; i < attribute.length; i++) {
      hash = attribute.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  };
  
  return (
    <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h5" gutterBottom>
        Граф сходства произведений искусства
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Typography gutterBottom>
            Порог сходства: {similarityThreshold.toFixed(2)}
          </Typography>
          <Slider
            value={similarityThreshold}
            onChange={(_, newValue) => setSimilarityThreshold(newValue)}
            min={0.5}
            max={0.95}
            step={0.01}
            valueLabelDisplay="auto"
          />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Typography gutterBottom>
            Макс. количество связей: {maxConnections}
          </Typography>
          <Slider
            value={maxConnections}
            onChange={(_, newValue) => setMaxConnections(newValue)}
            min={1}
            max={20}
            step={1}
            valueLabelDisplay="auto"
          />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Цвет узлов</InputLabel>
            <Select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value)}
              label="Цвет узлов"
            >
              <MenuItem value="era">Эпоха</MenuItem>
              <MenuItem value="style">Стиль</MenuItem>
              <MenuItem value="artist">Художник</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <CircularProgress />
        </Box>
      ) : graphData ? (
        <Box sx={{ flex: 1, minHeight: 500, position: 'relative' }}>
          <Box sx={{ position: 'absolute', top: 10, right: 10, zIndex: 100 }}>
            {/* Легенда будет здесь */}
          </Box>
          
          <ForceGraph3D
            ref={graphRef}
            graphData={graphData}
            nodeLabel="name"
            nodeColor={node => getNodeColor(node)}
            nodeThreeObject={node => {
              const sprite = new SpriteText(node.name);
              sprite.color = '#ffffff';
              sprite.backgroundColor = getNodeColor(node);
              sprite.borderWidth = 0;
              sprite.padding = [2, 4];
              sprite.textHeight = 2;
              sprite.position.y = -4;
              return sprite;
            }}
            nodeThreeObjectExtend={true}
            linkWidth={link => link.value * 3}
            linkDirectionalParticles={4}
            linkDirectionalParticleWidth={link => link.value * 2}
            linkDirectionalParticleSpeed={d => d.value * 0.01}
            linkOpacity={0.7}
            backgroundColor="#ffffff"
            showNavInfo={true}
            cameraPosition={{ z: 150 }}
            onNodeClick={node => {
              // Показать подробную информацию о произведении
              console.log('Node clicked:', node);
              
              // Перемещение камеры к узлу
              const distance = 40;
              const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
              graphRef.current.cameraPosition(
                { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                node,
                2000
              );
            }}
            cooldownTicks={100}
          />
        </Box>
      ) : (
        <Typography variant="body1">Нет данных для отображения</Typography>
      )}
    </Paper>
  );
};

export default ArtSimilarityGraph;
