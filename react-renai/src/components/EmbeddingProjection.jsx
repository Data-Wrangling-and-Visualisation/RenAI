import React, { useEffect, useState, useMemo } from 'react';
import { 
  Box, Paper, Typography, Grid, FormControl, 
  InputLabel, Select, MenuItem, Slider, Button, 
  CircularProgress, Tooltip 
} from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, 
  CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import * as d3 from 'd3';
import { runTSNE, runUMAP } from '../utils/dimensionReduction';

const EmbeddingProjection = ({ embeddings, artworkMetadata }) => {
  const [projectionType, setProjectionType] = useState('tsne');
  const [perplexity, setPerplexity] = useState(30);
  const [neighbors, setNeighbors] = useState(15);
  const [dimensions, setDimensions] = useState(2);
  const [colorBy, setColorBy] = useState('era');
  const [loading, setLoading] = useState(false);
  const [projectedData, setProjectedData] = useState([]);
  
  // Цветовая схема для разных атрибутов
  const colorScales = useMemo(() => ({
    era: d3.scaleOrdinal(d3.schemeCategory10),
    style: d3.scaleOrdinal(d3.schemeSet2),
    region: d3.scaleOrdinal(d3.schemePaired)
  }), []);
  
  // Выполнение проекции при изменении параметров
  useEffect(() => {
    const runProjection = async () => {
      setLoading(true);
      
      try {
        let result;
        if (projectionType === 'tsne') {
          result = await runTSNE(embeddings, { 
            perplexity, 
            dim: dimensions,
            iterations: 1000 
          });
        } else {
          result = await runUMAP(embeddings, {
            nNeighbors: neighbors,
            nComponents: dimensions,
            minDist: 0.1
          });
        }
        
        // Комбинируем результаты проекции с метаданными
        const projectedWithMetadata = result.map((coords, i) => ({
          ...artworkMetadata[i],
          x: coords[0],
          y: coords[1],
          z: dimensions === 3 ? coords[2] : 0
        }));
        
        setProjectedData(projectedWithMetadata);
      } catch (error) {
        console.error('Projection error:', error);
      } finally {
        setLoading(false);
      }
    };
    
    runProjection();
  }, [embeddings, projectionType, perplexity, neighbors, dimensions]);
  
  // Получение цвета для точки
  const getPointColor = (point) => {
    const attribute = point[colorBy] || 'unknown';
    return colorScales[colorBy](attribute);
  };
  
  // Подготовка легенды
  const legendItems = useMemo(() => {
    if (!projectedData.length) return [];
    
    const uniqueValues = [...new Set(projectedData.map(d => d[colorBy] || 'unknown'))];
    return uniqueValues.map(value => ({
      value,
      color: colorScales[colorBy](value)
    }));
  }, [projectedData, colorBy, colorScales]);
  
  return (
    <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h5" gutterBottom>
        Проекция эмбеддингов произведений искусства
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Метод проекции</InputLabel>
            <Select
              value={projectionType}
              onChange={(e) => setProjectionType(e.target.value)}
              label="Метод проекции"
            >
              <MenuItem value="tsne">t-SNE</MenuItem>
              <MenuItem value="umap">UMAP</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Размерность</InputLabel>
            <Select
              value={dimensions}
              onChange={(e) => setDimensions(e.target.value)}
              label="Размерность"
            >
              <MenuItem value={2}>2D</MenuItem>
              <MenuItem value={3}>3D</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Цвет по</InputLabel>
            <Select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value)}
              label="Цвет по"
            >
              <MenuItem value="era">Эпоха</MenuItem>
              <MenuItem value="style">Стиль</MenuItem>
              <MenuItem value="region">Регион</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Button 
            variant="contained" 
            fullWidth 
            disabled={loading}
            onClick={() => {
              // Триггер для перезапуска проекции с текущими параметрами
              setLoading(true);
              setTimeout(() => setLoading(false), 0);
            }}
          >
            {loading ? <CircularProgress size={24} /> : "Обновить"}
          </Button>
        </Grid>
        
        {projectionType === 'tsne' && (
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Перплексивность: {perplexity}</Typography>
            <Slider
              value={perplexity}
              onChange={(_, newValue) => setPerplexity(newValue)}
              min={5}
              max={50}
              step={1}
              disabled={loading}
            />
          </Grid>
        )}
        
        {projectionType === 'umap' && (
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>Количество соседей: {neighbors}</Typography>
            <Slider
              value={neighbors}
              onChange={(_, newValue) => setNeighbors(newValue)}
              min={2}
              max={50}
              step={1}
              disabled={loading}
            />
          </Grid>
        )}
      </Grid>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ flex: 1, minHeight: 500 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="x" name="Dimension 1" />
              <YAxis type="number" dataKey="y" name="Dimension 2" />
              {dimensions === 3 && <ZAxis type="number" dataKey="z" range={[100, 600]} name="Dimension 3" />}
              <RechartsTooltip 
                cursor={{ strokeDasharray: '3 3' }} 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <Paper sx={{ p: 2, maxWidth: 300 }}>
                        <Typography variant="subtitle2">{data.title || 'Untitled'}</Typography>
                        <Typography variant="body2">Эпоха: {data.era || 'Unknown'}</Typography>
                        <Typography variant="body2">Стиль: {data.style || 'Unknown'}</Typography>
                        {data.artist && (
                          <Typography variant="body2">Художник: {data.artist}</Typography>
                        )}
                      </Paper>
                    );
                  }
                  return null;
                }}
              />
              <Scatter 
                name="Произведения искусства" 
                data={projectedData} 
                fill="#8884d8"
                shape="circle"
                fillOpacity={0.8}
                strokeWidth={1}
                stroke="#fff"
                isAnimationActive={false} // Отключение анимации для большого количества точек
                onClick={(data) => {
                  // Показать детальную информацию о произведении
                  console.log('Selected artwork:', data);
                }}
              >
                {projectedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getPointColor(entry)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </Box>
      )}
      
      {/* Легенда */}
      <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {legendItems.map((item) => (
          <Tooltip key={item.value} title={item.value}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center',
              gap: 1,
              bgcolor: 'rgba(0,0,0,0.05)',
              px: 1,
              py: 0.5,
              borderRadius: 1
            }}>
              <Box 
                sx={{ 
                  width: 16, 
                  height: 16, 
                  bgcolor: item.color, 
                  borderRadius: '50%' 
                }} 
              />
              <Typography variant="caption" noWrap>
                {item.value}
              </Typography>
            </Box>
          </Tooltip>
        ))}
      </Box>
    </Paper>
  );
};

export default EmbeddingProjection;
