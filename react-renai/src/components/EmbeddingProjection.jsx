import React, { useEffect, useState, useMemo, useCallback } from 'react';
import {
  Box, Paper, Typography, Grid, FormControl,
  InputLabel, Select, MenuItem, Slider, Button,
  CircularProgress, Tooltip, List, ListItem, ListItemIcon, ListItemText
} from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import * as d3 from 'd3';
import { runTSNE, runUMAP } from '../utils/dimensionReduction';
import { debounce } from 'lodash';
import CircleIcon from '@mui/icons-material/Circle'; // For legend icons

// Fixed t-SNE perplexity
const FIXED_TSNE_PERPLEXITY = 30;

// --- Custom Tooltip Component for Image Preview ---
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload; // The data object for the hovered point
    const imageUrl = data.cachedImageUrl || data.primaryImageSmall; 
    const title = data.title || 'Unknown Title';
    const id = data.id || data.objectID || 'Unknown ID';
    const artist = data.artistDisplayName || 'Unknown Artist';
    const date = data.objectDate || 'Unknown Date';
    const medium = data.medium || 'Unknown Medium';

    return (
      <Paper sx={{ padding: '10px', maxWidth: '250px', backgroundColor: 'rgba(255, 255, 255, 0.95)' }}> {/* Slightly wider, more opaque */}
        {imageUrl ? (
          <img 
            src={imageUrl} 
            alt={`Artwork ${id}`}
            style={{ 
              width: '100%', 
              height: 'auto', 
              maxWidth: '230px', 
              display: 'block',
              marginBottom: '8px', // Increased spacing
              borderRadius: '4px' // Slightly rounded corners
            }} 
          />
        ) : (
          <Typography variant="caption" display="block" sx={{ mb: 1 }}> {/* Spacing even if no image */}
             Image not available
          </Typography>
        )}
        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 0.5, lineHeight: 1.3 }}>
          {title}
        </Typography>
        <Typography variant="caption" display="block">
          Artist: {artist}
        </Typography>
        <Typography variant="caption" display="block">
          Date: {date}
        </Typography>
        <Typography variant="caption" display="block" sx={{ mb: 0.5 }}>
          Medium: {medium}
        </Typography>
        <Typography variant="caption" display="block" sx={{ color: 'text.secondary' }}>
          ID: {id}
        </Typography>
        <Typography variant="caption" display="block" sx={{ color: 'text.secondary' }}>
          X: {data.x?.toFixed(2)}, Y: {data.y?.toFixed(2)}
        </Typography>
      </Paper>
    );
  }

  return null;
};
// --- End Custom Tooltip ---

const EmbeddingProjection = ({ embeddings, artworkMetadata, onArtworkSelect, selectedArtworkId }) => {
  const [projectionType, setProjectionType] = useState('tsne');
  // Removed perplexity state
  const [neighbors, setNeighbors] = useState(15);
  const [dimensions, setDimensions] = useState(2);
  const [colorBy, setColorBy] = useState('era');
  const [loading, setLoading] = useState(false);
  const [projectedData, setProjectedData] = useState([]);
  // Removed debouncedPerplexity state
  const [debouncedNeighbors, setDebouncedNeighbors] = useState(15);
  
  const colorScales = useMemo(() => ({
    era: d3.scaleOrdinal(d3.schemeCategory10),
    style: d3.scaleOrdinal(d3.schemeSet2),
    region: d3.scaleOrdinal(d3.schemePaired),
    // Add scales for other potential attributes if needed
    department: d3.scaleOrdinal(d3.schemeTableau10),
    classification: d3.scaleOrdinal(d3.schemeAccent),
    culture: d3.scaleOrdinal(d3.schemePastel1),
  }), []);
  
  // Removed debouncePerplexity
  const debounceNeighbors = useCallback(debounce((value) => setDebouncedNeighbors(value), 500), []);
  
  // Добавим логирование при получении данных
  useEffect(() => {
    console.log("EmbeddingProjection: Received data:", {
      embeddingsLength: embeddings?.length,
      metadataLength: artworkMetadata?.length
    });
  }, [embeddings, artworkMetadata]);
  
  useEffect(() => {
    const runProjection = async () => {
      // Basic validation moved up
      if (!embeddings || !Array.isArray(embeddings) || embeddings.length === 0) {
        console.warn("Embeddings are missing or invalid.");
        setProjectedData([]);
        setLoading(false); // Ensure loading stops
        return;
      }
      if (!artworkMetadata || !Array.isArray(artworkMetadata) || artworkMetadata.length === 0) {
          console.warn("Artwork metadata is missing or invalid.");
          setProjectedData([]);
          setLoading(false); // Ensure loading stops
          return;
      }
      // More robust check for alignment *before* filtering
      if (embeddings.length !== artworkMetadata.length) {
          console.warn(`Embeddings (${embeddings.length}) and metadata (${artworkMetadata.length}) lengths differ initially.`);
          // Attempt to align based on non-null embeddings, but proceed with caution
      }

      setLoading(true);
      try {
        console.log(`EmbeddingProjection: Processing embeddings (${embeddings.length}) and metadata (${artworkMetadata.length})`);
        
        // Filter embeddings and metadata *together* to maintain alignment
        const validData = embeddings
          .map((emb, i) => ({ emb, meta: artworkMetadata[i], originalIndex: i }))
          .filter(item => item.emb && Array.isArray(item.emb) && item.emb.length > 0 && item.meta); // Ensure meta exists too

        const validEmbeddings = validData.map(item => item.emb);
        const validMetadata = validData.map(item => item.meta);

        console.log(`EmbeddingProjection: Filtered to ${validEmbeddings.length} valid items for projection`);

        if (validEmbeddings.length === 0) {
            console.warn("No valid embeddings found to project after filtering.");
            setProjectedData([]);
            setLoading(false);
            return;
        }
        console.log(`Projecting ${validEmbeddings.length} valid points.`);

        // Ограничим количество точек для проекции, если их слишком много
        // Это необходимо для производительности
        const MAX_PROJECTION_POINTS = 1000;
        let embeddingsToProject = validEmbeddings;
        let metadataToProject = validMetadata;
        
        if (validEmbeddings.length > MAX_PROJECTION_POINTS) {
          console.log(`EmbeddingProjection: Limiting projection to ${MAX_PROJECTION_POINTS} random points for performance`);
          
          // Создаем индексы и перемешиваем их
          const indices = Array.from({ length: validEmbeddings.length }, (_, i) => i);
          const shuffledIndices = indices.sort(() => Math.random() - 0.5).slice(0, MAX_PROJECTION_POINTS);
          
          // Выбираем подмножество данных
          embeddingsToProject = shuffledIndices.map(i => validEmbeddings[i]);
          metadataToProject = shuffledIndices.map(i => validMetadata[i]);
        }

        let result;
        if (projectionType === 'tsne') {
          // Use fixed perplexity
          console.log(`Running t-SNE with fixed perplexity: ${FIXED_TSNE_PERPLEXITY}, dimensions: ${dimensions}`);
          result = await runTSNE(embeddingsToProject, {
            perplexity: FIXED_TSNE_PERPLEXITY,
            dim: dimensions,
            iterations: 500
          });
        } else { // umap
          // Use debounced neighbors
          console.log(`Running UMAP with neighbors: ${debouncedNeighbors}, dimensions: ${dimensions}`);
          result = await runUMAP(embeddingsToProject, {
            nNeighbors: debouncedNeighbors,
            nComponents: dimensions,
            minDist: 0.1
          });
        }

        if (!result || result.length !== embeddingsToProject.length) {
            console.error("Projection result length mismatch or missing result!");
            setProjectedData([]);
            setLoading(false);
            return;
        }

        const projectedWithMetadata = result.map((coords, i) => ({
          ...metadataToProject[i], // Use metadata corresponding to valid embeddings
          x: coords[0],
          y: coords[1],
          z: dimensions === 3 ? coords[2] : 0
        }));

        setProjectedData(projectedWithMetadata);
        console.log(`EmbeddingProjection: Projection complete, ${projectedWithMetadata.length} points available`);
      } catch (error) {
        console.error('Projection error:', error);
        setProjectedData([]); // Clear data on error
      } finally {
        setLoading(false);
      }
    };
    
    runProjection();
    // Depend on debouncedNeighbors for UMAP, and other relevant params
  }, [embeddings, artworkMetadata, projectionType, debouncedNeighbors, dimensions]);
  
  // Helper function to get the attribute value for coloring
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
            // Using classification as a proxy for style
            return point.classification || 'Unknown Classification';
        case 'region':
            // Using culture as region
            return point.culture || 'Unknown Culture';
        default:
            // Allow coloring by any valid metadata key
            return point[attribute] || `Unknown ${attribute}`;
    }
  };


  const getPointColor = (point, isSelected) => {
    if (isSelected) {
        return '#ff0000'; // Bright red for selected point
    }
    const attributeValue = getColorAttributeValue(point, colorBy);
    const scale = colorScales[colorBy] || d3.scaleOrdinal(d3.schemeCategory10); // Fallback scale

    return scale(attributeValue);
  };

  // Removed handlePerplexityChange

  const handleNeighborsChange = (_, newValue) => {
    setNeighbors(newValue);
    debounceNeighbors(newValue);
  };

  // Improved Legend Calculation
  const legendItems = useMemo(() => {
    if (!projectedData.length) return [];

    const valueMap = new Map();
    projectedData.forEach(d => {
        const value = getColorAttributeValue(d, colorBy);
        if (!valueMap.has(value)) {
            valueMap.set(value, getPointColor(d, false)); // Store value and its color
        }
    });

    // Convert map to array and sort for consistent order
    return Array.from(valueMap.entries())
                .map(([value, color]) => ({ value, color }))
                .sort((a, b) => String(a.value).localeCompare(String(b.value)));

  }, [projectedData, colorBy, colorScales]);


  // Get descriptive label for the "Color By" selection
  const getColorByLabel = (value) => {
    switch(value) {
      case 'era': return 'Эпоха';
      case 'style': return 'Стиль (Классификация)';
      case 'region': return 'Регион (Культура)';
      default: return value.charAt(0).toUpperCase() + value.slice(1); // Capitalize
    }
  }

  return (
    <Paper sx={{ p: 3, height: 'calc(100vh - 64px - 48px)', display: 'flex', flexDirection: 'column' }}> {/* Adjust height based on layout */}
      <Typography variant="h5" gutterBottom>
        Artwork embeddings projection
      </Typography>

      {/* Controls Grid */}
      <Grid container spacing={2} sx={{ mb: 2, flexShrink: 0 }}>
        {/* Projection Type */}
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel>Метод</InputLabel>
            <Select
              value={projectionType}
              onChange={(e) => setProjectionType(e.target.value)}
              label="Метод"
            >
              <MenuItem value="tsne">t-SNE</MenuItem>
              <MenuItem value="umap">UMAP</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        {/* Dimensions */}
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel>Размерность</InputLabel>
            <Select
              value={dimensions}
              onChange={(e) => setDimensions(e.target.value)}
              label="Размерность"
              disabled={loading}
            >
              <MenuItem value={2}>2D</MenuItem>
              {/* <MenuItem value={3}>3D</MenuItem> */} {/* 3D temporarily disabled for simplicity */}
            </Select>
          </FormControl>
        </Grid>

        {/* Color By */}
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel>Цвет по</InputLabel>
            <Select
              value={colorBy}
              onChange={(e) => setColorBy(e.target.value)}
              label="Цвет по"
              disabled={loading}
            >
              <MenuItem value="era">Era</MenuItem>
              <MenuItem value="style">Style (Classification)</MenuItem>
              <MenuItem value="region">Region (Culture)</MenuItem>
              {/* Dynamically add other potential keys? Could be complex */}
            </Select>
          </FormControl>
        </Grid>

        {/* Neighbors Slider (for UMAP) */}
        {projectionType === 'umap' && (
          <Grid item xs={12} sm={6} md={3}>
             <Box sx={{ px: 1 }}>
                 <Typography variant="caption" display="block" gutterBottom>Neighbors: {neighbors}</Typography>
                 <Slider
                    value={neighbors}
                    onChange={handleNeighborsChange}
                    min={2}
                    max={50}
                    step={1}
                    disabled={loading}
                    size="small"
                 />
             </Box>
          </Grid>
        )}
        {/* Removed Perplexity Slider */}
        {/* Removed Update Button */}
      </Grid>

      {/* Main Content Area (Chart and Legend) */}
      <Box sx={{ flex: 1, display: 'flex', minHeight: 0 }}> {/* Allow shrinking and growing */}
        {/* Chart Area */}
        <Box sx={{ flexGrow: 1, width: '100%', minHeight: 300 }}> {/* Ensure chart box grows */}
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <CircularProgress />
            </Box>
          ) : projectedData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{
                  top: 20,
                  right: 20,
                  bottom: 20,
                  left: 20,
                }}
              >
                <CartesianGrid />
                <XAxis type="number" dataKey="x" name="Dimension 1" hide />
                <YAxis type="number" dataKey="y" name="Dimension 2" hide />
                {/* Use the custom tooltip */} 
                <RechartsTooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  content={<CustomTooltip />} 
                />
                <Scatter 
                  name="Artworks" 
                  data={projectedData}
                  onClick={(data) => {
                    if (data && onArtworkSelect) {
                      onArtworkSelect(data.id || data.objectID); // Use id or objectID
                    }
                  }}
                  style={{ cursor: 'pointer' }} // Add pointer cursor on hover
                >
                  {projectedData.map((entry, index) => {
                      const isSelected = (entry.id || entry.objectID) === selectedArtworkId;
                      return (
                          <Cell 
                              key={`cell-${index}`}
                              fill={getPointColor(entry, isSelected)} 
                              fillOpacity={isSelected ? 1 : 0.7}
                              stroke={isSelected ? '#000' : 'none'} // Add border to selected
                              strokeWidth={isSelected ? 2 : 0} 
                          />
                      );
                  })}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <Typography sx={{ textAlign: 'center', mt: 4 }}>
              No data to display or projection failed.
            </Typography>
          )}
        </Box>

        {/* Legend Area */}
        <Box sx={{
            width: 250, // Fixed width for the legend
            ml: 2, // Margin from the chart
            borderLeft: '1px solid #e0e0e0',
            pl: 2, // Padding inside the legend area
            overflowY: 'auto', // Make legend scrollable
            flexShrink: 0, // Prevent legend from shrinking
            maxHeight: '100%' // Ensure it respects parent height
         }}>
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 1 }}>
            Legend ({getColorByLabel(colorBy)})
          </Typography>
          <List dense disablePadding>
             {/* Add item for selected point */}
             <ListItem disableGutters>
                <ListItemIcon sx={{ minWidth: 32 }}>
                    <CircleIcon sx={{ color: '#ff0000', fontSize: 16 }} />
                </ListItemIcon>
                <ListItemText primary="Selected" primaryTypographyProps={{ variant: 'caption' }} />
             </ListItem>
            {legendItems.map((item) => (
              <ListItem key={item.value} disableGutters>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Tooltip title={item.color} placement="left">
                    <CircleIcon sx={{ color: item.color, fontSize: 16 }} />
                  </Tooltip>
                </ListItemIcon>
                <ListItemText
                  primary={item.value}
                  primaryTypographyProps={{ variant: 'caption', noWrap: true }}
                  secondary={
                    <Tooltip title={item.value} placement="bottom-start">
                      <span></span>{/* Tooltip needs a child */}
                    </Tooltip>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Box>
      </Box>
    </Paper>
  );
};

export default EmbeddingProjection;
