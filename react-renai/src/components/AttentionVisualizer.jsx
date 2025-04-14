import React, { useState, useEffect } from 'react';
import { 
  Paper, Typography, Grid, Box, Slider, FormControl,
  InputLabel, Select, MenuItem, ToggleButtonGroup, ToggleButton
} from '@mui/material';
import CompareIcon from '@mui/icons-material/Compare';
import ViewModuleIcon from '@mui/icons-material/ViewModule';

const AttentionVisualizer = ({ artworks, attentionMaps, gradcamMaps }) => {
  const [selectedArtwork, setSelectedArtwork] = useState(0);
  const [visualizationType, setVisualizationType] = useState('attention');
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [viewMode, setViewMode] = useState('compare');
  const [selectedLayer, setSelectedLayer] = useState('last');
  
  // Моделирование слоев модели
  const layers = ['first', 'middle', 'last'];
  
  const handleOverlayOpacityChange = (_, newValue) => {
    setOverlayOpacity(newValue);
  };
  
  // Определение карты внимания для текущего произведения и слоя
  const getCurrentMap = () => {
    const maps = visualizationType === 'attention' ? attentionMaps : gradcamMaps;
    if (!maps || !maps[selectedArtwork]) return null;
    
    // В реальном проекте здесь должна быть логика выбора карты для конкретного слоя
    return maps[selectedArtwork];
  };
  
  const currentMap = getCurrentMap();
  const currentArtwork = artworks && artworks[selectedArtwork];
  
  // Custom color overlay для разных типов визуализации
  const getOverlayStyles = () => {
    const baseStyles = {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      opacity: overlayOpacity,
      mixBlendMode: 'screen',
      backgroundSize: 'cover',
      backgroundPosition: 'center'
    };
    
    if (visualizationType === 'attention') {
      return {
        ...baseStyles,
        backgroundImage: `url(${currentMap})`,
      };
    } else {
      return {
        ...baseStyles,
        backgroundImage: `url(${currentMap})`,
      };
    }
  };
  
  if (!artworks || !artworks.length) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6">Нет доступных произведений для визуализации</Typography>
      </Paper>
    );
  }
  
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Визуализация внимания модели
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Произведение</InputLabel>
            <Select
              value={selectedArtwork}
              onChange={(e) => setSelectedArtwork(e.target.value)}
              label="Произведение"
            >
              {artworks.map((artwork, index) => (
                <MenuItem key={index} value={index}>
                  {artwork.title || `Произведение ${index + 1}`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Тип визуализации</InputLabel>
            <Select
              value={visualizationType}
              onChange={(e) => setVisualizationType(e.target.value)}
              label="Тип визуализации"
            >
              <MenuItem value="attention">Attention Map</MenuItem>
              <MenuItem value="gradcam">GradCAM</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Слой</InputLabel>
            <Select
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(e.target.value)}
              label="Слой"
            >
              {layers.map((layer) => (
                <MenuItem key={layer} value={layer}>
                  {layer === 'first' ? 'Ранний слой' : 
                   layer === 'middle' ? 'Средний слой' : 'Финальный слой'}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, newValue) => newValue && setViewMode(newValue)}
            fullWidth
          >
            <ToggleButton value="compare">
              <CompareIcon sx={{ mr: 1 }} />
              Сравнение
            </ToggleButton>
            <ToggleButton value="overlay">
              <ViewModuleIcon sx={{ mr: 1 }} />
              Наложение
            </ToggleButton>
          </ToggleButtonGroup>
        </Grid>
        
        <Grid item xs={12}>
          <Typography gutterBottom>Прозрачность наложения: {overlayOpacity}</Typography>
          <Slider
            value={overlayOpacity}
            onChange={handleOverlayOpacityChange}
            step={0.01}
            min={0}
            max={1}
            valueLabelDisplay="auto"
          />
        </Grid>
      </Grid>
      
      {currentArtwork && (
        <Grid container spacing={3}>
          {viewMode === 'compare' ? (
            <>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom align="center">
                  Оригинальное изображение
                </Typography>
                <Box
                  sx={{
                    width: '100%',
                    pt: '75%', // Соотношение сторон 4:3
                    position: 'relative',
                    backgroundColor: '#f0f0f0',
                    backgroundImage: `url(${currentArtwork.imageUrl})`,
                    backgroundSize: 'contain',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat',
                    borderRadius: 1,
                    overflow: 'hidden'
                  }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom align="center">
                  {visualizationType === 'attention' ? 'Карта внимания' : 'GradCAM'}
                </Typography>
                <Box
                  sx={{
                    width: '100%',
                    pt: '75%', // Соотношение сторон 4:3
                    position: 'relative',
                    backgroundColor: '#f0f0f0',
                    backgroundImage: `url(${currentMap})`,
                    backgroundSize: 'contain',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat',
                    borderRadius: 1,
                    overflow: 'hidden'
                  }}
                />
              </Grid>
            </>
          ) : (
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom align="center">
                Оригинал с наложением {visualizationType === 'attention' ? 'карты внимания' : 'GradCAM'}
              </Typography>
              <Box
                sx={{
                  width: '100%',
                  pt: '56.25%', // Соотношение сторон 16:9
                  position: 'relative',
                  backgroundColor: '#f0f0f0',
                  borderRadius: 1,
                  overflow: 'hidden'
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    backgroundImage: `url(${currentArtwork.imageUrl})`,
                    backgroundSize: 'contain',
                    backgroundPosition: 'center',
                    backgroundRepeat: 'no-repeat'
                  }}
                />
                <Box sx={getOverlayStyles()} />
              </Box>
            </Grid>
          )}
          
          <Grid item xs={12}>
            <Paper sx={{ p: 2, bgcolor: 'rgba(0,0,0,0.02)' }}>
              <Typography variant="h6" gutterBottom>
                Интерпретация
              </Typography>
              <Typography variant="body1">
                {visualizationType === 'attention' 
                  ? 'Карта внимания показывает области, на которые нейронная сеть уделяет наибольшее внимание при анализе произведения. Яркие участки указывают на важные для модели элементы композиции, формы и детали.'
                  : 'GradCAM визуализирует градиенты последнего сверточного слоя для понимания, какие части изображения влияют на классификацию модели. Это помогает определить, какие элементы искусства модель считает характерными для данного стиля или эпохи.'}
              </Typography>
              
              <Typography variant="body1" sx={{ mt: 2 }}>
                {visualizationType === 'attention'
                  ? 'На данном изображении модель фокусируется на фигурах и их расположении в пространстве, что характерно для распознавания древнеегипетского стиля с его канонической передачей человеческих фигур.'
                  : 'GradCAM показывает, что для определения принадлежности к древнеегипетскому искусству, модель сосредотачивается на характерных позах фигур, силуэтах и иерархических пропорциях.'}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Paper>
  );
};

export default AttentionVisualizer;
