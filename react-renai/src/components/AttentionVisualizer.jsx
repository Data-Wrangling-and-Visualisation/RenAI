import React, { useState, useEffect, useMemo } from 'react';
import { 
  Paper, Typography, Grid, Box, Slider, FormControl,
  InputLabel, Select, MenuItem, ToggleButtonGroup, ToggleButton, Container, Alert, AlertTitle, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import CompareIcon from '@mui/icons-material/Compare';
import ViewModuleIcon from '@mui/icons-material/ViewModule';
import ImageIcon from '@mui/icons-material/Image';
import LayersIcon from '@mui/icons-material/Layers';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';

// Function to safely get nested properties
const getSafe = (fn, defaultValue) => {
  try {
    return fn() ?? defaultValue;
  } catch (e) {
    return defaultValue;
  }
};

const AttentionVisualizer = ({ selectedArtworkData }) => {
  const [visualizationType, setVisualizationType] = useState('attention'); // 'attention' or 'gradcam'
  // const [selectedLayer, setSelectedLayer] = useState('final'); // Layer selection might be obsolete if maps are pre-generated
  const [viewMode, setViewMode] = useState('comparison'); // 'comparison' or 'overlay'
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

  console.log("AttentionVisualizer received selectedArtworkData:", selectedArtworkData);

  // Extract data using safe navigation
  const currentArtworkMetadata = useMemo(() => getSafe(() => selectedArtworkData.metadata, null), [selectedArtworkData]);
  const currentAnalysisData = useMemo(() => getSafe(() => selectedArtworkData.analysis, null), [selectedArtworkData]);

  // Determine Image URL (prioritize cached, then primary, then small)
  const originalImageUrl = useMemo(() => {
      if (!currentArtworkMetadata) return null;
      return currentArtworkMetadata.cachedImageUrl || currentArtworkMetadata.primaryImage || currentArtworkMetadata.primaryImageSmall || null;
  }, [currentArtworkMetadata]);

  // Get Map URLs from analysis data
  const attentionMapUrl = useMemo(() => getSafe(() => currentAnalysisData.attention.map_url, null), [currentAnalysisData]);
  const gradcamMapUrl = useMemo(() => getSafe(() => currentAnalysisData.gradcam.map_url, null), [currentAnalysisData]); // Assuming gradcam is structured similarly

  console.log("Derived Data:", { originalImageUrl, attentionMapUrl, gradcamMapUrl });

  // Check if this is an uploaded image (might be relevant for map availability)
  const isUploadedImage = getSafe(() => currentArtworkMetadata.type === 'uploaded', false);

  // --- Helper function to get proxied URL --- (Keep as is)
  function getProxiedImageUrl(url) {
    if (!url) return '';
    if (url.startsWith('data:')) return url; // Return base64 directly
    const BACKEND_URL = 'http://localhost:5000'; // Ensure this is accessible
    if (url.startsWith('http') && !url.includes(window.location.hostname)) {
        try {
             return `${BACKEND_URL}/api/proxy_image?url=${encodeURIComponent(url)}`;
        } catch (err) {
            console.error("Error encoding URL for proxy:", err);
            return '/placeholder-image.png'; // Fallback on encoding error
        }
    }
    // If it's a relative URL or already proxied, return as is
    return url;
  }
  // --- End Helper --- 

  const proxiedImageUrl = useMemo(() => getProxiedImageUrl(originalImageUrl), [originalImageUrl]);

  // Determine if we have the appropriate map for the current visualization type
  const hasMapForCurrentVisualization = useMemo(() => {
      return visualizationType === 'attention' ? !!attentionMapUrl : !!gradcamMapUrl;
  }, [visualizationType, attentionMapUrl, gradcamMapUrl]);

  // Determine if this is a comparison or overlay view (Keep as is)
  const isComparisonView = viewMode === 'comparison';

  // Get the appropriate map URL based on visualization type
  const currentMapUrl = useMemo(() => {
      return visualizationType === 'attention' ? attentionMapUrl : gradcamMapUrl;
  }, [visualizationType, attentionMapUrl, gradcamMapUrl]);

  // --- Handlers (Keep existing handlers: handleOverlayOpacityChange, handleVisualizationTypeChange, handleViewModeChange) ---
  function handleOverlayOpacityChange(_, newValue) {
    setOverlayOpacity(newValue);
  }

  const handleVisualizationTypeChange = (_, newValue) => {
    if (newValue) setVisualizationType(newValue);
  };

  const handleViewModeChange = (_, newValue) => {
    if (newValue) setViewMode(newValue);
  };
  // --- End Handlers --- 

  // --- Overlay Styles Calculation (Adjust to use currentMapUrl) ---
  const overlayStyles = useMemo(() => ({
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundImage: currentMapUrl ? `url(${getProxiedImageUrl(currentMapUrl)})` : 'none', // Proxy the map URL too
      backgroundSize: 'contain', // Use contain to match image scaling
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat',
      opacity: overlayOpacity,
      // mixBlendMode: 'screen' // 'screen' or 'multiply' can work, depending on map colors
      mixBlendMode: 'overlay' // Let's try 'overlay' blend mode
  }), [currentMapUrl, overlayOpacity]);
  // --- End Overlay Styles --- 

  // --- Loading/Initial State (Check metadata instead of artwork) ---
  if (!currentArtworkMetadata) {
    return (
      <Container sx={{ padding: 2 }}>
        <Paper sx={{ padding: 3, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary">
            Select an artwork to view attention visualizations.
          </Typography>
        </Paper>
      </Container>
    );
  }
  // --- End Loading/Initial State --- 

  return (
    <Container sx={{ padding: 2 }}>
      <Paper sx={{ padding: 2 }}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="h5" component="h2" gutterBottom>
              {currentArtworkMetadata.title || 'Artwork'} - {visualizationType === 'attention' ? 'Attention Map Visualization' : 'GradCAM Visualization'}
            </Typography>
          </Grid>

          {/* Controls (Keep as is) */}
          <Grid item xs={12} container spacing={2} alignItems="center">
             <Grid item>
               <ToggleButtonGroup
                 value={visualizationType}
                 exclusive
                 onChange={handleVisualizationTypeChange}
                 aria-label="visualization type"
                 size="small"
               >
                 <ToggleButton value="attention" aria-label="attention map" disabled={!attentionMapUrl}>
                   Attention Map
                 </ToggleButton>
                 <ToggleButton value="gradcam" aria-label="gradcam" disabled={!gradcamMapUrl}>
                   GradCAM
                 </ToggleButton>
               </ToggleButtonGroup>
             </Grid>

             <Grid item>
               <ToggleButtonGroup
                 value={viewMode}
                 exclusive
                 onChange={handleViewModeChange}
                 aria-label="view mode"
                 size="small"
               >
                 <ToggleButton value="comparison" aria-label="comparison view">
                   Comparison
                 </ToggleButton>
                 <ToggleButton value="overlay" aria-label="overlay view">
                   Overlay
                 </ToggleButton>
               </ToggleButtonGroup>
             </Grid>

             {viewMode === 'overlay' && (
               <Grid item xs={12} sm={4}>
                 <Typography id="opacity-slider" gutterBottom>
                   Overlay Opacity
                 </Typography>
                 <Slider
                   value={overlayOpacity}
                   onChange={handleOverlayOpacityChange}
                   aria-labelledby="opacity-slider"
                   step={0.05}
                   marks
                   min={0}
                   max={1}
                 />
               </Grid>
             )}
          </Grid>

          {/* Explanations Section (Keep as is) */}
          <Grid item xs={12}>
            <Accordion defaultExpanded>
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls="panel1a-content"
                id="panel1a-header"
                sx={{ backgroundColor: 'rgba(0, 0, 0, 0.03)' }}
              >
                <Typography variant="subtitle1"><InfoIcon sx={{ verticalAlign: 'middle', mr: 1, fontSize: '1.1rem' }} /> What are these visualizations?</Typography>
              </AccordionSummary>
              <AccordionDetails>
                 {/* Existing explanation text goes here... */}
                <Typography variant="h6" gutterBottom>Attention Maps</Typography>
                <Typography paragraph variant="body2">
                  <b>What you see:</b> This visualization often appears as a heatmap overlaid on the original image or shown side-by-side. 
                  It highlights regions the AI model (specifically, a Vision Transformer or similar attention-based model) concentrated on during processing.
                  Think of it as mapping the model's "gaze" as it analyzes the image.
                </Typography>
                <Typography paragraph variant="body2">
                  <b>Interpreting the colors:</b> Areas with brighter or "hotter" colors (often reds and yellows in typical heatmaps) represent regions where the model placed higher attention. 
                  Cooler colors (blues and greens) indicate lower attention. High attention areas are those the model deemed most important for its internal representation or understanding of the image, 
                  which could relate to salient objects, complex textures, or defining stylistic elements.
                </Typography>
                
                <Typography variant="h6" gutterBottom>GradCAM (Gradient-weighted Class Activation Mapping)</Typography>
                <Typography paragraph variant="body2">
                  <b>What you see:</b> GradCAM also produces a heatmap, typically overlaid on the original image, pinpointing areas crucial for a specific model decision. 
                  It answers the question: "Which parts of the image most strongly support the model classifying it in a certain way?" 
                  (Note: In this application, the "classification" might be implicit, relating to the features extracted for analysis rather than a specific label).
                </Typography>
                <Typography paragraph variant="body2">
                  <b>Interpreting the colors:</b> Similar to attention maps, hotter colors (reds/yellows) signify regions that had the *most positive influence* on the model activating a particular internal feature or reaching its final output. 
                  These are the pixels that provide the strongest evidence for the model's interpretation. Cooler colors contributed less or not at all to that specific activation. 
                  It helps identify the discriminative regions used by the model.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Both visualizations help in understanding the AI's internal reasoning process and identifying which features of the artwork the model considers significant.
                </Typography>
              </AccordionDetails>
            </Accordion>
          </Grid>

          {/* Message when no maps are available (Use hasMapForCurrentVisualization) */}
          {!hasMapForCurrentVisualization && (
            <Grid item xs={12}>
              <Alert severity="info" icon={<ImageIcon />} sx={{ mt: 2 }}>
                <AlertTitle>Visualization Not Available</AlertTitle>
                {visualizationType === 'attention'
                  ? "Attention map data is not available for the selected artwork."
                  : "GradCAM data is not available for the selected artwork."}
                 This might be because the analysis hasn't been run or didn't produce this type of map.
              </Alert>
            </Grid>
          )}

          {/* Visualization Area */}
          {hasMapForCurrentVisualization && (
             <Grid item xs={12} container spacing={2} mt={1}>
                {/* Comparison View */}
                {isComparisonView && (
                    <>
                        <Grid item xs={12} sm={6}>
                           <Typography variant="subtitle2" align="center" gutterBottom>Original Image</Typography>
                           <Box sx={{ 
                                width: '100%', 
                                height: { xs: 300, sm: 400, md: 450 }, // Responsive fixed height 
                                display: 'flex', 
                                justifyContent: 'center', 
                                alignItems: 'center', 
                                position: 'relative', 
                                backgroundColor: '#f0f0f0', // Lighter background
                                borderRadius: 1, // Add slight rounding
                                overflow: 'hidden' // Ensure image stays within bounds
                           }}>
                                <img 
                                    src={proxiedImageUrl}
                                    alt="Original Artwork" 
                                    style={{ width: 'auto', height: '100%', maxWidth: '100%', objectFit: 'contain' }} // Adjust style for height limit
                                    onError={(e) => { 
                                        console.error(`AttentionVisualizer: Error loading ORIGINAL image. src attempted: ${proxiedImageUrl}`);
                                        e.target.src='/placeholder-image.png'; 
                                        e.target.alt='Error loading original' 
                                    }}
                                />
                           </Box>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                           <Typography variant="subtitle2" align="center" gutterBottom>{visualizationType === 'attention' ? 'Attention Map' : 'GradCAM'}</Typography>
                            {/* --- ADJUSTED BOX STYLE --- */}
                           <Box sx={{ 
                                width: '100%', 
                                height: { xs: 300, sm: 400, md: 450 }, // Match original image height 
                                display: 'flex', 
                                justifyContent: 'center', 
                                alignItems: 'center',
                                position: 'relative', 
                                backgroundColor: '#f0f0f0',
                                borderRadius: 1, 
                                overflow: 'hidden' 
                            }}>
                                <img 
                                    src={getProxiedImageUrl(currentMapUrl)} // Proxy the map URL
                                    alt={visualizationType === 'attention' ? 'Attention Map' : 'GradCAM'} 
                                    style={{ width: 'auto', height: '100%', maxWidth: '100%', objectFit: 'contain' }} // Adjust style for height limit
                                    onError={(e) => { 
                                        console.error(`AttentionVisualizer: Error loading MAP image. Type: ${visualizationType}, src attempted: ${getProxiedImageUrl(currentMapUrl)}`);
                                        e.target.src='/placeholder-image.png'; 
                                        e.target.alt='Error loading map' 
                                    }}
                                />
                           </Box>
                        </Grid>
                    </>
                )}

                {/* Overlay View */}
                {!isComparisonView && (
                     // Center the overlay view more effectively
                    <Grid item xs={12} display="flex" justifyContent="center">
                        <Box sx={{ 
                            width: '100%', 
                            maxWidth: { xs: '95%', sm: 500, md: 600 }, // Max width for overlay 
                            height: 'auto', // Let aspect ratio determine height
                            aspectRatio: '1 / 1', // Keep aspect ratio for the container
                            position: 'relative', 
                            backgroundColor: '#f0f0f0',
                            borderRadius: 1,
                            overflow: 'hidden'
                        }}> 
                           <Typography variant="subtitle2" align="center" gutterBottom sx={{position: 'absolute', top: 8, left: 8, zIndex: 2, backgroundColor: 'rgba(255,255,255,0.7)', px: 1, borderRadius: 1}}>Overlay View</Typography>
                            {/* Base Image */}
                            <img 
                                src={proxiedImageUrl}
                                alt="Original Artwork with Overlay" 
                                style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                                onError={(e) => { 
                                    console.error(`AttentionVisualizer: Error loading ORIGINAL image for OVERLAY. src attempted: ${proxiedImageUrl}`);
                                    e.target.src='/placeholder-image.png'; 
                                    e.target.alt='Error loading original for overlay' 
                                }}
                            />
                            {/* Overlay Map */}
                            {currentMapUrl && (
                               <Box 
                                  sx={overlayStyles} 
                                  // Add onError directly to the Box with backgroundImage is not standard
                                  // We rely on the comparison view's image load check or the hasMapForCurrentVisualization flag
                               />
                            )}
                       </Box>
                    </Grid>
                )}
             </Grid>
          )}

        </Grid>
      </Paper>
    </Container>
  );
};

export default AttentionVisualizer;