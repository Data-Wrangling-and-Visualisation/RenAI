import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Box, Paper, Typography, Grid, Tabs, Tab, 
  CircularProgress, Chip, Slider, CardMedia, Card, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Divider, List, ListItem, ListItemText, TextField, Container, CardContent, Button, IconButton,
  Accordion, AccordionSummary, AccordionDetails, ListItemAvatar, Avatar, ToggleButtonGroup, ToggleButton,
  Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Snackbar, LinearProgress,
  Select, MenuItem
} from '@mui/material';
import { Radar } from 'react-chartjs-2';
import { Doughnut } from 'react-chartjs-2';
import { PolarArea } from 'react-chartjs-2';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend, ArcElement } from 'chart.js';
import RefreshIcon from '@mui/icons-material/Refresh';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import InfoIcon from '@mui/icons-material/Info';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ViewModuleIcon from '@mui/icons-material/ViewModule';
import ViewListIcon from '@mui/icons-material/ViewList';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import CloseIcon from '@mui/icons-material/Close';
import ImageIcon from '@mui/icons-material/Image';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';

// Registration of Chart.js components
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend, ArcElement);

// Default empty state for charts to prevent errors before data loads
const emptyChartData = { labels: [], datasets: [] };

// New component to display the current image being analyzed
const CurrentImageDisplay = ({ selectedArtworkData }) => {
  // --- DEBUGGING ---
  console.log("CurrentImageDisplay: Received selectedArtworkData:", selectedArtworkData);
  // --- END DEBUGGING ---

  if (!selectedArtworkData) {
    console.log("CurrentImageDisplay: selectedArtworkData is null or undefined");
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300, border: '1px dashed grey' }}>
        <Typography variant="body2" color="text.secondary">
          Select an artwork to view analysis
        </Typography>
      </Box>
    );
  }
  
  // For uploaded items, selectedArtworkData structure is { metadata: {...}, analysis: {...} }
  const artwork = selectedArtworkData.metadata || selectedArtworkData;
  // --- DEBUGGING ---
  console.log("CurrentImageDisplay: Derived artwork object:", artwork);
  // --- END DEBUGGING ---

  // Refactor to handle all image URL cases in one place
  const getImageUrl = (artwork) => {
    if (!artwork) return '/placeholder-image.png';
    
    // For uploaded images with base64 data
    if (artwork.type === 'uploaded' && typeof artwork.primaryImageSmall === 'string' && 
        artwork.primaryImageSmall.startsWith('data:image')) {
      return artwork.primaryImageSmall;
    }
    
    // For cached images
    if (artwork.cachedImageUrl) {
      if (artwork.cachedImageUrl.startsWith('data:')) {
        return artwork.cachedImageUrl;
      }
      // Proxy non-data URLs
      try {
        return `http://localhost:5000/api/proxy_image?url=${encodeURIComponent(artwork.cachedImageUrl)}`;
      } catch (err) {
        console.error("Error proxying cachedImageUrl:", err);
      }
    }
    
    // For regular image URLs
    const imageUrl = artwork.primaryImageSmall || artwork.primaryImage;
    if (imageUrl) {
      if (typeof imageUrl === 'string' && imageUrl.startsWith('data:')) {
        return imageUrl;
      }
      // Proxy other URLs
      try {
        return `http://localhost:5000/api/proxy_image?url=${encodeURIComponent(imageUrl)}`;
      } catch (err) {
        console.error("Error proxying imageUrl:", err);
      }
    }
    
    return '/placeholder-image.png';
  };

  const imageUrl = getImageUrl(artwork);
  console.log("CurrentImageDisplay: Final imageUrl:", imageUrl ? (imageUrl.startsWith('data:') ? 'data:... (base64)' : imageUrl) : 'null');

  return (
    <Card sx={{ mb: 2 }}>
      {/* Use CardMedia for consistent image display */}
      <CardMedia
        component="img"
        sx={{
          height: 300, // Fixed height
          objectFit: 'contain', // Ensure the whole image is visible
          bgcolor: '#f0f0f0' // Background color for Letterboxing/Pillarboxing
        }}
        image={imageUrl}
        alt={artwork.title || 'Artwork Image'}
        onError={(e) => {
          console.error("CurrentImageDisplay: Error loading image:", imageUrl);
          e.target.onerror = null; // prevents looping
          e.target.src="/placeholder-image.png"; // Or some placeholder
          e.target.alt="Error loading image";
        }}
      />
      <CardContent>
        <Typography variant="h6" component="div" gutterBottom>
          {artwork.title || 'Untitled Artwork'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {artwork.artistDisplayName || 'Unknown Artist'} {artwork.objectDate ? `(${artwork.objectDate})` : ''}
        </Typography>
        {artwork.classification && (
          <Typography variant="body2" color="text.secondary">
            {artwork.classification}
          </Typography>
        )}
        {artwork.medium && (
          <Typography variant="body2" color="text.secondary">
            {artwork.medium}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

// Extended decoding of stylistic analysis parameters
const styleDescriptions = {
  'Linearity': 'The degree to which linear elements are utilized in the artwork. High values indicate clear contours and pronounced lines. Linear style is characteristic of graphics, drawing and some painting styles where the outlines of figures and objects are clearly defined.',
  'Colorfulness': 'A measure of the intensity and variety of colors. High values indicate bright, saturated colors with a wide range of colors. Low values indicate a muted, monochromatic or limited palette. High colorfulness is often found in Impressionist, Fauvist, and Pop Art works.',
  'Complexity': 'The level of detail and visual complexity of the composition. High values indicate a large number of details, elements and textural features. Complex works take more time to perceive and analyze, whereas simple works can be perceived instantly.',
  'Contrast': "The degree of difference between light and dark areas, as well as the contrast of colors. High values mean strong contrasts (as in Caravaggio's works or chiaroscuro style), low values mean soft tonal transitions and harmonious combination of close shades.",
  'Symmetry': 'The degree of visual balance and mirroring in a composition. High values indicate a more symmetrical structure where elements are evenly distributed about a central axis. Symmetry is often associated with a sense of order, stability and classical harmony.',
  'Texture': 'The expression of the surface qualities of an image. High values denote explicit texture (smears, roughness, relief of the paint layer). Texture can be either real (in oil painting) or simulated (in digital art or graphics).'
};

// Advanced decoding of compositional analysis parameters
const compositionDescriptions = {
  'Symmetry': 'Assesses the balance of the image along both axes. Higher values indicate a more symmetrical composition, where elements are balanced in relation to the center or axis. Symmetry creates a sense of formality, stability and classical order in a composition. It is characteristic of classical art, architectural elements and religious works.',
  'Rule of thirds': 'The conformity of a composition to the rule of thirds, where the image is mentally divided into 9 equal parts by two equidistant horizontal and two equidistant vertical lines. High values indicate the placement of key elements along these lines or at points where they intersect, creating a dynamic and natural composition.',
  'Leading Lines': "The presence and expression of guiding lines that lead the viewer's eye through the composition. High values mean clear guides (roads, rivers, diagonal elements, etc.) that help create depth and lead the eye to the main subject. An important element of composition in landscape and architectural photography and painting.",
  'Depth': 'A sense of space and multiplicity in the work. High values indicate a pronounced perspective, the use of plans (foreground, middle, background) and techniques to create the illusion of depth. Low values are characteristic of flat, decorative compositions or abstract art.',
  'Framing': "The use of framing elements within a composition. High values mean the pronounced framing of the main object by other elements of the image (tree branches, arches, tunnels, windows). This technique helps to direct the viewer's attention to the main object and create a sense of 'looking from outside'.",
  'Balance': 'The overall visual balance of the elements in a composition. High values indicate an even distribution of visual weight where the composition does not appear to be overloaded on one side. Balance can be symmetrical or asymmetrical, but balanced by contrasting size, color, and placement of elements.'
};

// Additional explanations for color analysis
const colorAnalysisDescriptions = {
  'Dominant Color': 'The most common color in an image, occupying the largest area. The dominant color sets the overall mood of the work and is key to the color composition.',
  'Color Harmony': 'An assessment of how harmoniously the colors in a piece of artwork work together. There are different types of harmonies: monochrome, analog, complementary, triad, etc.',
  'Saturation': 'The intensity or purity of a color. Highly saturated colors appear bright and intense, low saturated colors appear muted and soft.',
  'Brightness': 'The overall lightness level of a color. High brightness means lighter tones, low brightness means darker tones.',
  'Emotional Perception': 'Subjective evaluation of the emotions that a given color can evoke according to color psychology research.'
};

// Component for a detailed explanation of the analysis values
const AnalysisExplanations = ({ type }) => {
  const styleExplanations = [
    { 
      parameter: 'Linearity', 
      description: 'The degree to which linear elements are used in the work. High values indicate clear contours and pronounced lines.',
      details: 'Linear style is characteristic of graphics, drawing and some areas of painting, where the outlines of figures and objects are clearly defined. This style is often used in classical art, neoclassicism and academism. Low linearity may indicate an impressionist or expressionist approach, where the boundaries between objects are blurred.'
    },
    { 
      parameter: 'Colorfulness', 
      description: 'A measure of the intensity and variety of colors. High values indicate bright, saturated colors with a wide range of colors.',
      details: 'Low values indicate a muted, monochromatic or limited palette. High colorfulness is often found in Impressionist, Fauvist, and Pop Art works. Colorful works tend to attract more attention and may evoke a stronger emotional response.'
    },
    { 
      parameter: 'Complexity', 
      description: 'The level of detail and visual complexity of the composition.',
      details: 'High values indicate a large number of details, elements and textural features. Complex works take more time to perceive and analyze, whereas simple works can be perceived instantly. High complexity is often found in the work of Northern Renaissance, Baroque, and some strands of Surrealist artists.'
    },
    { 
      parameter: 'Contrast', 
      description: 'The degree of difference between light and dark areas, as well as the contrast of colors.', 
      details: 'High values mean strong contrasts (as in the works of Caravaggio or in the chiaroscuro style), low values mean soft tonal transitions and harmonious combination of close shades. Contrast is used to create dramatic effect, emphasize the main elements and create a sense of volume and depth.'
    },
    { 
      parameter: 'Symmetry', 
      description: 'The degree of visual balance and mirroring in a composition.',
      details: 'Higher values indicate a more symmetrical structure, where elements are evenly distributed about a central axis. Symmetry is often associated with a sense of order, stability and classical harmony. Symmetrical compositions are characteristic of religious art, classical architecture and formal portraits.'
    },
    { 
      parameter: 'Texture', 
      description: 'Expression of the surface qualities of an image.',
      details: 'High values mean explicit texture (strokes, roughness, relief of the paint layer). Texture can be either real (in oil painting) or simulated (in digital art or graphics). Artists use texture to create a tactile sensation, convey the materiality of objects, and add visual interest to the work.'
    }
  ];

  const colorExplanations = [
    { 
      parameter: 'Dominant color', 
      description: 'The most common color in the image, occupying the largest area.',
      details: 'The dominant color sets the overall mood of the work and is the key color in the color composition. Color psychology states that different colors can evoke different emotional responses: blue is often associated with calmness and reliability, red with energy and passion, green with nature and harmony.'
    },
    { 
      parameter: 'Color harmony', 
      description: 'An assessment of how harmoniously colors combine in a work of art.',
      details: 'There are different types of harmony: monochrome (different shades of one color), analog (colors located next to each other on the color wheel), complementary (opposite colors on the color wheel), triadic (three colors evenly distributed on the color wheel) and others. The type of harmony affects the emotional perception of the work and its visual unity.'
    },
    { 
      parameter: 'Saturation', 
      description: 'Intensity or purity of color.',
      details: 'Highly saturated colors look bright and intense, low-saturated colors look muted and soft. Color saturation affects emotional impact: bright saturated colors attract attention and create a sense of energy, while muted tones create a calmer, more sophisticated atmosphere.'
    },
    { 
      parameter: 'Brightness', 
      description: 'The overall lightness level of a color.',
      details: 'High brightness means lighter tones, low brightness means darker tones. Brightness affects the overall mood of the artwork: lighter tones are often associated with optimism, purity and openness, darker tones with mystery, depth and intensity.'
    },
    { 
      parameter: 'Emotional perception', 
      description: 'A subjective assessment of the emotions that a given color can evoke.',
      details: 'Based on color psychology research and cultural associations. For example, red may be associated with energy, passion or danger; blue with calmness, reliability or melancholy; green with nature, growth or envy, depending on the context and cultural background of the viewer.'
    },
    { 
      parameter: 'Emotional perception', 
      description: 'A subjective assessment of the emotions that a given color can evoke.',
      details: 'Based on color psychology research and cultural associations. For example, red may be associated with energy, passion, or danger; blue with calmness, reliability, or melancholy; and green with nature, growth, or envy, depending on the context and cultural background of the viewer.'
    },
    { 
      parameter: 'Contrast in monochrome images', 
      description: 'The degree of difference between the lightest and darkest parts of a monochrome image.',
      details: 'High contrast creates a dramatic effect with a clear separation between light and shadow, emphasizing shapes and textures. Low contrast creates a soft, nuanced image with smooth transitions between tones. Contrast in monochrome images is important for creating depth, volume and focus for the viewer.'
    },
    { 
      parameter: 'Low Key image', 
      description: 'Monochrome or color image dominated by dark tones and shadows.',
      details: 'Low Key images are characterized by the dominance of dark shades, creating an atmosphere of mystery, drama and intimacy. This technique is often used in portrait photography, film noir and Baroque painting (e.g. Rembrandt). Small patches of light against an overall dark background create strong accents and focal points.'
    },
    { 
      parameter: 'High Key image', 
      description: 'A monochrome or color image in which light tones predominate.',
      details: 'High Key images are characterized by the predominance of light tones and the minimal presence of shadows, creating a sense of lightness, spaciousness and optimism. This technique is often used in fashion photography, minimalist art and to convey an atmosphere of happiness or purity. High Key images tend to create a softer, airy impression with soft textures and gentle tonal transitions.'
    },
    { 
      parameter: 'Tonal range', 
      description: 'The overall range of brightness from the darkest to the lightest parts of an image.',
      details: 'The wide tonal range includes the full spectrum from deep black to bright white, creating a richness of detail in all areas of the image. A narrow tonal range concentrates on medium, dark or light tones, creating a specific mood and atmosphere. Tonal range is an important tool for artistic expression, especially in black-and-white photography and graphics.'
    }
  ];

  const compositionExplanations = [
    { 
      parameter: 'Symmetry', 
      description: 'An assessment of the balance of the image along both axes.',
      details: 'Higher values indicate a more symmetrical composition, with elements balanced on a center or axis. Symmetry creates a sense of formality, stability and classical order in a composition. It is characteristic of classical art, architectural elements and religious works.'
    },
    { 
      parameter: 'Rule of thirds', 
      description: 'The conformity of a composition to the rule of thirds, where the image is mentally divided into 9 equal parts by two equidistant horizontal and two equidistant vertical lines.',
      details: 'The image is divided by two equidistant horizontal lines and two equidistant vertical lines. High values indicate the location of key elements along these lines or at their intersection points, which creates a dynamic and natural composition. This principle is widely used in painting, photography, and cinematography to create visually appealing and balanced compositions.'
    },
    { 
      parameter: 'Leading lines', 
      description: 'The presence and expression of guiding lines that lead the viewer through the composition.',
      details: "High values mean clear guides (roads, rivers, diagonal elements, etc.) that help create depth and lead the eye to the main subject. An important element of composition in landscape and architectural photography and painting. Leading lines can be explicit (physical objects) or implied (created by the location of objects or the direction of characters' gazes)."
    },
    { 
      parameter: 'Depth', 
      description: 'A sense of space and multiplanarity in the work.',
      details: 'High values indicate a pronounced perspective, the use of plans (foreground, middle, background) and techniques to create the illusion of depth. Low values are characteristic of flat, decorative compositions or abstract art. Depth can be created by using linear perspective, aerial perspective (changing color and clarity with distance), overlapping objects, and changing the size of objects with distance.'
    },
    { 
      parameter: 'Framing', 
      description: 'Use of framing elements within the composition.',
      details: "High values mean a pronounced framing of the main object by other elements of the image (tree branches, arches, tunnels, windows). This technique helps to direct the viewer's attention to the main object and create a sense of 'looking out'. Framing can also add context and depth to an image, creating an 'image within an image.'"
    },
    { 
      parameter: 'Balance', 
      description: 'The overall visual balance of the elements in the composition.',
      details: 'High values indicate an even distribution of visual weight, where the composition does not appear overloaded on one side. The balance can be symmetrical or asymmetrical, but balanced by contrasting the size, color, and placement of elements. A well-balanced composition is perceived as harmonious and pleasing to the eye, even if it is asymmetrical.'
    }
  ];

  let explanations = [];
  if (type === 'style') {
    explanations = styleExplanations;
  } else if (type === 'color') {
    explanations = colorExplanations;
  } else if (type === 'composition') {
    explanations = compositionExplanations;
  }

  return (
    <Paper sx={{ p: 2, mb: 3, mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        <InfoIcon sx={{ verticalAlign: 'middle', mr: 1, color: 'primary.main' }} />
        Detailed explanation of the analysis parameters
      </Typography>
      
      {explanations.map((item, index) => (
        <Accordion key={index} sx={{ mb: 1 }}>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            sx={{ backgroundColor: 'rgba(0, 0, 0, 0.03)' }}
          >
            <Typography variant="subtitle1" fontWeight="medium">{item.parameter}</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Typography paragraph>{item.description}</Typography>
            <Typography variant="body2" color="text.secondary">{item.details}</Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Paper>
  );
};

const ImagePreview = ({ artwork, imageUrl }) => {
  // Safety check for invalid artwork
  if (!artwork) {
    return (
      <Box sx={{ 
        position: 'relative',
        width: '100%', 
        height: '100%', 
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#f5f5f5',
        borderRadius: 1,
        boxShadow: 1,
        overflow: 'hidden',
        mb: 2
      }}>
        <Typography variant="body2" color="text.secondary">
          Изображение недоступно
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ 
      position: 'relative',
      width: '100%', 
      height: '100%', 
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: '#f5f5f5',
      borderRadius: 1,
      boxShadow: 1,
      overflow: 'hidden',
      mb: 2
    }}>
      <img 
        src={imageUrl || '/placeholder-image.png'} 
        alt={artwork?.title || 'Artwork'}
        style={{ 
          maxWidth: '100%', 
          maxHeight: '100%', 
          objectFit: 'contain'
        }} 
        onError={(e) => {
          console.error("Image failed to load:", imageUrl);
          e.target.onerror = null; // prevents looping
          e.target.src = '/placeholder-image.png'; // Or some placeholder
          e.target.alt = 'Error loading image';
        }}
      />
      
      {artwork?.objectID && (
        <Box sx={{
          position: 'absolute',
          bottom: 0,
          right: 0,
          backgroundColor: 'rgba(0,0,0,0.6)',
          color: 'white',
          padding: '4px 8px',
          fontSize: '0.7rem',
          borderTopLeftRadius: 4
        }}>
          ID: {artwork.objectID}
        </Box>
      )}
    </Box>
  );
};

const MonochromeVisualization = ({ colorData, colorProperties, colorHarmony }) => {
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'medium' }}>
        <InfoIcon sx={{ verticalAlign: 'middle', mr: 1, color: 'primary.main' }} />
        Анализ монохромного изображения
      </Typography>
      
      <Typography variant="body2" paragraph>
        {colorHarmony?.description || 'Монохромное изображение использует только оттенки серого или одного цвета.'}
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            Распределение тонов:
          </Typography>
          <Box sx={{ height: 250 }}>
            <Doughnut 
              data={colorData}
              options={{
                plugins: {
                  legend: {
                    position: 'bottom',
                    display: true
                  }
                },
                cutout: '50%',
                responsive: true,
                maintainAspectRatio: true
              }}
            />
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            Характеристики изображения:
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableBody>
                <TableRow>
                  <TableCell><strong>Тип</strong></TableCell>
                  <TableCell>{colorHarmony?.monochrome_type || 'Монохромное'}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Уровень контраста</strong></TableCell>
                  <TableCell>{colorHarmony?.contrast_level || 'Средний'}</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Доминирующий тон</strong></TableCell>
                  <TableCell>
                    {colorProperties && colorProperties.length > 0 ? (
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{ 
                          width: 16, 
                          height: 16, 
                          backgroundColor: colorProperties[0]?.hex || '#888',
                          borderRadius: '2px',
                          marginRight: 1,
                          border: '1px solid #ddd'
                        }} />
                        {colorProperties[0]?.name} ({colorProperties[0]?.percentage}%)
                      </Box>
                    ) : 'Нет данных'}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Эмоциональное восприятие</strong></TableCell>
                  <TableCell>
                    {colorHarmony?.emotional_impact?.join(', ') || 'Классический, Сдержанный'}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Тональное распределение:
            </Typography>
            {colorProperties && colorProperties.length > 0 && (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {colorProperties.map((color, idx) => (
                  <Chip 
                    key={idx}
                    size="small"
                    label={`${color.name}: ${color.percentage}%`}
                    style={{
                      backgroundColor: color.hex,
                      color: color.is_dark ? 'white' : 'black',
                      border: '1px solid #ddd'
                    }}
                  />
                ))}
              </Box>
            )}
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

const StyleAnalysis = ({ 
  artworks, 
  embeddings, 
  selectedArtworkData, 
  showUploadDialog,
  onOpenUploadDialog,
  onCloseUploadDialog,
  onUploadedImageAnalysis,
  onArtworkSelect
}) => {
  // --- DEBUGGING: Log received selectedArtworkData ---
  console.log("StyleAnalysis: Received selectedArtworkData:", selectedArtworkData);
  // --- ADDED LOG: Log received artworks prop ---
  console.log("StyleAnalysis: Received artworks prop:", artworks);
  // --- END LOG ---
  if (selectedArtworkData && selectedArtworkData.metadata && selectedArtworkData.metadata.type === 'uploaded') {
    console.log("StyleAnalysis: Uploaded image metadata:", selectedArtworkData.metadata);
    console.log("StyleAnalysis: Uploaded image primaryImageSmall (first 100 chars):", selectedArtworkData.metadata.primaryImageSmall ? selectedArtworkData.metadata.primaryImageSmall.substring(0, 100) + '...' : 'null');
  }
  // --- END DEBUGGING ---

  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [displayedArtworks, setDisplayedArtworks] = useState([]); // Added this line back
  const [displayMode, setDisplayMode] = useState('list'); // Changed from 'grid' to 'list'
  const [selectedArtwork, setSelectedArtwork] = useState(null);
  const [styleData, setStyleData] = useState(emptyChartData);
  const [colorData, setColorData] = useState(emptyChartData);
  const [compositionData, setCompositionData] = useState(emptyChartData);
  const [colorProperties, setColorProperties] = useState([]);
  const [colorHarmony, setColorHarmony] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);
  const [error, setError] = useState(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [showStyleFallbackNotice, setShowStyleFallbackNotice] = useState(false);
  const [isStyleFallback, setIsStyleFallback] = useState(false);
  const [isColorFallback, setIsColorFallback] = useState(false);
  const [isCompositionFallback, setIsCompositionFallback] = useState(false);
  const [isMonochrome, setIsMonochrome] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedImageFile, setUploadedImageFile] = useState(null);
  const [uploadedImageAnalysis, setUploadedImageAnalysis] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [artworkTitle, setArtworkTitle] = useState('');
  const [artwork, setArtwork] = useState(null);
  const [showDetailedExplanation, setShowDetailedExplanation] = useState(false);
  const fileInputRef = useRef(null);
  
  const BACKEND_URL = 'http://localhost:5000';
  
  // --- ADD: Define clearAnalysisState function --- 
  const clearAnalysisState = useCallback(() => {
    console.log("Clearing analysis state...");
    setStyleData(emptyChartData);
    setColorData(emptyChartData);
    setCompositionData(emptyChartData);
    setColorProperties([]);
    setColorHarmony(null);
    setError(null); // Optionally clear errors too
    setIsStyleFallback(false);
    setIsColorFallback(false);
    setIsCompositionFallback(false);
    setIsMonochrome(false);
    // Don't clear selectedArtwork or searchTerm here, only analysis results
  }, []); // Empty dependency array as it uses setters directly
  // --- END ADD ---
  
  // Effect to load artworks
  useEffect(() => {
    if (artworks && artworks.length > 0) {
      setDisplayedArtworks(artworks);
    }
  }, [artworks]);
  
  // Function to get random artworks for display
  const refreshArtworks = useCallback(() => {
    if (!artworks || artworks.length === 0) return;
    
    // Take a subset of artworks or shuffle them for display
    const shuffled = [...artworks].sort(() => 0.5 - Math.random());
    const randomArtworks = shuffled.slice(0, Math.min(20, artworks.length));
    setDisplayedArtworks(randomArtworks);
  }, [artworks]);
  
  // Initialize displayed artworks when artworks array changes
  useEffect(() => {
    if (artworks && artworks.length > 0 && (!displayedArtworks || displayedArtworks.length === 0)) {
      refreshArtworks();
    }
  }, [artworks, displayedArtworks, refreshArtworks]);
  
  // Effect to update component state when selectedArtworkData prop changes
  useEffect(() => {
    console.log('StyleAnalysis: selectedArtworkData prop changed:', selectedArtworkData); // <--- ADDED LOG 6
    if (selectedArtworkData) {
      const artworkMetadata = selectedArtworkData.metadata || selectedArtworkData; // Handle both direct artwork and nested structure
      const analysisData = selectedArtworkData.analysis; // Analysis data
      
      console.log('StyleAnalysis: Derived artworkMetadata:', artworkMetadata);
      console.log('StyleAnalysis: Analysis data available:', !!analysisData, 
                  'Style:', !!analysisData?.style,
                  'Color:', !!analysisData?.color,
                  'Composition:', !!analysisData?.composition);

      // Синхронизируем внутреннее состояние selectedArtwork с внешним selectedArtworkData
      setSelectedArtwork(artworkMetadata);
      setArtwork(artworkMetadata);
      setArtworkTitle(artworkMetadata.title || `Work ${artworkMetadata.objectID || artworkMetadata.id || 'unknown'}`);
      
      if (artworkMetadata && artworkMetadata.type === 'uploaded') {
        // Handle uploaded image specific logic if needed
        console.log('StyleAnalysis: Using pre-analyzed data for uploaded image.');
        setUploadedImage(artworkMetadata.primaryImageSmall); // Show the uploaded image
        
        if (analysisData) {
          processAnalysisData(analysisData);
        } else {
          // Clear analysis if none is present for the uploaded image
          clearAnalysisState();
          // Добавим сообщение о том, что анализа нет
          setError("No analysis data available for this uploaded image. Try selecting a different artwork.");
        }
      } else if (artworkMetadata && (artworkMetadata.objectID || artworkMetadata.id)) {
        // Regular artwork selected
        setUploadedImage(null);
        setUploadedImageFile(null);
        setUploadedImageAnalysis(null);

        // Check if analysis data is already available in the prop
        if (analysisData) {
          console.log('StyleAnalysis: Using analysis data from prop');
          setError(null); // Clear any previous errors
          processAnalysisData(analysisData);
        } else {
          // If analysis data is not in the prop, fetch it (optional, could be handled by App.jsx)
          // fetchAnalysisData(artworkMetadata.objectID); // Example: Fetch if needed
          console.log('StyleAnalysis: No analysis data provided in prop, clearing state.');
          clearAnalysisState();
          // Добавим сообщение о том, что анализа нет
          setError("Analysis data not available for this artwork. The image is displayed but detailed analysis cannot be shown.");
        }
      } else {
        console.log('StyleAnalysis: selectedArtworkData changed, but no valid metadata or objectID found.');
        clearAnalysisState();
        setError("Invalid artwork data received. Cannot display analysis.");
      }
      
    } else {
      console.log('StyleAnalysis: selectedArtworkData is null, clearing state.');
      clearAnalysisState();
      setUploadedImage(null);
      setUploadedImageFile(null);
      setUploadedImageAnalysis(null);
      setError("No artwork selected. Please select an artwork from the list to view analysis.");
    }
  }, [selectedArtworkData, clearAnalysisState]); // Dependency array ensures this runs when selectedArtworkData changes

  // Function to process the analysis data (color, style, composition)
  const processAnalysisData = (analysisData) => {
    if (!analysisData) {
      clearAnalysisState();
      return;
    }

    // Process Color Data
    if (analysisData.color) {
      // *** CHANGE: Directly pass analysisData.color ***
      processColorData(analysisData.color);
    } else {
      setColorData(emptyChartData);
      setColorProperties([]);
      setColorHarmony(null);
      setIsMonochrome(false);
    }

    // Process style data
    if (analysisData.style) {
      const styleResult = analysisData.style;
      setIsStyleFallback(styleResult.is_fallback === true);
      setStyleData({
        labels: styleResult.labels || ['Linearity', 'Colorfulness', 'Complexity', 'Contrast', 'Symmetry', 'Texture'],
    datasets: [
      {
            label: 'Style profile',
            data: styleResult.values || [
              styleResult.linearity, styleResult.colorfulness, styleResult.complexity,
              styleResult.contrast, styleResult.symmetry, styleResult.texture
            ].filter(v => v !== undefined),
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgb(54, 162, 235)',
            pointBackgroundColor: 'rgb(54, 162, 235)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(54, 162, 235)'
          }
        ]
      });
    } else {
      setStyleData(emptyChartData);
    }

    // Process composition data
    if (analysisData.composition) {
      const compResult = analysisData.composition;
      setIsCompositionFallback(compResult.is_fallback === true);
      setCompositionData({
        labels: ['Symmetry', 'Rule of thirds', 'Leading lines', 'Depth', 'Framing', 'Balance'],
    datasets: [
      {
            label: 'Composition analysis',
            data: [
              compResult.symmetry, compResult.rule_of_thirds, compResult.leading_lines,
              compResult.depth, compResult.framing, compResult.balance
            ],
            backgroundColor: [
              'rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 'rgba(255, 206, 86, 0.5)',
              'rgba(75, 192, 192, 0.5)', 'rgba(153, 102, 255, 0.5)', 'rgba(255, 159, 64, 0.5)'
            ]
          }
        ]
      });
    } else {
      setCompositionData(emptyChartData);
    }
  };

  const fetchAnalysisData = async (objectId) => {
    if (!objectId) {
      console.error('Failed to execute query: objectId not defined');
      setError('Failed analysis: Work ID not defined');
      return;
    }
    
    setLoading(true);
    setError(null);
    setStyleData(emptyChartData);
    setColorData(emptyChartData);
    setCompositionData(emptyChartData);
    setColorProperties([]);
    setColorHarmony(null);
    
    try {
      console.log(`Requesting style analysis for objectId: ${objectId}`);
      
      // Force parameter to bypass cache
      const forceParam = "?force=true";
      
      // Fetch style analysis
      console.log(`Sending request: ${BACKEND_URL}/api/analyze/style/${objectId}${forceParam}`);
      const styleResponse = await fetch(`${BACKEND_URL}/api/analyze/style/${objectId}${forceParam}`);
      if (!styleResponse.ok) {
        console.error(`Style analysis failed with status: ${styleResponse.status}`);
        throw new Error(`Style analysis failed: ${styleResponse.statusText}`);
      }
      const styleResult = await styleResponse.json();
      console.log("Received style analysis results:", styleResult);
      setIsStyleFallback(styleResult.is_fallback === true);
      
      setStyleData({
        labels: ['Linearity', 'Colorfulness', 'Complexity', 'Contrast', 'Symmetry', 'Texture'],
    datasets: [
      {
            label: 'Style profile',
            data: [
              styleResult.linearity,
              styleResult.colorfulness,
              styleResult.complexity,
              styleResult.contrast,
              styleResult.symmetry,
              styleResult.texture
            ],
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgb(54, 162, 235)',
            pointBackgroundColor: 'rgb(54, 162, 235)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(54, 162, 235)'
          }
        ]
      });
      
      // Fetch color analysis
      console.log(`Отправка запроса: ${BACKEND_URL}/api/analyze/color/${objectId}${forceParam}`);
      const colorResponse = await fetch(`${BACKEND_URL}/api/analyze/color/${objectId}${forceParam}`);
      if (!colorResponse.ok) {
        console.error(`Color analysis failed with status: ${colorResponse.status}`);
        throw new Error(`Color analysis failed: ${colorResponse.statusText}`);
      }
      const colorResult = await colorResponse.json();
      console.log("Received color analysis results:", colorResult);
      setIsColorFallback(colorResult.is_fallback === true);
      
      // Process dominant colors
      if (colorResult.dominant_colors && colorResult.dominant_colors.length > 0) {
        const labels = colorResult.dominant_colors.map(c => c.name || `Цвет ${c.index || 0 + 1}`);
        const data = colorResult.dominant_colors.map(c => c.percentage);
        const backgroundColor = colorResult.dominant_colors.map(c => c.hex || c.rgb || '#CCCCCC');
        
        setColorData({
          labels: labels,
          datasets: [{
            data: data,
            backgroundColor: backgroundColor,
            borderColor: new Array(labels.length).fill('#ffffff'),
            borderWidth: 1
          }]
        });
        
        // Преобразуем данные dominant_colors, добавляя недостающие поля для совместимости
        const processedColorProperties = colorResult.dominant_colors.map(color => ({
          ...color,
          hex: color.hex || `rgb(${color.rgb?.[0] || 0}, ${color.rgb?.[1] || 0}, ${color.rgb?.[2] || 0})`,
          is_dark: color.is_dark !== undefined 
            ? color.is_dark 
            : (color.rgb?.[0] < 128 && color.rgb?.[1] < 128 && color.rgb?.[2] < 128),
          brightness: color.brightness !== undefined 
            ? color.brightness 
            : (color.rgb ? ((color.rgb[0] + color.rgb[1] + color.rgb[2]) / (3 * 255)) : 0.5),
          saturation: color.saturation !== undefined 
            ? color.saturation 
            : (colorResult.is_monochrome ? 0 : 0.5)
        }));
        
        setColorProperties(processedColorProperties);
        
        if (colorResult.color_harmony || colorResult.harmony) {
          const harmonyData = colorResult.color_harmony || colorResult.harmony;
          setColorHarmony({
            type: harmonyData.type || 'Unknown',
            description: harmonyData.description || 'No description',
            score: harmonyData.score !== undefined ? harmonyData.score : 0.5,
            emotional_impact: colorResult.emotional_impact || [],
            is_monochrome: colorResult.is_monochrome || false,
            monochrome_type: colorResult.monochrome_type || (colorResult.is_monochrome ? 'Monochrome' : null),
            contrast_level: colorResult.contrast_level || null,
            contrast_value: colorResult.contrast_value || null
          });
        }
      } else {
        console.warn("No dominant colors data available");
        setColorData(emptyChartData);
        setColorProperties([]);
        setColorHarmony(null);
      }
      
      // Fetch composition analysis
      console.log(`Sending request: ${BACKEND_URL}/api/analyze/composition/${objectId}${forceParam}`);
      const compositionResponse = await fetch(`${BACKEND_URL}/api/analyze/composition/${objectId}${forceParam}`);
      if (!compositionResponse.ok) {
        console.error(`Composition analysis failed with status: ${compositionResponse.status}`);
        throw new Error(`Composition analysis failed: ${compositionResponse.statusText}`);
      }
      const compositionResult = await compositionResponse.json();
      console.log("Received composition analysis results:", compositionResult);
      setIsCompositionFallback(compositionResult.is_fallback === true);
      
      setCompositionData({
        labels: ['Symmetry', 'Rule of thirds', 'Leading lines', 'Depth', 'Framing', 'Balance'],
    datasets: [
      {
            label: 'Composition analysis',
            data: [
              compositionResult.symmetry,
              compositionResult.rule_of_thirds,
              compositionResult.leading_lines,
              compositionResult.depth,
              compositionResult.framing,
              compositionResult.balance
            ],
        backgroundColor: [
              'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
              'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
              'rgba(255, 159, 64, 0.5)'
            ]
          }
        ]
      });
      
    } catch (err) {
      console.error('Error fetching analysis data:', err);
      setError(`Error fetching analysis data: ${err.message}`);
      
      // Set default empty states for all analysis types
      setStyleData(emptyChartData);
      setColorData(emptyChartData);
      setCompositionData(emptyChartData);
      setColorProperties([]);
      setColorHarmony(null);
    } finally {
      setLoading(false);
    }
  };
  
  // Function to open the upload dialog
  const handleOpenUploadDialog = () => {
    console.log('Opening upload dialog', showUploadDialog);
    try {
      onOpenUploadDialog();
    } catch (error) {
      console.error('Error opening upload dialog:', error);
    }
  };
  
  // Function to close the upload dialog
  const handleCloseUploadDialog = () => {
    // Clear upload state
    handleClearUploadedImage();
  };
  
  // Handle artwork selection
  const handleSelectArtwork = (artwork) => {
    if (!artwork) {
      console.error("Attempt to select null/undefined artwork");
      setError("Unable to select: artwork not defined");
      return;
    }
    
    try {
      console.log("Selecting artwork:", artwork);
      
      const objectId = artwork.objectID || artwork.id;
      if (!objectId) {
        console.error("Selected artwork has no ID", artwork);
        setError("Unable to analyze: Work ID not defined");
        return;
      }
      
      setSelectedArtwork(artwork);
      setArtwork(artwork);
      setArtworkTitle(artwork.title || `Work ${objectId}`);
      
      // Notify the parent App component to update the GlobalImagePanel
      // This uses the handleArtworkSelect function defined in App.jsx
      if (onArtworkSelect && typeof onArtworkSelect === 'function') {
        onArtworkSelect(objectId); // This call triggers App.jsx to fetch analysis data
      }
      
      // REMOVED: App.jsx now fetches analysis data via the onArtworkSelect callback.
      // fetchAnalysisData(objectId); 
    } catch (err) {
      console.error("Error selecting artwork:", err);
      setError(`Error selecting artwork: ${err.message}`);
    }
  };
  
  const getFilteredArtworks = () => {
    if (!displayedArtworks || displayedArtworks.length === 0) return [];
    
    return displayedArtworks.filter(artwork => {
      const title = (artwork.title || '').toLowerCase();
      const artist = (artwork.artistDisplayName || '').toLowerCase();
      const department = (artwork.department || '').toLowerCase();
      const searchMatch = 
        searchTerm === '' || 
        title.includes(searchTerm.toLowerCase()) || 
        artist.includes(searchTerm.toLowerCase());
      
      let typeMatch = true;
      
      if (filterType === 'paintings') {
        typeMatch = 
          department.includes('paint') || 
          department.includes('art') || 
          !title.toLowerCase().includes('chair') && 
          !title.toLowerCase().includes('furniture') && 
          !title.toLowerCase().includes('table') &&
          !title.toLowerCase().includes('cabinet');
      } else if (filterType === 'colorful') {
        typeMatch = 
          !title.includes('gray') && 
          !title.includes('grey') && 
          !title.includes('black and white') &&
          !title.includes('monochrome');
      }
      
      return searchMatch && typeMatch;
    });
  };
  
  const filteredArtworks = getFilteredArtworks();
  
  const handleClearUploadedImage = () => {
    setUploadedImage(null);
    setUploadedImageFile(null);
    setUploadedImageAnalysis(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Reset upload state when the dialog is opened
  useEffect(() => {
    if (showUploadDialog) {
      setUploadedImage(null);
      setUploadedImageFile(null);
      setUploadedImageAnalysis(null);
      setUploadError(null);
      setUploadLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [showUploadDialog]);
  
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setUploadError('Please upload an image in JPEG, PNG, GIF, or WebP format');
      return;
    }
    
    const maxSizeInBytes = 10 * 1024 * 1024;
    if (file.size > maxSizeInBytes) {
      setUploadError('File size exceeds 10 MB. Please select a smaller file.');
      return;
    }
    
    setUploadedImageFile(file);
    setUploadError(null);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target.result);
    };
    reader.readAsDataURL(file);
  };
  
  const handleAnalyzeUploadedImage = async () => {
    if (!uploadedImageFile) {
      setUploadError('Please upload an image for analysis');
      return;
    }
    
    setUploadLoading(true);
    setUploadError(null);
    console.log('Sending image for analysis...');
    
    try {
      const formData = new FormData();
      formData.append('image', uploadedImageFile);
      
      console.log(`Uploading image: ${uploadedImageFile.name}, size: ${uploadedImageFile.size} bytes, type: ${uploadedImageFile.type}`);
      
      const response = await fetch(`${BACKEND_URL}/api/analyze-uploaded-image`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header as it will be set automatically with the correct boundary for FormData
      });
      
      console.log(`Upload response status: ${response.status}`);
      
      if (!response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Image analysis error');
        } else {
          const errorText = await response.text();
          throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
      }
      
      const data = await response.json();
      console.log('Received analysis results:', data);
      setUploadedImageAnalysis(data);
      
      // Pass the uploaded image data to the parent component
      if (onUploadedImageAnalysis) {
        onUploadedImageAnalysis(uploadedImage, data);
      }
      
      // Close the upload dialog
      onCloseUploadDialog();
      
      // Process color analysis data
      if (data.color) {
        processColorData(data.color);
      }
      
      // Process style analysis data
      if (data.style) {
        setStyleData({
          labels: data.style.labels,
    datasets: [
      {
              label: 'Style analysis',
              data: data.style.values,
              backgroundColor: 'rgba(54, 162, 235, 0.5)',
              borderColor: 'rgba(54, 162, 235, 1)',
              borderWidth: 1,
            }
          ]
        });
      }
      
      // Process composition analysis data
      if (data.composition) {
        setCompositionData({
          labels: ['Symmetry', 'Rule of thirds', 'Leading lines', 'Depth', 'Framing', 'Balance'],
          datasets: [
            {
              label: 'Composition analysis',
              data: [
                data.composition.symmetry,
                data.composition.rule_of_thirds,
                data.composition.leading_lines,
                data.composition.depth,
                data.composition.framing,
                data.composition.balance
              ],
        backgroundColor: [
                'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
                'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
                'rgba(255, 159, 64, 0.5)'
              ]
            }
          ]
        });
      }
      
      // Показываем уведомление об успешном анализе
      setSnackbarMessage('Image successfully analyzed!');
      setSnackbarOpen(true);
    } catch (err) {
      console.error('Error analyzing uploaded image:', err);
      setUploadError(`Error analyzing: ${err.message}`);
    } finally {
      setUploadLoading(false);
    }
  };
  
  // Helper function to process color analysis data
  const processColorData = (colorResult) => {
    // --- Keep this log active for debugging --- 
    console.log("processColorData: Received colorResult:", JSON.stringify(colorResult, null, 2)); 
    // --- 

    // Reset states initially
    setColorData(emptyChartData);
    setColorProperties([]);
    setColorHarmony(null);
    setIsMonochrome(false);

    if (!colorResult) {
      console.warn("processColorData: Received null or undefined colorResult.");
      return; // Exit if no data
    }

    setIsColorFallback(colorResult.is_fallback === true); // Update fallback status

    if (colorResult.is_monochrome === true) {
      // ... (monochrome handling remains the same) ...
      console.log("processColorData: Processing as monochrome.");
      setIsMonochrome(true);

      if (colorResult.grayscale_distribution && colorResult.grayscale_distribution.length > 0) {
        const monochromeData = {
          labels: colorResult.grayscale_distribution.map(item => item.shade || 'Shade'),
          datasets: [
            {
              label: 'Grayscale distribution',
              data: colorResult.grayscale_distribution.map(item => Number(item.percentage || 0)), // Ensure data is number
              backgroundColor: colorResult.grayscale_distribution.map(item => item.color_code || '#cccccc'),
              borderColor: '#ffffff',
              borderWidth: 1,
            }
          ]
        };
        setColorData(monochromeData);

        const monoProps = colorResult.grayscale_distribution.map(item => ({
            name: item.shade || 'Shade',
            hex: item.color_code || '#cccccc',
            percentage: Number(item.percentage || 0),
            brightness: '-', // Not applicable
            saturation: '-',
        }));
        setColorProperties(monoProps);

      } else {
        console.warn("processColorData: Monochrome detected, but grayscale_distribution is missing or empty.");
      }

      if (colorResult.color_harmony) {
         setColorHarmony({
           type: colorResult.color_harmony.type || 'Monochromatic',
           description: colorResult.color_harmony.description || 'Gray harmony.',
           score: colorResult.color_harmony.score,
           emotional_impact: colorResult.color_harmony.emotional_impact || [],
           monochrome_type: colorResult.monochrome_type || 'Monochrome',
           contrast_level: colorResult.contrast_level || null,
         });
      } else {
         setColorHarmony({
           type: 'Monochromatic',
           description: 'The image uses shades of gray or a single color.',
         });
      }

    } else if (colorResult.dominant_colors && colorResult.dominant_colors.length > 0) {
      console.log("processColorData: Processing as color image.");
      setIsMonochrome(false);

      // --- Re-check Doughnut chart data preparation --- 
      // Filter for colors that have percentage and some form of color code
      const validDominantColors = colorResult.dominant_colors.filter(
          color => color && 
                   color.percentage !== undefined && 
                   (color.hex || color.color_code || color.color)
      );

      if (validDominantColors.length > 0) {
          const colorChartData = {
            // Labels: Prioritize name, fallback to hex/code or generic
            labels: validDominantColors.map(color => String(color.name || color.hex || color.color_code || color.color || 'Цвет')),
            datasets: [
              {
                label: 'Dominant colors',
                // Data: Ensure percentage is a number
                data: validDominantColors.map(color => Number(color.percentage || 0)),
                // BackgroundColor: Prioritize hex, then color_code, then color field, then fallback
                backgroundColor: validDominantColors.map(color => color.hex || color.color_code || color.color || '#cccccc'),
                borderColor: '#ffffff', // White border for separation
                borderWidth: 1,
              }
            ]
          };
          console.log("processColorData: Setting colorChartData:", colorChartData);
          setColorData(colorChartData); // Set state for the Doughnut chart
      } else {
          console.warn("processColorData: dominant_colors array exists but contains no valid items with percentages.");
          setColorData(emptyChartData); // Reset chart data if no valid colors
      }
      // --- End Re-check ---

      // --- Table data preparation (should be correct from previous fix) ---
      // Adapt to handle potentially missing fields more gracefully
      const propertiesForTable = colorResult.dominant_colors.map(color => ({
        // Hex: Prioritize hex, then color_code, then color, then fallback
        hex: color.hex || color.color_code || color.color || '#CCCCCC',
        // Name: Use if available, otherwise fallback
        name: color.name || 'Unknown',
        percentage: color.percentage !== undefined ? Number(color.percentage) : 0,
        // Brightness: Use if available (assuming 0-1 range), format, else fallback
        brightness: color.brightness !== undefined ? `${Math.round(color.brightness * 100)}%` : '-',
        // Saturation: Use if available (assuming 0-1 range), format, else fallback
        saturation: color.saturation !== undefined ? `${Math.round(color.saturation * 100)}%` : '-',
      }));
      console.log("processColorData: Setting colorProperties (for table):", propertiesForTable);
      setColorProperties(propertiesForTable);
      // --- End Table data ---

      // --- Harmony data (should be correct) ---
      if (colorResult.color_harmony) {
        console.log("processColorData: Setting colorHarmony:", colorResult.color_harmony);
        setColorHarmony(colorResult.color_harmony);
      } else {
        console.warn("processColorData: color_harmony data is missing.");
        setColorHarmony(null);
      }
      // --- End Harmony data ---

    } else {
      console.warn("processColorData: Fallback - Neither monochrome nor valid dominant colors found.");
      // States already reset at the beginning
    }
  };
  
  // Обработчик закрытия сообщения
  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };
  
  // Переопределение функции для получения URL изображения
  const getImageUrl = (artwork) => {
    // Use a default placeholder if artwork is invalid
    const placeholder = '/placeholder-image.png';
    if (!artwork) {
        console.warn("getImageUrl: Called with null or undefined artwork.");
        return placeholder;
    }

    // Define BACKEND_URL locally or ensure it's available in scope
    const BACKEND_URL = 'http://localhost:5000';

    // 1. Check for cached image URL first (most reliable source)
    if (artwork.cachedImageUrl) {
        console.log("getImageUrl: Using cachedImageUrl:", artwork.cachedImageUrl.substring(0, 100) + '...');
        // If it's a direct data URL, return it. Otherwise, assume it needs proxy (though caching *should* store base64).
        // Safety check: if it's not a data URL, proxy it.
        if (artwork.cachedImageUrl.startsWith('data:image')) {
            return artwork.cachedImageUrl;
        } else if (artwork.cachedImageUrl.startsWith('http')) {
             // Proxy potentially cached external URLs just in case
             try {
               return `${BACKEND_URL}/api/proxy_image?url=${encodeURIComponent(artwork.cachedImageUrl)}`;
             } catch (err) {
               console.error("getImageUrl: Error encoding cached http URL:", err);
               return placeholder;
             }
        } else {
             // If it's neither data nor http, it might be an invalid cache entry
             console.warn("getImageUrl: cachedImageUrl is not a recognizable URL format:", artwork.cachedImageUrl.substring(0, 100) + '...');
             // Fall through to other checks
        }
    }

    // 2. Handle uploaded image data structure (metadata wrapper)
    if (artwork.metadata?.type === 'uploaded' && artwork.metadata?.primaryImageSmall?.startsWith('data:image')) {
        console.log("getImageUrl: Using base64 from metadata.primaryImageSmall for uploaded image.");
        return artwork.metadata.primaryImageSmall;
    }

    // 3. Handle uploaded image data structure (direct object)
    if (artwork.type === 'uploaded' && artwork.primaryImageSmall?.startsWith('data:image')) {
        console.log("getImageUrl: Using base64 from primaryImageSmall for top-level uploaded image.");
        return artwork.primaryImageSmall;
    }
    
    // --- Determine the correct object to check for API image URLs ---
    // If selectedArtworkData has a metadata object, use that. Otherwise, use the artwork object directly.
    const sourceObject = artwork.metadata || artwork;

    // 4. Check primaryImageSmall and primaryImage from the source object (likely needs proxy)
    const apiImageUrl = sourceObject.primaryImageSmall || sourceObject.primaryImage;
    if (apiImageUrl && typeof apiImageUrl === 'string' && apiImageUrl.startsWith('http')) {
        console.log(`getImageUrl: Using API URL (proxied): ${apiImageUrl}`);
        try {
            return `${BACKEND_URL}/api/proxy_image?url=${encodeURIComponent(apiImageUrl)}`;
        } catch (err) {
            console.error("getImageUrl: Error encoding API image URL:", err);
            return placeholder;
        }
    }

    // 5. Fallback if no valid URL is found
    console.warn("getImageUrl: No valid image URL found for artwork:", artwork);
    return placeholder;
  };
  
  // Функция для обработки Drag & Drop
  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  
  const handleDragEnter = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  
  const handleDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  
  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      const file = event.dataTransfer.files[0];
      
      const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
      if (!allowedTypes.includes(file.type)) {
        setUploadError('Please upload an image in JPEG, PNG, GIF, or WebP format');
        return;
      }
      
      const maxSizeInBytes = 10 * 1024 * 1024;
      if (file.size > maxSizeInBytes) {
        setUploadError('File size exceeds 10 MB. Please select a smaller file.');
        return;
      }
      
      setUploadedImageFile(file);
      setUploadError(null);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };
  
  if (!artworks || artworks.length === 0) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6">No available artworks for analysis</Typography>
      </Paper>
    );
  }
  
  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Style Analysis
        <IconButton 
          onClick={refreshArtworks} 
          color="primary" 
          aria-label="refresh artworks"
          sx={{ ml: 2 }}
        >
          <RefreshIcon />
        </IconButton>
        <Button
          variant="contained"
          color="primary"
          startIcon={<FileUploadIcon />}
          onClick={handleOpenUploadDialog}
          sx={{ ml: 2 }}
        >
          Upload image
        </Button>
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>
      )}
      
      {/* Current Image Display */}
      {!uploadedImageAnalysis && <CurrentImageDisplay selectedArtworkData={selectedArtworkData} />}
      
      {/* Uploaded Image Analysis Results */}
      {uploadedImageAnalysis && (
        <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            <ImageIcon sx={{ verticalAlign: 'middle', mr: 1, color: 'primary.main' }} />
            Uploaded image
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, alignItems: 'center', gap: 2 }}>
            <Box sx={{ 
              width: { xs: '100%', sm: '200px' }, 
              height: '180px', 
              display: 'flex', 
              justifyContent: 'center',
              alignItems: 'center',
              backgroundColor: '#f5f5f5',
              borderRadius: 1,
            }}>
              <img 
                src={uploadedImage} 
                alt="Uploaded artwork"
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '100%', 
                  objectFit: 'contain'
                }} 
              />
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6">Uploaded image</Typography>
              <Typography variant="body2" color="text.secondary">
                Analysis results below
              </Typography>
              <Button 
                variant="outlined" 
                color="error" 
                size="small"
                startIcon={<CloseIcon />}
                onClick={() => {
                  setUploadedImageAnalysis(null);
                  setUploadedImage(null);
                  setUploadedImageFile(null);
                }}
                sx={{ mt: 2 }}
              >
                Clear and return to gallery
              </Button>
            </Box>
          </Box>
        </Paper>
      )}
      
      {/* Upload Dialog */}
      <Dialog 
        open={showUploadDialog} 
        onClose={onCloseUploadDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Upload image for analysis</DialogTitle>
        <DialogContent>
          <DialogContentText paragraph>
            Upload an image to perform a full style analysis. 
            Supported formats: JPEG, PNG, GIF, WebP. Maximum size: 10 MB.
          </DialogContentText>
          
          <Box sx={{ mt: 2, mb: 2 }}>
            <Box
              sx={{
                border: '2px dashed #ccc',
                borderRadius: 2,
                p: 3,
                textAlign: 'center',
                mb: 2,
                backgroundColor: '#f9f9f9',
                cursor: 'pointer',
                position: 'relative',
                height: 200,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center'
              }}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {!uploadedImage ? (
                <>
                  <FileUploadIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                  <Typography variant="body1" gutterBottom>
                    Drag and drop an image here or click to upload
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    JPEG, PNG, GIF or WebP, max. 10 MB
                  </Typography>
                </>
              ) : (
                <Box sx={{ position: 'relative', width: '100%', height: '100%' }}>
                  <img 
                    src={uploadedImage} 
                    alt="Preview" 
                    style={{ 
                      maxWidth: '100%', 
                      maxHeight: '100%', 
                      objectFit: 'contain', 
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)'
                    }} 
                  />
                  <Box sx={{ 
                    position: 'absolute', 
                    top: 5, 
                    right: 5, 
                    backgroundColor: 'rgba(0,0,0,0.7)', 
                    color: 'white',
                    borderRadius: '50%',
                    width: 30,
                    height: 30,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    cursor: 'pointer'
                  }} onClick={(e) => {
                    e.stopPropagation();
                    handleClearUploadedImage();
                  }}>
                    <CloseIcon fontSize="small" />
                  </Box>
                </Box>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/gif,image/webp"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
            </Box>
            
            {uploadError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {uploadError}
              </Alert>
            )}
          </Box>
          
        </DialogContent>
        <DialogActions>
          <Button onClick={onCloseUploadDialog}>Cancel</Button>
          <Button 
            onClick={handleAnalyzeUploadedImage} 
            variant="contained" 
            disabled={!uploadedImageFile || uploadLoading}
            startIcon={uploadLoading ? <CircularProgress size={20} /> : null}
          >
            {uploadLoading ? 'Analysis...' : 'Analyze'}
          </Button>
        </DialogActions>
      </Dialog>
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
        action={
          <IconButton size="small" color="inherit" onClick={handleSnackbarClose}>
            <CloseIcon fontSize="small" />
          </IconButton>
        }
      />
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs>
                <TextField
                  fullWidth
                  size="small"
                  placeholder="Search by title or artist..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <Box component="span" sx={{ color: 'text.secondary', mr: 1 }}>🔍</Box>
                    ),
                  }}
                />
              </Grid>
              <Grid item>
                <ToggleButtonGroup
                  value={displayMode}
                  exclusive
                  onChange={(_, newValue) => newValue && setDisplayMode(newValue)}
                  size="small"
                >
                  <ToggleButton value="grid" aria-label="grid view">
                    <ViewModuleIcon />
                  </ToggleButton>
                  <ToggleButton value="list" aria-label="list view">
                    <ViewListIcon />
                  </ToggleButton>
                </ToggleButtonGroup>
              </Grid>
            </Grid>
          </Paper>
          
          <Typography variant="h6" gutterBottom>
            Select Artwork to Analyze
          </Typography>
          
          {displayMode === 'grid' ? (
            <Grid container spacing={2}>
              {filteredArtworks.map((artwork, index) => (
                <Grid item xs={6} sm={4} md={6} lg={4} key={`artwork-grid-${artwork.objectID || artwork.id || index}`}>
              <Card 
                    onClick={() => artwork && handleSelectArtwork(artwork)}
                sx={{ 
                  cursor: 'pointer',
                      border: selectedArtwork?.objectID === (artwork?.objectID || artwork?.id) 
                        ? '2px solid #3f51b5' 
                        : 'none',
                      transition: 'transform 0.2s',
                      '&:hover': {
                        transform: 'scale(1.05)'
                      }
                    }}
              >
                <CardMedia
                  component="img"
                  height="140"
                      image={getImageUrl(artwork)}
                      alt={artwork?.title || 'Artwork'}
                      sx={{ objectFit: 'contain', bgcolor: '#f5f5f5' }}
                      onError={(e) => {
                        console.error("Error loading image for artwork:", artwork);
                        e.target.onerror = null; // prevents looping
                        e.target.src = '/placeholder-image.png'; // Or some placeholder
                        e.target.alt = 'Error loading image';
                      }}
                    />
                    <CardContent sx={{ p: 1, '&:last-child': { pb: 1 } }}>
                      <Typography variant="caption" noWrap>
                        {artwork?.title || 'Без названия'}
                  </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <List sx={{ 
              bgcolor: 'background.paper',
              border: '1px solid #e0e0e0',
              borderRadius: 1,
              maxHeight: '70vh',
              overflow: 'auto'
            }}>
              {filteredArtworks.map((artwork, index) => (
                <React.Fragment key={`artwork-list-${artwork.objectID || artwork.id || index}`}>
                  <ListItem 
                    button 
                    onClick={() => artwork && handleSelectArtwork(artwork)}
                    selected={selectedArtwork?.objectID === (artwork?.objectID || artwork?.id)}
                    sx={{ 
                      borderLeft: selectedArtwork?.objectID === (artwork?.objectID || artwork?.id) 
                        ? '4px solid #3f51b5' 
                        : '4px solid transparent' 
                    }}
                  >
                    <ListItemAvatar>
                      <Avatar 
                        variant="rounded" 
                        src={getImageUrl(artwork)} 
                        alt={artwork?.title || 'Artwork'}
                        sx={{ width: 60, height: 60, mr: 1 }}
                        imgProps={{
                          onError: (e) => {
                            console.error("Error loading thumbnail for artwork list:", getImageUrl(artwork));
                            e.target.onerror = null; // prevents looping
                            e.target.src = '/placeholder-image.png'; // Or some placeholder
                            e.target.alt = 'Error loading image';
                          }
                        }}
                      />
                    </ListItemAvatar>
                    <ListItemText
                      primary={artwork?.title || 'No title'}
                      secondary={artwork?.artistDisplayName || 'Unknown artist'}
                    />
                  </ListItem>
                  <Divider variant="inset" component="li" />
                </React.Fragment>
              ))}
            </List>
          )}
        </Grid>
        
        <Grid item xs={12} md={8}>
          {selectedArtwork ? (
            <Box>
              <Typography variant="h6" gutterBottom>
                {artworkTitle}
              </Typography>
              
              {/* --- ADDED LOG 3 --- */}
              {console.log("StyleAnalysis: Rendering main image. selectedArtwork:", selectedArtwork)}
              {console.log("StyleAnalysis: Result of getImageUrl(selectedArtwork):", getImageUrl(selectedArtwork)?.substring(0, 100) + '...')}
              {/* --- END ADDED LOG 3 --- */}

              {/* Main Image Display & Details */}
              <Grid container spacing={2} mb={2}>
                <Grid item xs={12} md={6}>
                  <Box 
                    sx={{ 
                      height: 300, 
                      display: 'flex',
                      justifyContent: 'center',
                      alignItems: 'center',
                      backgroundColor: '#f5f5f5',
                      borderRadius: 2,
                      boxShadow: 1,
                      overflow: 'hidden'
                    }}
                  >
                    {/* Use getImageUrl for the main display */}
                    <img 
                      src={getImageUrl(selectedArtwork)} 
                      alt={selectedArtwork?.title || 'Artwork'}
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '100%', 
                        objectFit: 'contain'
                      }}
                      onError={(e) => {
                        console.error("Error loading selected artwork image in StyleAnalysis main panel:", getImageUrl(selectedArtwork));
                        e.target.onerror = null; // prevents looping
                        e.target.src = '/placeholder-image.png'; // Or some placeholder
                        e.target.alt = 'Error loading image';
                      }}
                    />
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%', boxShadow: 1 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {selectedArtwork?.title || 'No title'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        <strong>Artist:</strong> {selectedArtwork?.artistDisplayName || 'Unknown'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        <strong>Period:</strong> {selectedArtwork?.objectDate || 'Unknown'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        <strong>Material:</strong> {selectedArtwork?.medium || 'Unknown'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        <strong>Category:</strong> {selectedArtwork?.department || 'Unknown'}
                      </Typography>
                      {selectedArtwork?.objectURL && (
                        <Typography variant="body2">
                          <a href={selectedArtwork.objectURL} target="_blank" rel="noopener noreferrer">
                            View on Metropolitan Museum of Art website
                          </a>
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
              
              {/* Summary panel for the artwork analysis */}
              {!loading && selectedArtwork && (styleData.labels || colorData.labels || compositionData.labels) && (
                <Paper sx={{ p: 2, mb: 3, borderLeft: '4px solid #3f51b5' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                      One-view analysis
                    </Typography>
                    <Button 
                        variant="outlined"
                      color="primary" 
                      size="small"
                      onClick={() => setShowDetailedExplanation(!showDetailedExplanation)}
                      startIcon={showDetailedExplanation ? <VisibilityOffIcon /> : <VisibilityIcon />}
                    >
                      {showDetailedExplanation ? 'Hide details' : 'Show detailed explanations'}
                    </Button>
                  </Box>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Typography variant="subtitle2" color="primary">Style</Typography>
                      {styleData.labels && styleData.datasets && styleData.datasets[0]?.data && (
                        <Typography variant="body2">
                          High scores: {styleData.labels
                            .map((label, i) => ({ label, value: styleData.datasets[0].data[i] }))
                            .filter(item => item.value >= 0.7)
                            .map(item => item.label)
                            .join(', ') || 'not detected'}
                        </Typography>
                      )}
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Typography variant="subtitle2" color="primary">Color</Typography>
                      {colorProperties && colorProperties.length > 0 && (
                        <Typography variant="body2">
                          Dominant color: {colorProperties[0]?.name || 'not defined'}
                          {colorHarmony && `, harmony type: ${colorHarmony.type || 'not defined'}`}
                        </Typography>
                      )}
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Typography variant="subtitle2" color="primary">Composition</Typography>
                      {compositionData.labels && compositionData.datasets && compositionData.datasets[0]?.data && (
                        <Typography variant="body2">
                          Key elements: {compositionData.labels
                            .map((label, i) => ({ label, value: compositionData.datasets[0].data[i] }))
                            .filter(item => item.value >= 0.7)
                            .map(item => item.label)
                            .join(', ') || 'not detected'}
                        </Typography>
                      )}
                    </Grid>
                  </Grid>
                </Paper>
              )}
              
              {/* Подробное объяснение всех значений анализа */}
              {showDetailedExplanation && !loading && selectedArtwork && (styleData.labels || colorData.labels || compositionData.labels) && (
                <Paper sx={{ p: 3, mb: 3, borderRadius: 2, boxShadow: 3, bgcolor: '#f9f9fa' }}>
                  <Typography variant="h5" gutterBottom sx={{ color: '#3f51b5', fontWeight: 'medium' }}>
                    Detailed explanation of analysis results
                  </Typography>
                  
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Box sx={{ 
                        height: 250, 
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        backgroundColor: '#fff',
                        borderRadius: 2,
                        boxShadow: 1,
                        overflow: 'hidden',
                        border: '1px solid #eaeaea'
                      }}>
                        <img 
                          src={getImageUrl(selectedArtwork)} 
                          alt={selectedArtwork.title}
                          style={{ 
                            maxWidth: '100%', 
                            maxHeight: '100%', 
                            objectFit: 'contain'
                          }} 
                          onError={(e) => {
                            console.error("Image failed to load:", getImageUrl(selectedArtwork));
                            e.target.onerror = null; // prevents looping
                            e.target.src = '/placeholder-image.png'; // Or some placeholder
                            e.target.alt = 'Error loading image';
                          }}
                        />
                      </Box>
                      
                      <Typography variant="h6" sx={{ mt: 2, mb: 1, fontWeight: 'medium' }}>
                        {selectedArtwork.title || 'Untitled'}
                      </Typography>
                      <Typography variant="body2" sx={{ fontStyle: 'italic', mb: 2 }}>
                        {selectedArtwork.artistDisplayName}, {selectedArtwork.objectDate}
                      </Typography>
                      
                      <Typography variant="body2" paragraph sx={{ bgcolor: '#f0f4ff', p: 2, borderRadius: 1 }}>
                        The analysis is performed using computer vision and machine learning algorithms. 
                        The system evaluates stylistic features, color composition, and structural organization of elements in the image, 
                        providing quantitative estimates of various visual perception parameters.
                      </Typography>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom sx={{ borderBottom: '2px solid #eaeaea', pb: 1 }}>
                        Key analysis results
                      </Typography>
                      
                      {/* Стилистические характеристики */}
                      <Typography variant="subtitle1" sx={{ mt: 2, fontWeight: 'medium', color: '#3f51b5' }}>
                        Stylistic characteristics:
                      </Typography>
                      {styleData.labels && styleData.datasets && styleData.datasets[0]?.data && (
                        <Box sx={{ ml: 2 }}>
                          {styleData.labels.map((label, i) => (
                            <Box key={i} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 'medium', minWidth: 100 }}>
                                {label}:
                              </Typography>
                              <Box sx={{ flex: 1, mx: 1 }}>
                                <LinearProgress 
                                  variant="determinate" 
                                  value={styleData.datasets[0].data[i] * 100} 
                                  sx={{ 
                                    height: 8, 
                                    borderRadius: 1,
                                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                    '& .MuiLinearProgress-bar': {
                                      backgroundColor: 'rgb(54, 162, 235)'
                                    }
                                  }}
                                />
                              </Box>
                              <Typography variant="body2" sx={{ ml: 1, minWidth: 40 }}>
                                {(styleData.datasets[0].data[i] * 100).toFixed(0)}%
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      )}
                      
                      <Typography variant="subtitle1" sx={{ mt: 3, fontWeight: 'medium', color: '#3f51b5' }}>
                        Color characteristics:
                      </Typography>
                      {colorProperties && colorProperties.length > 0 && (
                        <Box sx={{ ml: 2 }}>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 2 }}>
                            {colorProperties.slice(0, 5).map((color, i) => (
                              <Box key={i} sx={{ 
                                width: 24, 
                                height: 24, 
                                backgroundColor: color.hex,
                                borderRadius: '50%',
                                mr: 1,
                                mb: 1,
                                border: '1px solid #ddd'
                              }} title={`${color.name} (${color.percentage}%)`} />
                            ))}
                  </Box>
                          <Typography variant="body2" paragraph>
                            Dominant color: <b>{colorProperties[0]?.name}</b> ({colorProperties[0]?.percentage}%)
                          </Typography>
                          {colorHarmony && (
                            <Typography variant="body2" paragraph>
                              Harmony type: <b>{colorHarmony.type}</b> - {colorHarmony.description}
                            </Typography>
                          )}
                </Box>
                      )}
                      
                      <Typography variant="subtitle1" sx={{ mt: 3, fontWeight: 'medium', color: '#3f51b5' }}>
                        Composition characteristics:
                      </Typography>
                      {compositionData.labels && compositionData.datasets && compositionData.datasets[0]?.data && (
                        <Box sx={{ ml: 2 }}>
                          {compositionData.labels.map((label, i) => (
                            <Box key={i} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 'medium', minWidth: 120 }}>
                                {label}:
                              </Typography>
                              <Box sx={{ flex: 1, mx: 1 }}>
                                <LinearProgress 
                                  variant="determinate" 
                                  value={compositionData.datasets[0].data[i] * 100} 
                                  sx={{ 
                                    height: 8, 
                                    borderRadius: 1,
                                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                    '& .MuiLinearProgress-bar': {
                                      backgroundColor: compositionData.datasets[0].backgroundColor[i] || 'rgba(255, 99, 132, 0.5)'
                                    }
                                  }}
                                />
                              </Box>
                              <Typography variant="body2" sx={{ ml: 1, minWidth: 40 }}>
                                {(compositionData.datasets[0].data[i] * 100).toFixed(0)}%
                              </Typography>
                            </Box>
            ))}
          </Box>
                      )}
        </Grid>
                  </Grid>
                </Paper>
              )}
              
              <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs value={selectedTab} onChange={(event, newValue) => setSelectedTab(newValue)} aria-label="analysis tabs">
                  <Tab label="Style" />
                  <Tab label="Color" />
                  <Tab label="Composition" />
                </Tabs>
                
              </Box>
              
          {/* --- ADD Image Display Below Tabs --- */}
          {selectedArtwork && !loading && (
                <Paper sx={{ 
                    p: 1, 
                    mb: 3, 
                    mt: 2, 
                    display: 'flex', 
                    justifyContent: 'center',
                    alignItems: 'center',
                    maxHeight: '300px', // Limit height 
                    overflow: 'hidden',
                    backgroundColor: '#f5f5f5' // Background for consistency
                }}> 
                  <img 
                    src={getImageUrl(selectedArtwork)}
                    alt={`Analyzed image: ${selectedArtwork.title || 'Untitled'}`}
                    style={{ 
                        maxWidth: '100%', 
                        maxHeight: '280px', // Slightly less than container 
                        objectFit: 'contain',
                        display: 'block' // Helps with potential spacing issues
                    }}
                    onError={(e) => {
                        console.error("Error loading selected artwork image below tabs:", getImageUrl(selectedArtwork));
                        e.target.onerror = null; // prevents looping
                        e.target.style.display = 'none'; // Hide if error
                    }}
                  />
                </Paper>
              )}
          {/* --- END ADD Image Display --- */}

          {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            <>
                  {selectedTab === 0 && styleData && (
                <Box>
                      {styleData.labels && styleData.labels.length > 0 && (
                        <>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6">Stylistic profile</Typography>
                            {isStyleFallback && (
                              <Alert severity="warning" sx={{ py: 0 }}>
                                Displayed backup data
                              </Alert>
                            )}
                          </Box>
                          
                          <Paper sx={{ p: 2, mb: 3, backgroundColor: 'rgba(245, 245, 250, 0.8)' }}>
                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'medium' }}>
                              <InfoIcon sx={{ verticalAlign: 'middle', mr: 1, color: 'primary.main' }} />
                              About stylistic analysis
                  </Typography>
                  <Typography variant="body2" paragraph>
                              Stylistic analysis measures the key visual characteristics of the work, reflecting its artistic style.
                              Each parameter is evaluated in the range from 0 to 1, where higher values indicate a stronger manifestation of the given characteristic.
                  </Typography>
                            <Typography variant="body2">
                              The hexagon chart allows visually assessing the "stylistic profile" of the work and comparing it with the profiles of other works.
                              Different artistic directions often have characteristic stylistic profiles - for example, impressionism is usually characterized by high brightness and texture.
                  </Typography>
                          </Paper>
                          
                          <Grid container spacing={2}>
                            <Grid item xs={12} md={6}>
                              <Box sx={{ height: 350 }}>
                    <Radar 
                                  data={styleData}
                      options={{
                        scales: {
                          r: {
                            beginAtZero: true,
                            min: 0,
                            max: 1,
                            ticks: {
                              stepSize: 0.2
                            }
                          }
                        },
                        plugins: {
                          legend: {
                            position: 'bottom'
                          }
                        },
                        elements: {
                          line: {
                            tension: 0.2
                          }
                                    },
                                    responsive: true,
                                    maintainAspectRatio: true
                      }}
                    />
                  </Box>
                            </Grid>
                            
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle1" gutterBottom>
                                Interpretation of parameters:
                  </Typography>
                              
                              <TableContainer>
                                <Table size="small">
                                  <TableBody>
                                    {styleData.labels && styleData.labels.map((label, idx) => (
                                      <TableRow key={idx}>
                                        <TableCell><strong>{label}</strong></TableCell>
                                        <TableCell>{styleData.datasets[0]?.data[idx]?.toFixed(2) || 'N/A'}</TableCell>
                                        <TableCell>{styleDescriptions[label] || 'Нет описания'}</TableCell>
                                      </TableRow>
                                    ))}
                                  </TableBody>
                                </Table>
                              </TableContainer>
                            </Grid>
                          </Grid>
                          
                          {/* Add the explanation component */}
                          <AnalysisExplanations type="style" />
                        </>
                      )}
                </Box>
              )}
              
                  {selectedTab === 1 && colorData && (
                <Box>
                      {colorData.labels && colorData.labels.length > 0 ? (
                        <>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6">Color analysis</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              {isColorFallback && (
                                <Alert severity="warning" sx={{ py: 0, mr: 1 }}>
                                  Backup data
                                </Alert>
                              )}
                              {colorHarmony && colorHarmony.type === 'Monochrome' && (
                                <Chip 
                                  label="Monochrome" 
                                  color="secondary" 
                                  size="small" 
                                  sx={{ mr: 1 }} 
                                />
                              )}
                            </Box>
                          </Box>
                          
                          <Grid container spacing={2} mb={3}>
                            <Grid item xs={12} md={4}>
                              <ImagePreview 
                                artwork={selectedArtwork} 
                                imageUrl={getImageUrl(selectedArtwork)} 
                              />
                            </Grid>
                            <Grid item xs={12} md={8}>
                              <Paper sx={{ p: 2, mb: 2, backgroundColor: 'rgba(245, 245, 250, 0.8)' }}>
                                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'medium' }}>
                                  <InfoIcon sx={{ verticalAlign: 'middle', mr: 1, color: 'primary.main' }} />
                                  About color analysis
                  </Typography>
                  <Typography variant="body2" paragraph>
                                  Color analysis determines the dominant colors of the work, their proportions and relationships. 
                                  The algorithm identifies the main color groups using clustering and determines their characteristics: 
                                  name, percentage, brightness, saturation and emotional perception.
                  </Typography>
                                <Typography variant="body2">
                                  {colorHarmony && colorHarmony.type === 'Monochrome' ? 
                                    "Monochrome works use one color or shades of gray. Such images often create a refined, restrained and elegant image, allowing the viewer to focus on the form, texture and composition." :
                                    "Color harmony evaluates how harmoniously colors combine in the work. Different types of harmony (monochrome, complementary, triadic, etc.) create different emotional effects."}
                                </Typography>
                              </Paper>
                            </Grid>
                          </Grid>
                          
                          {colorHarmony && colorHarmony.type === 'Monochrome' ? (
                            <MonochromeVisualization 
                              colorData={colorData} 
                              colorProperties={colorProperties} 
                              colorHarmony={colorHarmony} 
                            />
                          ) : (
                            <Grid container spacing={2}>
                              <Grid item xs={12} md={6}>
                                <Box sx={{ height: 320, display: 'flex', justifyContent: 'center' }}>
                                  <Box sx={{ width: '100%', maxWidth: 320 }}>
                                    {colorData && colorData.datasets && colorData.datasets.length > 0 && colorData.datasets[0].data && (
                      <Doughnut 
                                        data={colorData}
                        options={{
                          plugins: {
                            legend: {
                                              position: 'bottom',
                                              display: colorData.labels.length <= 8 // Скрываем легенду, если слишком много меток
                                            },
                                            tooltip: {
                                              callbacks: {
                                                label: function(context) {
                                                  const label = context.label || '';
                                                  const value = context.raw || 0;
                                                  return `${label}: ${value.toFixed(1)}%`;
                                                }
                                              }
                                            }
                                          },
                                          cutout: '50%',
                                          responsive: true,
                                          maintainAspectRatio: true
                                        }}
                                      />
                                    )}
                    </Box>
                  </Box>
                                
                                {colorHarmony && (
                                  <Paper sx={{ mt: 2, p: 2, textAlign: 'center', px: 2 }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                      Color harmony: {colorHarmony.type}
                  </Typography>
                                    <Typography variant="body2">
                                      {colorHarmony.description}
                                    </Typography>
                                    {colorHarmony.score !== undefined && (
                                      <Typography variant="body2" sx={{ mt: 1 }}>
                                        Harmony score: {(colorHarmony.score * 100).toFixed(0)}%
                                      </Typography>
                                    )}
                                  </Paper>
                                )}
                              </Grid>
                              
                              <Grid item xs={12} md={6}>
                                <Typography variant="subtitle1" gutterBottom>
                                  Dominant colors:
                                </Typography>
                                
                                {colorProperties && colorProperties.length > 0 ? (
                                  <TableContainer sx={{ maxHeight: 320, overflow: 'auto' }}>
                                    <Table size="small" stickyHeader>
                                      <TableHead>
                                        <TableRow>
                                          <TableCell>Color</TableCell>
                                          <TableCell>Name</TableCell>
                                          <TableCell>%</TableCell>
                                          <TableCell>Brightness</TableCell>
                                          <TableCell>Saturation</TableCell>
                                        </TableRow>
                                      </TableHead>
                                      <TableBody>
                                        {/* Iterates over colorProperties which now holds the correct array */}
                                        {colorProperties.map((prop, idx) => (
                                          <TableRow key={idx}>
                                            <TableCell>
                                              {/* Use prop.hex for background color */}
                                              <Box sx={{
                                                width: 20,
                                                height: 20,
                                                backgroundColor: prop.hex || '#CCCCCC', // Fallback added
                                                border: '1px solid #ddd',
                                                borderRadius: '2px'
                                              }} />
                                            </TableCell>
                                            {/* Use prop.name, prop.percentage etc. */}
                                            <TableCell>{prop.name || 'Unknown'}</TableCell>
                                            <TableCell>{prop.percentage !== undefined ? prop.percentage.toFixed(1) : 0}%</TableCell>
                                            {/* Brightness/Saturation are now pre-formatted strings or '-' */}
                                            <TableCell>{prop.brightness !== undefined ? prop.brightness : '-'}</TableCell>
                                            <TableCell>{prop.saturation !== undefined ? prop.saturation : '-'}</TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                  </TableContainer>
                                ) : (
                                  <Alert severity="info">No data on dominant colors</Alert>
                                )}
                                
                                {/* Emotional impact panel */}
                                {colorHarmony?.emotional_impact && colorHarmony.emotional_impact.length > 0 && (
                                  <Paper sx={{ mt: 2, p: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom>
                                      Emotional impact:
                                    </Typography>
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                      {colorHarmony.emotional_impact.map((emotion, idx) => (
                                        <Chip 
                                          key={idx}
                                          label={emotion}
                                          size="small"
                                          color="primary"
                                          variant="outlined"
                                        />
                                      ))}
                </Box>
                                  </Paper>
                                )}
                              </Grid>
                            </Grid>
                          )}
                          
                          {/* Add the explanation component */}
                          <AnalysisExplanations type="color" />
                        </>
                      ) : (
                        <Alert severity="info">
                          No data for color analysis. Please select an image for analysis.
                        </Alert>
                      )}
                    </Box>
                  )}
                  
                  {selectedTab === 2 && compositionData && (
                <Box>
                      {compositionData.labels && compositionData.labels.length > 0 && (
                        <>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h6">Composition analysis</Typography>
                            {isCompositionFallback && (
                              <Alert severity="warning" sx={{ py: 0 }}>
                                Displayed backup data
                              </Alert>
                            )}
                          </Box>
                          
                          <Paper sx={{ p: 2, mb: 3, backgroundColor: 'rgba(245, 245, 250, 0.8)' }}>
                            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'medium' }}>
                              <InfoIcon sx={{ verticalAlign: 'middle', mr: 1, color: 'primary.main' }} />
                              About composition analysis
                  </Typography>
                  <Typography variant="body2" paragraph>
                              Composition analysis studies the structural organization of elements in an artwork.
                              The algorithm evaluates various aspects of composition, including symmetry, adherence to the rule of thirds,
                              presence of leading lines, depth, framing and overall balance of the composition.
                  </Typography>
                            <Typography variant="body2">
                              Composition is a fundamental aspect of visual art, determining how the viewer perceives the work.
                              A strong composition guides the eye, creates a sense of order and harmony, and emphasizes the main elements of the work.
                              The circular chart shows the relative strength of different composition principles in the analyzed artwork.
                  </Typography>
                          </Paper>
                          
                          <Grid container spacing={2}>
                            <Grid item xs={12} md={6}>
                              <Box sx={{ height: 350 }}>
                    <PolarArea 
                                  data={compositionData}
                      options={{
                        scales: {
                          r: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                              display: false
                            }
                          }
                        },
                        plugins: {
                          legend: {
                            position: 'right'
                          }
                                    },
                                    responsive: true,
                                    maintainAspectRatio: true
                      }}
                    />
                  </Box>
                            </Grid>
                            
                            <Grid item xs={12} md={6}>
                              <Typography variant="subtitle1" gutterBottom>
                                Composition parameters:
                  </Typography>
                              
                              <TableContainer sx={{ maxHeight: 350, overflow: 'auto' }}>
                                <Table size="small" stickyHeader>
                                  <TableHead>
                                    <TableRow>
                                      <TableCell>Parameter</TableCell>
                                      <TableCell>Value</TableCell>
                                      <TableCell>Description</TableCell>
                                    </TableRow>
                                  </TableHead>
                                  <TableBody>
                                    {compositionData.labels && compositionData.labels.map((label, idx) => (
                                      <TableRow key={idx}>
                                        <TableCell><strong>{label}</strong></TableCell>
                                        <TableCell>{compositionData.datasets[0]?.data[idx]?.toFixed(2) || 'N/A'}</TableCell>
                                        <TableCell>{compositionDescriptions[label] || 'No description'}</TableCell>
                                      </TableRow>
                                    ))}
                                  </TableBody>
                                </Table>
                              </TableContainer>
                            </Grid>
                          </Grid>
                          
                          {/* Add the explanation component */}
                          <AnalysisExplanations type="composition" />
                        </>
                      )}
                </Box>
              )}
            </>
              )}
            </Box>
          ) : (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '60vh',
                backgroundColor: '#f5f5f5',
                borderRadius: 2,
                p: 4
              }}
            >
              <LightbulbIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" align="center">
                Select an artwork to analyze its style, color, and composition
              </Typography>
              <Button 
                variant="outlined" 
                onClick={refreshArtworks}
                startIcon={<RefreshIcon />}
                sx={{ mt: 2 }}
              >
                Show different artworks
              </Button>
            </Box>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default StyleAnalysis;


