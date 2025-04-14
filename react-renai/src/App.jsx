import React, { useState, useEffect } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Container } from '@mui/material';
import Sidebar from './components/Sidebar';
import EmbeddingProjection from './components/EmbeddingProjection';
import AttentionVisualizer from './components/AttentionVisualizer';
import ArtSimilarityGraph from './components/ArtSimilarityGraph';
import SimilarityHeatmap from './components/SimilarityHeatmap';
import StyleAnalysis from './components/StyleAnalysis';
import { processRawEmbeddings } from './utils/dataProcessing';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#f5f5f7',
      paper: '#ffffff'
    }
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    }
  },
  components: {
    MuiPaper: {
      defaultProps: {
        elevation: 2,
      },
      styleOverrides: {
        root: {
          borderRadius: 8,
        }
      }
    }
  }
});

const artworksMockData = [
  {
    id: 1,
    title: "Сцены из древнеегипетского храма",
    era: "Древний Египет",
    style: "Египетская настенная живопись",
    region: "Северная Африка",
    imageUrl: "https://example.com/art1.jpg"
  },
];

function App() {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [embeddings, setEmbeddings] = useState(null);
  const [artworkMetadata, setArtworkMetadata] = useState(artworksMockData);
  const [attentionMaps, setAttentionMaps] = useState({});
  const [gradcamMaps, setGradcamMaps] = useState({});
  
  useEffect(() => {
    const loadEmbeddings = async () => {
      try {
        const rawEmbeddings = [];
        const processed = processRawEmbeddings(rawEmbeddings);
        setEmbeddings(processed);
      } catch (error) {
        console.error("Error loading embeddings:", error);
      }
    };
    
    loadEmbeddings();
    
  }, []);
  
  const handleNavigate = (section) => {
    setActiveSection(section);
  };
  
  const renderContent = () => {
    switch (activeSection) {
      case 'dashboard':
        return (
          <Box p={3}>
            <h1>Панель управления</h1>
            {/* Дашборд с основными метриками и визуализациями */}
          </Box>
        );
      case 'embeddingProjection':
        return <EmbeddingProjection embeddings={embeddings} artworkMetadata={artworkMetadata} />;
      case 'attentionMaps':
      case 'gradcamVisualization':
        return <AttentionVisualizer artworks={artworkMetadata} attentionMaps={attentionMaps} gradcamMaps={gradcamMaps} />;
      case 'networkGraph':
        return <ArtSimilarityGraph embeddings={embeddings} artworkMetadata={artworkMetadata} />;
      case 'heatmap':
        return <SimilarityHeatmap embeddings={embeddings} artworkMetadata={artworkMetadata} />;
      case 'styleAnalysis':
        return <StyleAnalysis artworks={artworkMetadata} embeddings={embeddings} />;
      default:
        return (
          <Box p={3}>
            <h2>Выберите раздел из меню</h2>
          </Box>
        );
    }
  };
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', minHeight: '100vh' }}>
        <Sidebar onNavigate={handleNavigate} />
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            width: { sm: `calc(100% - 280px)` },
            ml: { sm: '280px' },
            overflow: 'auto'
          }}
        >
          <Container maxWidth="xl" sx={{ mt: 2, mb: 4 }}>
            {renderContent()}
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
