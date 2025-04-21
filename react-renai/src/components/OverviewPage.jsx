import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  Box, 
  Container, 
  Typography, 
  Grid, 
  Card, 
  CardActionArea, 
  CardContent, 
  CardMedia 
} from '@mui/material';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import HubIcon from '@mui/icons-material/Hub';
import SchemaIcon from '@mui/icons-material/Schema';
import VisibilityIcon from '@mui/icons-material/Visibility';
import InsightsIcon from '@mui/icons-material/Insights';

const sections = [
  {
    title: 'Embedding Projection',
    description: 'Visualization of multi-dimensional artwork embeddings in 2D space using t-SNE.',
    link: '/embeddings',
    icon: <ScatterPlotIcon fontSize="large" color="primary" />,
    image: '/thumbnails/embeddings_thumbnail.png' // Placeholder - replace with actual image path
  },
  {
    title: 'Artwork Similarity Graph',
    description: 'Study of relationships between similar artworks in the form of an interactive 3D graph.',
    link: '/similarity',
    icon: <HubIcon fontSize="large" color="secondary" />,
    image: '/thumbnails/similarity_graph_thumbnail.png' // Placeholder
  },
  {
    title: 'Heatmap of Similarity',
    description: 'Matrix showing pairwise similarity between all artworks in the collection.',
    link: '/heatmap',
    icon: <SchemaIcon fontSize="large" color="success" />,
    image: '/thumbnails/heatmap_thumbnail.png' // Placeholder
  },
  {
    title: 'Visual Attention',
    description: 'Analysis of image regions that the model pays attention to (Attention Maps, GradCAM).',
    link: '/attention',
    icon: <VisibilityIcon fontSize="large" color="warning" />,
    image: '/thumbnails/attention_thumbnail.png' // Placeholder
  },
  {
    title: 'Style and Composition Analysis',
    description: 'Detailed analysis of stylistic features, color palette, and compositional techniques.',
    link: '/analysis',
    icon: <InsightsIcon fontSize="large" color="info" />,
    image: '/thumbnails/analysis_thumbnail.png' // Placeholder
  }
];

const OverviewPage = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ textAlign: 'center', mb: 4 }}>
        RenAI: Art Research with AI
      </Typography>
      <Typography variant="h6" component="p" sx={{ textAlign: 'center', color: 'text.secondary', mb: 6 }}>
        Welcome! This tool uses artificial intelligence to analyze and visualize collections of artworks. Explore relationships, styles, and model attention.
      </Typography>

      <Grid container spacing={4} justifyContent="center">
        {sections.map((section) => (
          <Grid item key={section.title} xs={12} sm={6} md={4}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardActionArea component={RouterLink} to={section.link} sx={{ flexGrow: 1 }}>
                {/* Optional: Add CardMedia for visual thumbnails if available 
                <CardMedia
                  component="img"
                  height="140"
                  image={section.image || '/placeholder.png'} // Use a default placeholder if specific image is missing
                  alt={section.title}
                  sx={{ objectFit: 'cover' }} // Ensures image covers the area
                />
                */}
                 <Box sx={{ display: 'flex', justifyContent: 'center', pt: 3, pb: 1 }}>
                   {section.icon}
                 </Box>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography gutterBottom variant="h6" component="div">
                    {section.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {section.description}
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default OverviewPage; 