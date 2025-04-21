import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Paper, Typography, Box, Button, Card, CardMedia, 
  Divider, IconButton, Tooltip, Stack, CardContent, Skeleton
} from '@mui/material';
import { 
  Visibility, Palette, BubbleChart, 
  AccountTree, Close, CloudUpload, Info,
  GridOn
} from '@mui/icons-material';

/**
 * Component for displaying the currently selected image globally across the app
 */
const GlobalImagePanel = ({ 
  selectedArtwork, 
  onClose, 
  onUploadClick,
  baseUrl = "http://localhost:5001"
}) => {
  const navigate = useNavigate();
  
  // Log the received prop at the very beginning
  console.log("GlobalImagePanel received prop:", selectedArtwork);

  if (!selectedArtwork || typeof selectedArtwork !== 'object') {
    console.log("GlobalImagePanel: selectedArtwork is null, undefined, or not an object.");
    return (
      <Paper elevation={3} sx={{ height: '100%', p: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Current image
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
          <Typography variant="body2" color="text.secondary" align="center">
            Select an image for analysis
          </Typography>
        </Box>
      </Paper>
    );
  }
  
  // Now we know selectedArtwork is an object, but log its keys for structure check
  console.log("GlobalImagePanel: Displaying data. Keys:", Object.keys(selectedArtwork));

  const getImageUrl = () => {
    console.log("GlobalImagePanel: Forming URL image from:", selectedArtwork);

    if (selectedArtwork?.type === 'uploaded' && typeof selectedArtwork.primaryImageSmall === 'string' && selectedArtwork.primaryImageSmall.startsWith('data:image')) {
      console.log("GlobalImagePanel: Using base64 primaryImageSmall for uploaded:", selectedArtwork.primaryImageSmall.substring(0, 100) + '...');
      return selectedArtwork.primaryImageSmall;
    }

    if (selectedArtwork?.cachedImageUrl) {
       console.log("GlobalImagePanel: Using cachedImageUrl:", selectedArtwork.cachedImageUrl.substring(0,100) + '...');
       if (!selectedArtwork.cachedImageUrl.startsWith('data:')) {
         try {
           return `http://localhost:5000/api/proxy_image?url=${encodeURIComponent(selectedArtwork.cachedImageUrl)}`;
         } catch (err) {
            console.error("GlobalImagePanel: Error encoding cachedImageUrl for proxy:", err);
            return '/placeholder-image.png'; // Fallback
         }
       }
       return selectedArtwork.cachedImageUrl; // Return data URI from cache directly
    }

    // 3. Check primaryImage (usually external URL)
    if (selectedArtwork?.primaryImage) {
      console.log("GlobalImagePanel: Using primaryImage (through proxy):", selectedArtwork.primaryImage);
       try {
         // --- FIX: Added check to prevent proxying data URIs here too ---
         if (typeof selectedArtwork.primaryImage === 'string' && selectedArtwork.primaryImage.startsWith('data:')) {
            console.warn("GlobalImagePanel: primaryImage contains data URI, should not be proxied!");
            return selectedArtwork.primaryImage;
         }
         return `http://localhost:5000/api/proxy_image?url=${encodeURIComponent(selectedArtwork.primaryImage)}`;
       } catch (err) {
         console.error("GlobalImagePanel: Error encoding primaryImage for proxy:", err);
         return '/placeholder-image.png'; // Fallback
       }
    }

    // 4. Check primaryImageSmall (usually external URL)
    if (selectedArtwork?.primaryImageSmall) {
       console.log("GlobalImagePanel: Using primaryImageSmall (through proxy):", selectedArtwork.primaryImageSmall);
       try {
         // --- FIX: Added check to prevent proxying data URIs here too ---
          if (typeof selectedArtwork.primaryImageSmall === 'string' && selectedArtwork.primaryImageSmall.startsWith('data:')) {
            console.warn("GlobalImagePanel: primaryImageSmall contains data URI, should not be proxied!");
            return selectedArtwork.primaryImageSmall;
         }
         return `http://localhost:5000/api/proxy_image?url=${encodeURIComponent(selectedArtwork.primaryImageSmall)}`;
       } catch (err) {
         console.error("GlobalImagePanel: Error encoding primaryImageSmall for proxy:", err);
         return '/placeholder-image.png'; // Fallback
       }
    }
    
    // 5. If nothing found
    console.log("GlobalImagePanel: No suitable URL found, using placeholder");
    return '/placeholder-image.png';
  };

  const imageUrl = getImageUrl();

  // Transition to other pages with this image
  const navigateToAnalysis = () => {
    navigate('/analysis');
  };

  const navigateToAttention = () => {
    navigate('/attention');
  };

  const navigateToEmbedding = () => {
    navigate('/embeddings');
  };

  const navigateToSimilarity = () => {
    navigate('/similarity');
  };

  const navigateToHeatmap = () => {
    navigate('/heatmap');
  };

  return (
    <Paper elevation={3} sx={{ height: '100%', p: 2 }}>
      <Typography variant="subtitle1" gutterBottom>
        Current image
      </Typography>
      <Divider sx={{ mb: 2 }} />
      <Card sx={{ mb: 2 }}>
        <CardMedia
          component="img"
          image={imageUrl || '/placeholder-image.png'}
          alt={selectedArtwork.title || 'Artwork image'}
          sx={{ maxHeight: 300, objectFit: 'contain', bgcolor: '#f5f5f5' }}
          onError={(e) => {
            console.error('Error loading image:', imageUrl);
            e.target.onerror = null;
            e.target.src = '/placeholder-image.png'; // Fallback image
          }}
        />
        <CardContent sx={{ py: 1 }}>
          <Typography variant="body2" color="text.secondary" noWrap>
            {selectedArtwork.title || 'No title'}
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block">
            {selectedArtwork.artistDisplayName || 'Unknown author'}
          </Typography>
          {selectedArtwork.objectDate && (
            <Typography variant="caption" color="text.secondary">
              {selectedArtwork.objectDate}
            </Typography>
          )}
        </CardContent>
      </Card>
      
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        ID: {selectedArtwork.id || selectedArtwork.objectID}
      </Typography>
      
      {selectedArtwork.classification && (
        <Typography variant="body2" color="text.secondary">
          Type: {selectedArtwork.classification}
        </Typography>
      )}

      {selectedArtwork.medium && (
        <Typography variant="body2" color="text.secondary">
          Material: {selectedArtwork.medium}
        </Typography>
      )}
      
      <Divider sx={{ my: 1 }} />
      
      {/* Buttons for image analysis */}
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Analyze in:
        </Typography>
        
        <Stack spacing={1.5} sx={{ mt: 2 }}>
          <Button 
            variant="outlined" 
            startIcon={<Palette />} 
            fullWidth
            size="small"
            onClick={navigateToAnalysis}
          >
            Stylistic analysis
          </Button>
          
          <Button 
            variant="outlined" 
            startIcon={<Visibility />} 
            fullWidth
            size="small"
            onClick={navigateToAttention}
          >
            Attention maps
          </Button>
          
          <Button 
            variant="outlined" 
            startIcon={<BubbleChart />} 
            fullWidth
            size="small"
            onClick={navigateToEmbedding}
          >
            Embeddings
          </Button>
          
          <Button 
            variant="outlined" 
            startIcon={<AccountTree />} 
            fullWidth
            size="small"
            onClick={navigateToSimilarity}
          >
            Similarity analysis
          </Button>
          
          <Button 
            variant="outlined" 
            startIcon={<GridOn />} 
            fullWidth
            size="small"
            onClick={navigateToHeatmap}
          >
            Similarity Heatmap
          </Button>
          
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button 
              variant="text" 
              startIcon={<CloudUpload />}
              onClick={onUploadClick}
              size="small"
            >
              Upload another
            </Button>
          </Box>
        </Stack>
      </Box>
      
      <Box sx={{ mt: 'auto', p: 2, backgroundColor: '#f5f5f7', borderTop: '1px solid #e0e0e0' }}>
        <Typography variant="caption" color="text.secondary" display="block" textAlign="center">
          <Info fontSize="inherit" sx={{ mr: 0.5, verticalAlign: 'text-bottom' }} />
          The image is available for use on all pages of the application
        </Typography>
      </Box>
    </Paper>
  );
};

export default GlobalImagePanel; 